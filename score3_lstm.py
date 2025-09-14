#!/usr/bin/env python3
import argparse, io, math, sys, warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Model ----------------

class AWDCharLSTM(nn.Module):
    def __init__(self, vocab_size=26, emb=512, hidden=512, layers=3,
                 p_in=0.0, p_h=0.0, p_out=0.0, tie_weights=False):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, emb)
        self.drop_in  = nn.Dropout(p_in)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=p_h)
        self.drop_out = nn.Dropout(p_out)
        self.decoder = nn.Linear(hidden, vocab_size, bias=False)
        if tie_weights:
            assert emb == hidden, "tie_weights requires emb==hidden"
            self.decoder.weight = self.encoder.weight
        self.vocab_size = vocab_size
        self.emb_dim = emb
        self.hidden_dim = hidden
        self.layers = layers

    def forward(self, x, h=None):
        x = self.drop_in(self.encoder(x))
        out, h = self.lstm(x, h)
        out = self.drop_out(out)
        logits = self.decoder(out)
        return logits, h

# ---------------- Utilities ----------------

def _alphabet_for_vocab(vsz: int) -> str:
    if vsz == 26:
        return "abcdefghijklmnopqrstuvwxyz"
    elif vsz == 27:
        return "#abcdefghijklmnopqrstuvwxyz"
    else:
        raise ValueError(f"Unsupported vocab size {vsz} (expected 26 or 27).")

def _clean_for_alphabet(s: str, alphabet: str) -> str:
    s = (s or "").lower()
    allow = set(alphabet)
    return "".join(ch for ch in s if ch in allow)

def _infer_layers_from_state(sd) -> int:
    layers = set()
    for k in sd.keys():
        if k.startswith("lstm.weight_ih_l"):
            try:
                idx = int(k.split("lstm.weight_ih_l", 1)[1].split('.')[0])
                layers.add(idx)
            except Exception:
                pass
    if layers:
        return max(layers) + 1
    for k in sd.keys():
        if k.startswith("lstm.weight_hh_l"):
            try:
                idx = int(k.split("lstm.weight_hh_l", 1)[1].split('.')[0])
                layers.add(idx)
            except Exception:
                pass
    if layers:
        return max(layers) + 1
    return 3

def load_model(ckpt_path: str, device: str = "cuda"):
    # Prefer safe loading; fall back quietly if unsupported.
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            ckpt = torch.load(ckpt_path, map_location=device)

    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    if "encoder.weight" in sd:
        vocab_size, emb = sd["encoder.weight"].shape
    else:
        for k in sd:
            if k.endswith("encoder.weight"):
                vocab_size, emb = sd[k].shape
                break
        else:
            raise KeyError("encoder.weight not found in state dict")

    if "decoder.weight" in sd:
        _, hidden = sd["decoder.weight"].shape
    else:
        for k in sd:
            if k.startswith("lstm.weight_hh_l0"):
                hidden = sd[k].shape[1]
                break
        else:
            raise KeyError("decoder.weight not found to infer hidden size")

    layers = _infer_layers_from_state(sd)
    vocab = _alphabet_for_vocab(vocab_size)

    model = AWDCharLSTM(vocab_size=vocab_size, emb=emb, hidden=hidden, layers=layers,
                        p_in=0.0, p_h=0.0, p_out=0.0, tie_weights=False).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, vocab

# ---------------- Scoring ----------------

@torch.no_grad()
def bpc_for_text_conditional(model: AWDCharLSTM,
                             text: str,
                             prompt: str = "",
                             device: str = "cuda",
                             skip_first: bool = False) -> float:
    """
    Score TEXT given PROMPT. We warm the hidden state by feeding the prompt, but:
      - We never add prompt's internal loss.
      - If skip_first=False (default): include -log P(text[0] | prompt[-1]).
      - If skip_first=True: DO NOT include that boundary term, but still feed text[0]
        so subsequent terms are conditioned correctly. This makes the denominator
        len(text)-1, matching the no-prompt interior scoring.
    """
    alphabet = _alphabet_for_alphabet = _alphabet_for_vocab(model.encoder.num_embeddings)
    s_text   = _clean_for_alphabet(text,   alphabet)
    s_prompt = _clean_for_alphabet(prompt, alphabet)

    if len(s_text) == 0:
        return float("inf")

    idx = {c: i for i, c in enumerate(alphabet)}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h = None

    # Warm up on prompt (no loss counted).
    if len(s_prompt) >= 2:
        for t in range(1, len(s_prompt)):
            x = torch.tensor([[idx[s_prompt[t-1]]]], dtype=torch.long, device=device)
            _, h = model(x, h)

    total_nll = 0.0
    total_tok = 0

    if len(s_prompt) > 0:
        # Step to boundary distribution P(. | prompt[-1])
        last_p = torch.tensor([[idx[s_prompt[-1]]]], dtype=torch.long, device=device)
        logits, h = model(last_p, h)
        logp = F.log_softmax(logits[0, -1, :], dim=-1)

        if not skip_first:
            total_nll += float(-logp[idx[s_text[0]]].item())
            total_tok += 1

        # In both cases, we must ADVANCE the model by feeding text[0]
        x0 = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
        _, h = model(x0, h)

        # Score the interior transitions of the text
        for t in range(1, len(s_text)):
            x = torch.tensor([[idx[s_text[t-1]]]], dtype=torch.long, device=device)
            logits, h = model(x, h)
            logp = F.log_softmax(logits[0, -1, :], dim=-1)
            total_nll += float(-logp[idx[s_text[t]]].item())
            total_tok += 1
    else:
        # No prompt: standard interior scoring
        if len(s_text) < 2:
            return float("inf")
        for t in range(1, len(s_text)):
            x = torch.tensor([[idx[s_text[t-1]]]], dtype=torch.long, device=device)
            logits, h = model(x, h)
            logp = F.log_softmax(logits[0, -1, :], dim=-1)
            total_nll += float(-logp[idx[s_text[t]]].item())
            total_tok += 1

    bpc = (total_nll / max(1, total_tok)) / math.log(2)
    return bpc

# ---------------- CLI ----------------

def _read_all(path: str) -> str:
    if path == "-" or path == "/dev/stdin":
        return sys.stdin.read()
    with io.open(path, "r", encoding="utf8", errors="ignore") as f:
        return f.read()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to .pt checkpoint")
    ap.add_argument("textfile", help="TEXT to score (path or '-' for stdin)")
    ap.add_argument("--prompt", type=str, default="", help="warm-up string; NOT scored")
    ap.add_argument("--skip_first", action="store_true",
                    help="Do NOT score the cross-boundary term P(text[0] | prompt[-1])")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.model, device=device)

    text = _read_all(args.textfile)
    bpc = bpc_for_text_conditional(model, text, prompt=args.prompt, device=device, skip_first=args.skip_first)
    ppl = 2 ** bpc
    print(f"bpc={bpc:.6f}  ppl={ppl:.6f}")

if __name__ == "__main__":
    main()
