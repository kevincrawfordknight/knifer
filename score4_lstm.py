#!/usr/bin/env python3
import argparse, io, math, sys, warnings
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- Model ----------------

class AWDCharLSTM(nn.Module):
    def __init__(self, vocab_size=27, emb=512, hidden=512, layers=3,
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

def safe_load(ckpt_path: str, device: str):
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            return torch.load(ckpt_path, map_location=device)

def load_model(ckpt_path: str, device: str = "cuda") -> Tuple[AWDCharLSTM, str]:
    ckpt = safe_load(ckpt_path, device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # Infer shapes
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


# ---------------- Unigram prior P0 ----------------

_ENG_ORDER = "etaoinshrdlcumwfgypbvkjxqz"

def prior_from_text(text: str, alphabet: str, smoothing: float = 1e-6) -> torch.Tensor:
    s = _clean_for_alphabet(text, alphabet)
    counts = {c: 0.0 for c in alphabet}
    for ch in s:
        counts[ch] += 1.0
    # Dirichlet smoothing
    total = sum(counts.values()) + smoothing * len(alphabet)
    probs = torch.tensor([(counts[c] + smoothing) / total for c in alphabet], dtype=torch.float32)
    return probs

def prior_etaoin(alphabet: str) -> torch.Tensor:
    """Heuristic English single-letter prior (position-agnostic). '#' gets small mass if present."""
    scores: Dict[str, float] = {}
    if alphabet[0] == "#":
        # assign a small base mass to '#'
        scores["#"] = 0.5  # relative weight
        letters = alphabet[1:]
    else:
        letters = alphabet
    # rank-based weights 26..1
    for rank, ch in enumerate(_ENG_ORDER[::-1], start=1):
        if ch in letters:
            scores[ch] = rank
    # any letters not in the list get the minimum weight
    for ch in letters:
        scores.setdefault(ch, 1.0)
    total = sum(scores[c] for c in scores)
    probs = torch.tensor([scores[c] / total for c in alphabet], dtype=torch.float32)
    return probs

def prior_uniform(alphabet: str) -> torch.Tensor:
    return torch.full((len(alphabet),), 1.0/len(alphabet), dtype=torch.float32)


# ---------------- Scoring with boundary policy ----------------

@torch.no_grad()
def bpc_score4(model: AWDCharLSTM,
               text: str,
               prompt: str = "",
               p0: Optional[torch.Tensor] = None,
               device: str = "cuda") -> float:
    """
    Boundary policy:
      - If prompt != "": boundary = -log P(x0 | prompt)  (model conditional)
      - If prompt == "": boundary = -log P0(x0)         (fixed unigram prior)
    Then feed x0 and score interior conditionals. Average over all T symbols.
    """
    alphabet = _alphabet_for_vocab(model.encoder.num_embeddings)
    s_text   = _clean_for_alphabet(text, alphabet)
    s_prompt = _clean_for_alphabet(prompt, alphabet)

    if len(s_text) == 0:
        return float("inf")

    idx = {c: i for i, c in enumerate(alphabet)}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    h = None
    # Warm prompt
    if len(s_prompt) >= 1:
        for t in range(1, len(s_prompt)):
            x = torch.tensor([[idx[s_prompt[t-1]]]], dtype=torch.long, device=device)
            _, h = model(x, h)

    total_nll = 0.0
    T = len(s_text)

    # Boundary term
    if len(s_prompt) > 0:
        last_p = torch.tensor([[idx[s_prompt[-1]]]], dtype=torch.long, device=device)
        logits, h = model(last_p, h)
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        total_nll += float(-logp[idx[s_text[0]]].item())
    else:
        # prior P0
        if p0 is None:
            p0 = prior_etaoin(alphabet)
        logp0 = torch.log(p0.clamp_min(1e-30))
        total_nll += float(-logp0[idx[s_text[0]]].item())

    # Feed x0 and score interior
    x0 = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
    _, h = model(x0, h)
    for t in range(1, T):
        x = torch.tensor([[idx[s_text[t-1]]]], dtype=torch.long, device=device)
        logits, h = model(x, h)
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        total_nll += float(-logp[idx[s_text[t]]].item())

    bpc = (total_nll / T) / math.log(2)
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
    ap.add_argument("--prompt", type=str, default="", help="warm-up string; boundary uses model conditional if provided")
    ap.add_argument("--prior_text", type=str, default="", help="file path for estimating P0 from text (used when no prompt)")
    ap.add_argument("--prior", choices=["etaoin","uniform"], default="etaoin", help="fallback prior if --prior_text not supplied")
    ap.add_argument("--smooth", type=float, default=1e-6, help="Dirichlet smoothing for --prior_text")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_model(args.model, device=device)

    # Build P0 (only used when prompt == "")
    p0 = None
    if args.prior_text:
        txt = _read_all(args.prior_text)
        p0 = prior_from_text(txt, vocab, smoothing=args.smooth)
    else:
        p0 = prior_etaoin(vocab) if args.prior == "etaoin" else prior_uniform(vocab)

    text = _read_all(args.textfile)
    bpc = bpc_score4(model, text, prompt=args.prompt, p0=p0, device=device)
    ppl = 2 ** bpc
    print(f"bpc={bpc:.6f}  ppl={ppl:.6f}")

if __name__ == "__main__":
    main()
