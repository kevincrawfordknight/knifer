#!/usr/bin/env python3
import argparse, io, math, sys, torch
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

def clean_text(s: str) -> str:
    """Back-compat helper (26-char cleaning)."""
    return _clean_for_alphabet(s, "abcdefghijklmnopqrstuvwxyz")

def _infer_layers_from_state(sd) -> int:
    # Count distinct layer indices present in lstm.weight_ih_l{n}
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
    # Fallback: try weight_hh_l pattern
    for k in sd.keys():
        if k.startswith("lstm.weight_hh_l"):
            try:
                idx = int(k.split("lstm.weight_hh_l", 1)[1].split('.')[0])
                layers.add(idx)
            except Exception:
                pass
    if layers:
        return max(layers) + 1
    return 3  # sensible default

def load_model(ckpt_path: str, device: str = "cuda"):
    """
    Load a model from a plain state_dict checkpoint (no metadata).
    Infers vocab size, emb dim, hidden dim, and #layers from the tensors.
    Returns: (model, alphabet_string)
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Unrecognized checkpoint format")

    # Infer shapes
    if "encoder.weight" in sd:
        vocab_size, emb = sd["encoder.weight"].shape
    else:
        # Some saves might call it 'encoder.embed.weight'
        for k in sd:
            if k.endswith("encoder.weight"):
                vocab_size, emb = sd[k].shape
                break
        else:
            raise KeyError("encoder.weight not found in state dict")

    if "decoder.weight" in sd:
        _, hidden = sd["decoder.weight"].shape
    else:
        # Fallback: infer hidden from an LSTM weight
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

import warnings
import torch

def load_model(ckpt_path: str, device: str = "cuda"):
    """
    Load a model from a plain state_dict checkpoint (no metadata).
    Infers vocab size, emb dim, hidden dim, and #layers from the tensors.
    Returns: (model, alphabet_string)
    """
    # Prefer safe loading (PyTorch >= 2.4). Fall back quietly on older versions.
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



@torch.no_grad()
def bpc_for_text(model: AWDCharLSTM, text: str, device: str = "cuda", block: int = 1024) -> float:
    """
    Compute bits-per-character on `text` for the given model.
    Cleans text to the model's alphabet (26: a..z, 27: #a..z).
    Uses next-char prediction: sum_t -log P(x_t | x_{t-1}).
    """
    alphabet = _alphabet_for_vocab(model.encoder.num_embeddings)
    s = _clean_for_alphabet(text, alphabet)
    if len(s) < 2:
        return float("inf")
    idx = {c: i for i, c in enumerate(alphabet)}
    ids = torch.tensor([[idx[ch] for ch in s]], dtype=torch.long, device=device)  # [1,T]

    total_nll = 0.0
    total_tok = 0
    h = None
    T = ids.size(1)
    # Process in simple sequential fashion (one step at a time)
    for t in range(1, T):
        x = ids[:, t-1:t]  # [1,1]
        logits, h = model(x, h)  # logits [1,1,V]
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        total_nll += float(-logp[ids[0, t]].item())
        total_tok += 1

    bpc = (total_nll / total_tok) / math.log(2)
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
    ap.add_argument("textfile", help="path or '-' for stdin")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, vocab = load_model(args.model, device=device)

    data = _read_all(args.textfile)
    bpc = bpc_for_text(model, data, device=device)
    ppl = 2 ** bpc
    print(f"bpc={bpc:.6f}  ppl={ppl:.6f}")

if __name__ == "__main__":
    main()
