#!/usr/bin/env python3
"""
score5.py — Language-agnostic LSTM text scoring.
Based on score4_lstm.py with minimal changes to remove English bias.
"""
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

def _alphabet_for_vocab_fallback(vsz: int) -> str:
    """Fallback for old checkpoints without saved vocab."""
    if vsz == 26: return "abcdefghijklmnopqrstuvwxyz"
    if vsz == 27: return "#abcdefghijklmnopqrstuvwxyz"
    raise ValueError(f"Unsupported vocab size {vsz}")

def load_model(ckpt_path: str, device: str = "cuda") -> Tuple[AWDCharLSTM, str, Optional[torch.Tensor]]:
    """Load model and extract alphabet and character priors from checkpoint."""
    ckpt = safe_load(ckpt_path, device)

    # Extract state dict - handle multiple formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith(('encoder.', 'lstm.', 'decoder.')) for k in ckpt.keys()):
        # Old format: direct state dict
        sd = ckpt
    else:
        sd = ckpt

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

    # Extract alphabet from checkpoint, with fallback for old checkpoints
    if isinstance(ckpt, dict) and 'vocab' in ckpt:
        alphabet = ckpt['vocab']
        print("Using alphabet from checkpoint")
    else:
        alphabet = _alphabet_for_vocab_fallback(vocab_size)
        print(f"Warning: No alphabet in checkpoint, falling back to {alphabet}")

    # Extract character priors if available
    char_priors = None
    if isinstance(ckpt, dict) and 'char_priors' in ckpt:
        char_priors = ckpt['char_priors']

    model = AWDCharLSTM(vocab_size=vocab_size, emb=emb, hidden=hidden, layers=layers,
                        p_in=0.0, p_h=0.0, p_out=0.0, tie_weights=False).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, alphabet, char_priors


# ---------------- Unigram prior P0 ----------------

def prior_from_text(text: str, alphabet: str, smoothing: float = 1e-6) -> torch.Tensor:
    s = _clean_for_alphabet(text, alphabet)
    counts = {c: 0.0 for c in alphabet}
    for ch in s:
        counts[ch] += 1.0
    # Dirichlet smoothing
    total = sum(counts.values()) + smoothing * len(alphabet)
    probs = torch.tensor([(counts[c] + smoothing) / total for c in alphabet], dtype=torch.float32)
    return probs

def prior_uniform(alphabet: str) -> torch.Tensor:
    return torch.full((len(alphabet),), 1.0/len(alphabet), dtype=torch.float32)


# ---------------- Scoring ----------------

def clean_text_for_model(text: str, alphabet: str) -> str:
    """
    Cleans text to the model's alphabet.
    """
    return _clean_for_alphabet(text, alphabet)

@torch.no_grad()
def score_text(model: AWDCharLSTM, text: str, prompt: str, alphabet: str,
               p0_prior: Optional[torch.Tensor] = None, skip_first: int = 0,
               prior_text: str = "", smoothing: float = 1e-6) -> Tuple[float, float, int]:
    """
    Score text with language model, with optional prompt warm-up and P0 boundary handling.

    Returns (bpc, total_logp, num_chars)
    """
    device = next(model.parameters()).device

    s_text   = clean_text_for_model(text, alphabet)
    s_prompt = clean_text_for_model(prompt, alphabet)

    if not s_text:
        return 0.0, 0.0, 0

    # Character to index mapping
    idx = {c: i for i, c in enumerate(alphabet)}

    # Determine P0 for boundary handling
    if p0_prior is not None:
        p0_log = torch.log(p0_prior.clamp(min=1e-10)).to(device)
    elif prior_text:
        p0 = prior_from_text(prior_text, alphabet, smoothing=smoothing).to(device)
        p0_log = torch.log(p0.clamp(min=1e-10))
    else:
        p0_log = None

    # Convert to tensors for processing
    text_ids = torch.tensor([idx[c] for c in s_text], dtype=torch.long).unsqueeze(0).to(device)

    total_logp = 0.0
    h = None

    # Process prompt if provided (warm-up)
    if s_prompt:
        prompt_ids = torch.tensor([idx[c] for c in s_prompt], dtype=torch.long).unsqueeze(0).to(device)
        _, h = model(prompt_ids, h)

    # Score the text
    start_idx = skip_first
    for i in range(start_idx, len(s_text)):
        ch_idx = idx[s_text[i]]

        # For first character after prompt (position 0), use boundary policy:
        # If we have P0 prior and no prompt, use it directly
        if i == 0 and p0_log is not None and not s_prompt:
            char_logp = p0_log[ch_idx].item()
        else:
            # Use model-conditional: P(char | prev_context)
            if i == 0:
                # First character - use initial or post-prompt hidden state
                if h is None:
                    # No context at all
                    dummy_ids = torch.zeros(1, 1, dtype=torch.long).to(device)
                    logits, h = model(dummy_ids)
                    char_logp = F.log_softmax(logits[0, 0], dim=-1)[ch_idx].item()
                else:
                    # After prompt - get next char probabilities
                    dummy_ids = torch.zeros(1, 1, dtype=torch.long).to(device)
                    logits, h = model(dummy_ids, h)
                    char_logp = F.log_softmax(logits[0, 0], dim=-1)[ch_idx].item()
            else:
                # Use previous character as context
                prev_ids = torch.tensor([[idx[s_text[i-1]]]], dtype=torch.long).to(device)
                logits, h = model(prev_ids, h)
                char_logp = F.log_softmax(logits[0, 0], dim=-1)[ch_idx].item()

        total_logp += char_logp

    num_chars = len(s_text) - start_idx
    if num_chars == 0:
        return 0.0, 0.0, 0

    bpc = -total_logp / (num_chars * math.log(2))
    return bpc, total_logp, num_chars


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Score text with language-agnostic LSTM")
    ap.add_argument("model", help="Path to LSTM model checkpoint")
    ap.add_argument("text_file", help="Path to text file to score")

    ap.add_argument("--prompt", type=str, default="",
                    help="warm-up string; boundary uses model conditional if provided")
    ap.add_argument("--skip_first", type=int, default=0,
                    help="skip first N characters when scoring")

    ap.add_argument("--prior_text", type=str, default="",
                    help="text file to extract character frequencies for P0 prior")
    ap.add_argument("--prior", choices=["data", "uniform"], default="data",
                    help="prior type: data-driven from training (default) or uniform")
    ap.add_argument("--smooth", type=float, default=1e-6,
                    help="smoothing parameter for text-based priors")

    ap.add_argument("--device", default="auto", help="device: cuda, cpu, or auto")
    args = ap.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device

    # Load model
    print(f"Loading model from {args.model}")
    model, alphabet, saved_priors = load_model(args.model, device=device)
    print(f"Alphabet: {alphabet} ({len(alphabet)} chars)")

    # Load text to score
    with io.open(args.text_file, "r", encoding="utf8", errors="ignore") as f:
        text = f.read()

    # Load prior text if specified
    prior_text_content = ""
    if args.prior_text:
        try:
            with io.open(args.prior_text, "r", encoding="utf8", errors="ignore") as f:
                prior_text_content = f.read()
            print(f"Using character priors from {args.prior_text}")
        except FileNotFoundError:
            print(f"Warning: Prior text file {args.prior_text} not found")

    # Determine character priors for P0
    p0_prior = None
    if args.prior_text and prior_text_content:
        p0_prior = prior_from_text(prior_text_content, alphabet, smoothing=args.smooth)
        print("Using character priors from", args.prior_text)
    elif saved_priors is not None and args.prior == "data":
        p0_prior = saved_priors
        print("Using character priors from model checkpoint")
    elif args.prior == "uniform":
        p0_prior = prior_uniform(alphabet)
        print("Using uniform character priors")
    else:
        # Default: try saved priors, fall back to uniform
        if saved_priors is not None:
            p0_prior = saved_priors
            print("Using character priors from model checkpoint")
        else:
            p0_prior = prior_uniform(alphabet)
            print("Using uniform character priors (no training data available)")

    if p0_prior is not None:
        p0_prior = p0_prior.to(device)

    # Score the text
    bpc, total_logp, num_chars = score_text(
        model, text, args.prompt, alphabet,
        p0_prior=p0_prior, skip_first=args.skip_first,
        prior_text=prior_text_content, smoothing=args.smooth
    )

    # Load and show the text being scored
    clean_text_shown = clean_text_for_model(text, alphabet)

    print(f"\nText: {clean_text_shown}")
    print(f"\nResults:")
    print(f"Characters scored: {num_chars}")
    print(f"Bits per character: {bpc:.3f}")


if __name__ == "__main__":
    main()