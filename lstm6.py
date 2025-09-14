#!/usr/bin/env python3
import argparse, io, math, os, sys, time, warnings
from typing import Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


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


# ---------------- Vocab / IO ----------------

def build_alphabet_from_data(train_path: str, test_path: str) -> Tuple[str, dict]:
    """
    Auto-detect alphabet from training and test data.
    Returns (alphabet_string, char_frequencies).
    """
    # Read both files
    chars = set()
    all_text = ""

    for path in [train_path, test_path]:
        if os.path.exists(path):
            with io.open(path, "r", encoding="utf8", errors="ignore") as f:
                text = f.read().lower()
                all_text += text
                chars.update(text)

    # Remove whitespace and non-printable chars, keep only letters and useful symbols
    valid_chars = set()
    for c in chars:
        if c.isprintable() and not c.isspace():
            valid_chars.add(c)

    # Sort alphabetically for consistent ordering
    alphabet = ''.join(sorted(valid_chars))

    # Compute character frequencies
    counter = Counter(c for c in all_text if c in valid_chars)

    return alphabet, counter

def build_vocab(alphabet: str):
    stoi = {c: i for i, c in enumerate(alphabet)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos

def clean_to_alphabet(s: str, alphabet: str) -> str:
    s = (s or "").lower()
    allow = set(alphabet)
    return "".join(ch for ch in s if ch in allow)

def load_text_ids(path: str, alphabet: str, stoi: dict) -> torch.Tensor:
    with io.open(path, "r", encoding="utf8", errors="ignore") as f:
        s = f.read().lower()
    s = clean_to_alphabet(s, alphabet)
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def compute_char_priors(char_frequencies: dict, alphabet: str, smoothing: float = 1e-6) -> torch.Tensor:
    """
    Compute character prior probabilities from training data frequencies.
    """
    counts = [char_frequencies.get(c, 0) for c in alphabet]
    total = sum(counts) + smoothing * len(alphabet)
    probs = [(count + smoothing) / total for count in counts]
    return torch.tensor(probs, dtype=torch.float32)


# ---------------- Dataset ----------------

class CharStream(torch.utils.data.IterableDataset):
    """
    Sequential TBPTT chunks: yield (x,y) where y=x shifted by 1.
    """
    def __init__(self, ids: torch.Tensor, block: int):
        self.ids = ids
        self.block = block

    def __iter__(self):
        L = len(self.ids)
        i = 0
        # Ensure we have at least one full block + 1 target
        while i + self.block < L:
            x = self.ids[i:i+self.block]
            y = self.ids[i+1:i+self.block+1]
            i += self.block
            yield x, y

    def __len__(self):
        L = len(self.ids)
        return max(0, (L - 1) // self.block)


# ---------------- Checkpoint helpers ----------------

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
    return 3

def _infer_shapes_from_state(sd) -> Tuple[int,int,int,int]:
    # returns (vocab_size, emb, hidden, layers)
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
    return vocab_size, emb, hidden, layers

def safe_torch_load(path, device):
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            ck = torch.load(path, map_location=device)
    return ck


# ---------------- Train / Eval ----------------

@torch.no_grad()
def evaluate(model: AWDCharLSTM, ids: torch.Tensor, block: int, bsz: int, device: str) -> Tuple[float,float]:
    ds = CharStream(ids, block)
    loader = torch.utils.data.DataLoader(ds, batch_size=bsz, shuffle=False, num_workers=0, drop_last=True)
    ce = nn.CrossEntropyLoss(reduction="sum")
    model.train(False)
    total_loss = 0.0
    total_tok = 0
    h = None
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        if h is not None and h[0].size(1) != xb.size(0):
            h = None
        logits, h = model(xb, h)
        h = (h[0].detach(), h[1].detach())
        loss = ce(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        total_loss += float(loss.item())
        total_tok  += int(yb.numel())
    if total_tok == 0:
        return float("inf"), float("inf")
    bpc = (total_loss / total_tok) / math.log(2)
    ppl = 2 ** bpc
    return bpc, ppl

def run_epoch(model: AWDCharLSTM, loader, optimizer, device: str, clip: float) -> Tuple[float,float]:
    ce = nn.CrossEntropyLoss(reduction="sum")
    model.train(True if optimizer is not None else False)
    total_loss = 0.0
    total_tok = 0
    h = None
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if h is not None and h[0].size(1) != xb.size(0):
            h = None
        logits, h = model(xb, h)
        h = (h[0].detach(), h[1].detach())
        loss = ce(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        if optimizer is not None:
            loss.backward()
            clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total_loss += float(loss.item())
        total_tok  += int(yb.numel())
    if total_tok == 0:
        return float("inf"), float("inf")
    bpc = (total_loss / total_tok) / math.log(2)
    ppl = 2 ** bpc
    return bpc, ppl


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test',  required=True)
    ap.add_argument('--epochs', type=int, default=5)

    ap.add_argument('--block', type=int, default=512, help="TBPTT length")
    ap.add_argument('--bsz',   type=int, default=64,  help="batch size")

    ap.add_argument('--emb',    type=int, default=512)
    ap.add_argument('--hidden', type=int, default=512)
    ap.add_argument('--layers', type=int, default=3)

    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--clip', type=float, default=0.5)

    ap.add_argument('--dropin',  type=float, default=0.2)
    ap.add_argument('--droph',   type=float, default=0.2)
    ap.add_argument('--dropout', type=float, default=0.2)

    ap.add_argument('--save', default='char_lstm.pt', help='path for best-model weights (state_dict only)')
    ap.add_argument('--save_dir', default='.', help='directory for periodic full checkpoints')
    ap.add_argument('--save_every', type=int, default=0, help='if >0, save full checkpoint every N epochs')
    ap.add_argument('--full_ckpt', action='store_true', help='when saving periodic checkpoints, include optimizer etc.')

    ap.add_argument('--resume', type=str, default='', help='path to checkpoint (.pt) to resume from')
    ap.add_argument('--resume_strict', action='store_true', help='strict state_dict load (default: False)')
    ap.add_argument('--no_opt_resume', action='store_true', help="ignore optimizer state in checkpoint")

    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Auto-detect alphabet and character frequencies from training data
    alphabet = None
    char_frequencies = None

    if args.resume:
        # When resuming, load alphabet from checkpoint first
        ck = safe_torch_load(args.resume, device=device)
        if isinstance(ck, dict) and 'vocab' in ck and isinstance(ck['vocab'], str):
            alphabet = ck['vocab']
            if 'char_frequencies' in ck:
                char_frequencies = ck['char_frequencies']

    # If we don't have alphabet from checkpoint, detect from data
    if alphabet is None:
        alphabet, char_frequencies = build_alphabet_from_data(args.train, args.test)
        print(f"Auto-detected alphabet: {alphabet} ({len(alphabet)} chars)")

    stoi, itos = build_vocab(alphabet)

    # If resuming, inspect checkpoint to decide model shapes
    resume_sd = None
    start_epoch = 1
    best_test_bpc = float('inf')

    model = None
    opt = None

    if args.resume:
        ck = safe_torch_load(args.resume, device=device)
        # extract state dict
        if isinstance(ck, dict) and ('model' in ck or 'state_dict' in ck):
            sd = ck.get('model', ck.get('state_dict'))
        elif isinstance(ck, dict):
            sd = ck
        else:
            raise ValueError("Unrecognized checkpoint format for --resume")

        # infer shapes
        vocab_size, emb_r, hidden_r, layers_r = _infer_shapes_from_state(sd)

        # Build model *matching the checkpoint*
        model = AWDCharLSTM(vocab_size=vocab_size, emb=emb_r, hidden=hidden_r, layers=layers_r,
                            p_in=args.dropin, p_h=args.droph, p_out=args.dropout, tie_weights=False).to(device)
        # Load weights
        model.load_state_dict(sd, strict=bool(args.resume_strict))
        print(f"[resume] loaded weights: vocab={vocab_size} emb={emb_r} hidden={hidden_r} layers={layers_r}")

        # Optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if (not args.no_opt_resume) and isinstance(ck, dict) and ('opt' in ck):
            try:
                opt.load_state_dict(ck['opt'])
                print("[resume] loaded optimizer state")
            except Exception as e:
                print(f"[resume] warning: could not load optimizer state: {e}")

        # Starting epoch / best
        if isinstance(ck, dict) and ('epoch' in ck):
            start_epoch = int(ck['epoch']) + 1
            print(f"[resume] continuing from epoch {start_epoch}")
        if isinstance(ck, dict) and ('best_bpc' in ck):
            try:
                best_test_bpc = float(ck['best_bpc'])
            except Exception:
                pass
    else:
        # Fresh model from args
        vocab_size = len(alphabet)
        model = AWDCharLSTM(vocab_size=vocab_size, emb=args.emb, hidden=args.hidden, layers=args.layers,
                            p_in=args.dropin, p_h=args.droph, p_out=args.dropout, tie_weights=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Load data *after* alphabet is finalized
    train_ids = load_text_ids(args.train, alphabet, stoi)
    test_ids  = load_text_ids(args.test,  alphabet, stoi)

    # DataLoaders
    train_ds = CharStream(train_ids, args.block)
    test_ds  = CharStream(test_ids,  args.block)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bsz, shuffle=False, num_workers=0, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=args.bsz, shuffle=False, num_workers=0, drop_last=True)

    # Compute character priors from training data
    char_priors = compute_char_priors(char_frequencies, alphabet)

    # Info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"#params: {n_params:,}   vocab_size={len(alphabet)} ({alphabet})")
    print(f"Character frequencies computed from training data")

    os.makedirs(args.save_dir, exist_ok=True)

    # Train
    for e in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tbpc, tppl = run_epoch(model, train_loader, opt, device=device, clip=args.clip)
        ebpc, eppl = evaluate(model, test_ids, args.block, args.bsz, device=device)
        dt = time.time() - t0

        print(f"epoch {e:02d}  TRAIN bpc={tbpc:.4f}   TEST bpc={ebpc:.4f}   [{dt:.1f}s]")

        # Save best (state_dict + alphabet + priors)
        if ebpc < best_test_bpc:
            best_test_bpc = ebpc
            torch.save({
                'model': model.state_dict(),
                'vocab': alphabet,
                'char_priors': char_priors
            }, args.save)
            print(f"Best TEST bpc: {best_test_bpc:.4f}  (saved to {args.save})")

        # Periodic full checkpoint
        if args.save_every and (e % args.save_every == 0):
            ckpath = os.path.join(args.save_dir, f"char_lstm_epoch{e}.pt")
            if args.full_ckpt:
                torch.save({
                    'model': model.state_dict(),
                    'opt': opt.state_dict(),
                    'epoch': e,
                    'best_bpc': best_test_bpc,
                    'vocab': alphabet,
                    'char_frequencies': char_frequencies,
                    'char_priors': char_priors,
                    'emb': model.emb_dim,
                    'hidden': model.hidden_dim,
                    'layers': model.layers,
                    'block': args.block,
                    'bsz': args.bsz,
                    'lr': args.lr,
                    'wd': args.wd,
                    'dropin': args.dropin,
                    'droph': args.droph,
                    'dropout': args.dropout,
                }, ckpath)
            else:
                torch.save(model.state_dict(), ckpath)
            print(f"[ckpt] saved {ckpath}")

    # Final save of best already handled; also save a "last.pt" full checkpoint for convenience
    last_ckpt = os.path.join(args.save_dir, "char_lstm_last.pt")
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'epoch': args.epochs,
        'best_bpc': best_test_bpc,
        'vocab': alphabet,
        'char_frequencies': char_frequencies,
        'char_priors': char_priors,
        'emb': model.emb_dim,
        'hidden': model.hidden_dim,
        'layers': model.layers,
    }, last_ckpt)
    print(f"[done] wrote last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()