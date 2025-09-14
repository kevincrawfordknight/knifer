#!/usr/bin/env python3
import argparse, io, math, os, sys, time, warnings
from typing import Tuple, Optional

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

def _alphabet_for_vocab(vsz: int) -> str:
    if vsz == 26: return "abcdefghijklmnopqrstuvwxyz"
    if vsz == 27: return "#abcdefghijklmnopqrstuvwxyz"
    raise ValueError(f"Unsupported vocab size {vsz} (expected 26 or 27).")

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


# ---------------- Dataset (for training only) ----------------

class CharStream(torch.utils.data.IterableDataset):
    """Sequential TBPTT chunks for training: yield (x,y) where y=x shifted by 1."""
    def __init__(self, ids: torch.Tensor, block: int):
        self.ids = ids
        self.block = block
    def __iter__(self):
        L = len(self.ids)
        i = 0
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
                layers.add(int(k.split("lstm.weight_ih_l", 1)[1].split('.')[0]))
            except Exception:
                pass
    if layers: return max(layers) + 1
    for k in sd.keys():
        if k.startswith("lstm.weight_hh_l"):
            try:
                layers.add(int(k.split("lstm.weight_hh_l", 1)[1].split('.')[0]))
            except Exception:
                pass
    return 3

def _infer_shapes_from_state(sd) -> Tuple[int,int,int,int]:
    if "encoder.weight" in sd:
        vocab_size, emb = sd["encoder.weight"].shape
    else:
        for k in sd:
            if k.endswith("encoder.weight"):
                vocab_size, emb = sd[k].shape; break
        else:
            raise KeyError("encoder.weight not found in state dict")
    if "decoder.weight" in sd:
        _, hidden = sd["decoder.weight"].shape
    else:
        for k in sd:
            if k.startswith("lstm.weight_hh_l0"):
                hidden = sd[k].shape[1]; break
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


# ---------------- Priors P0 ----------------

_ENG_ORDER = "etaoinshrdlcumwfgypbvkjxqz"

def prior_from_text(text: str, alphabet: str, smoothing: float = 1e-6) -> torch.Tensor:
    s = clean_to_alphabet(text, alphabet)
    counts = {c: 0.0 for c in alphabet}
    for ch in s:
        counts[ch] += 1.0
    total = sum(counts.values()) + smoothing * len(alphabet)
    probs = torch.tensor([(counts[c] + smoothing) / total for c in alphabet], dtype=torch.float32)
    return probs

def prior_etaoin(alphabet: str) -> torch.Tensor:
    scores = {}
    if alphabet[0] == "#":
        scores["#"] = 0.5
        letters = alphabet[1:]
    else:
        letters = alphabet
    for rank, ch in enumerate(_ENG_ORDER[::-1], start=1):
        if ch in letters: scores[ch] = rank
    for ch in letters:
        scores.setdefault(ch, 1.0)
    total = sum(scores[c] for c in scores)
    return torch.tensor([scores[c]/total for c in alphabet], dtype=torch.float32)

def prior_uniform(alphabet: str) -> torch.Tensor:
    return torch.full((len(alphabet),), 1.0/len(alphabet), dtype=torch.float32)


# ---------------- Training / Eval ----------------

def run_epoch(model: AWDCharLSTM, loader, optimizer, device: str, clip: float) -> Tuple[float,float]:
    """Standard training CE over TBPTT chunks (not boundary policy)."""
    ce = nn.CrossEntropyLoss(reduction="sum")
    model.train(True)
    total_loss = 0.0
    total_tok = 0
    h = None
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        if h is not None and h[0].size(1) != xb.size(0):
            h = None
        logits, h = model(xb, h)
        h = (h[0].detach(), h[1].detach())
        loss = ce(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
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

@torch.no_grad()
def eval_bpc_boundary(model: AWDCharLSTM,
                      ids: torch.Tensor,
                      alphabet: str,
                      device: str,
                      p0_log: Optional[torch.Tensor],
                      prompt: str = "",
                      max_len: int = 0,
                      chunk: int = 8192) -> Tuple[float,float]:
    """
    Boundary-compatible evaluation (matches score4 / beam_search15):
      - If prompt != "": boundary = -log P(x0 | prompt)
      - If prompt == "": boundary = -log P0(x0)
      - Then feed x0 and score interior conditionals; average over T.
    Evaluates sequentially with recurrent state; processes in chunks for speed.
    """
    if len(ids) == 0:
        return float("inf"), float("inf")
    T = len(ids)
    if max_len and max_len > 0:
        T = min(T, max_len)
        ids = ids[:T]

    idx_to_char = alphabet  # for completeness; not used
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_nll = 0.0
    h = None

    # Warm prompt (string-based; only affects hidden state and boundary when used)
    if prompt:
        s = clean_to_alphabet(prompt, alphabet)
        if len(s) >= 1:
            A = {c:i for i,c in enumerate(alphabet)}
            for t in range(1, len(s)):
                x = torch.tensor([[A[s[t-1]]]], dtype=torch.long, device=device)
                _, h = model(x, h)
            # boundary from prompt -> predict next after last prompt char
            last = torch.tensor([[A[s[-1]]]], dtype=torch.long, device=device)
            logits, h = model(last, h)
            logp = F.log_softmax(logits[0, -1, :], dim=-1)
            total_nll += float(-logp[int(ids[0].item())].item())
    else:
        # P0 boundary
        if p0_log is None:
            raise ValueError("p0_log must be provided when no prompt is used for evaluation")
        total_nll += float(-p0_log[int(ids[0].item())].item())

    # Now feed x0 and score interior in chunks
    x0 = ids[0:1].to(device).view(1, 1)
    _, h = model(x0, h)

    pos = 1
    while pos < T:
        clen = min(chunk, T - pos)
        x = ids[pos-1: pos-1+clen].to(device).view(1, clen)   # inputs
        tgt = ids[pos: pos+clen].to(device).view(clen)        # targets
        logits, h = model(x, h)
        logp = F.log_softmax(logits[0], dim=-1)               # [clen, V]
        total_nll += float(-logp[torch.arange(clen, device=device), tgt].sum().item())
        pos += clen

    bpc = (total_nll / T) / math.log(2)
    ppl = 2 ** bpc
    return bpc, ppl


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test',  required=True)
    ap.add_argument('--epochs', type=int, default=5)

    ap.add_argument('--alphabet', type=str, default="#abcdefghijklmnopqrstuvwxyz",
                    help="Character set; use 'abcdefghijklmnopqrstuvwxyz' for 26.")

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

    ap.add_argument('--save', default='char_lstm.pt', help='best-model weights (state_dict)')
    ap.add_argument('--save_dir', default='.', help='dir for periodic checkpoints')
    ap.add_argument('--save_every', type=int, default=0, help='if >0, save full checkpoint every N epochs')
    ap.add_argument('--full_ckpt', action='store_true', help='save optimizer etc. in periodic checkpoints')

    ap.add_argument('--resume', type=str, default='', help='checkpoint to resume from')
    ap.add_argument('--resume_strict', action='store_true', help='strict state_dict load (default: False)')
    ap.add_argument('--no_opt_resume', action='store_true', help="ignore optimizer state in checkpoint")

    # Boundary-eval knobs (to match score4/beam_search15)
    ap.add_argument('--eval_prompt', type=str, default="", help="prompt used during boundary scoring")
    ap.add_argument('--prior_text', type=str, default="", help="file to estimate P0 (used when no prompt)")
    ap.add_argument('--prior', choices=['etaoin','uniform'], default='etaoin', help="fallback P0 if no --prior_text")
    ap.add_argument('--smooth', type=float, default=1e-6, help="Dirichlet smoothing for --prior_text")
    ap.add_argument('--eval_train_max_len', type=int, default=0, help="limit train tokens for boundary eval (0=all)")
    ap.add_argument('--eval_test_max_len',  type=int, default=0, help="limit test tokens for boundary eval (0=all)")
    ap.add_argument('--eval_chunk', type=int, default=8192, help="token chunk length for boundary eval forward")

    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build vocab from requested alphabet (may be overridden by resume)
    alphabet = args.alphabet
    stoi, itos = build_vocab(alphabet)

    # Resume (optional)
    start_epoch = 1
    best_test_bpc = float('inf')

    if args.resume:
        ck = safe_torch_load(args.resume, device=device)
        sd = ck.get('model', ck.get('state_dict', ck if isinstance(ck, dict) else None))
        if sd is None: raise ValueError("Unrecognized checkpoint format for --resume")

        vocab_size, emb_r, hidden_r, layers_r = _infer_shapes_from_state(sd)
        if isinstance(ck, dict) and ('vocab' in ck) and isinstance(ck['vocab'], str):
            alphabet = ck['vocab']
        else:
            alphabet = _alphabet_for_vocab(vocab_size)
        stoi, itos = build_vocab(alphabet)

        model = AWDCharLSTM(vocab_size=vocab_size, emb=emb_r, hidden=hidden_r, layers=layers_r,
                            p_in=args.dropin, p_h=args.droph, p_out=args.dropout, tie_weights=False).to(device)
        model.load_state_dict(sd, strict=bool(args.resume_strict))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if (not args.no_opt_resume) and isinstance(ck, dict) and ('opt' in ck):
            try:
                opt.load_state_dict(ck['opt'])
                print("[resume] loaded optimizer state")
            except Exception as e:
                print(f"[resume] warning: could not load optimizer state: {e}")
        if isinstance(ck, dict) and ('epoch' in ck):
            start_epoch = int(ck['epoch']) + 1
            print(f"[resume] continuing from epoch {start_epoch}")
        if isinstance(ck, dict) and ('best_bpc' in ck):
            try: best_test_bpc = float(ck['best_bpc'])
            except Exception: pass
    else:
        model = AWDCharLSTM(vocab_size=len(alphabet), emb=args.emb, hidden=args.hidden, layers=args.layers,
                            p_in=args.dropin, p_h=args.droph, p_out=args.dropout, tie_weights=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Load data AFTER alphabet finalized
    train_ids = load_text_ids(args.train, alphabet, stoi)
    test_ids  = load_text_ids(args.test,  alphabet, stoi)

    # Training loader
    train_loader = torch.utils.data.DataLoader(CharStream(train_ids, args.block),
                                               batch_size=args.bsz, shuffle=False, num_workers=0, drop_last=True)

    # P0 for boundary evaluation (used only when no prompt)
    if args.prior_text:
        with io.open(args.prior_text, "r", encoding="utf8", errors="ignore") as f:
            p0 = prior_from_text(f.read(), alphabet, smoothing=args.smooth)
    else:
        p0 = prior_etaoin(alphabet) if args.prior == "etaoin" else prior_uniform(alphabet)
    p0_log = torch.log(p0.clamp_min(1e-30)).to(device)

    # Info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"#params: {n_params:,}   vocab_size={len(alphabet)} ({alphabet})")
    print(f"[eval policy] boundary = "
          f"{'model-conditional (prompted)' if args.eval_prompt else 'P0 prior (no prompt)'}; "
          f"P0={'from text' if args.prior_text else args.prior}")

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = args.save

    # Train loop
    for e in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tbpc_ce, _ = run_epoch(model, train_loader, opt, device=device, clip=args.clip)

        # Boundary-compatible evals
        tr_bpc, tr_ppl = eval_bpc_boundary(model, train_ids, alphabet, device,
                                           p0_log=None if args.eval_prompt else p0_log,
                                           prompt=args.eval_prompt,
                                           max_len=args.eval_train_max_len,
                                           chunk=args.eval_chunk)
        te_bpc, te_ppl = eval_bpc_boundary(model, test_ids, alphabet, device,
                                           p0_log=None if args.eval_prompt else p0_log,
                                           prompt=args.eval_prompt,
                                           max_len=args.eval_test_max_len,
                                           chunk=args.eval_chunk)

        dt = time.time() - t0
        print(f"epoch {e:02d}  TRAIN_CE bpc={tbpc_ce:.4f}   "
              f"TRAIN(boundary) bpc={tr_bpc:.4f} ppl={tr_ppl:.3f}   "
              f"TEST(boundary) bpc={te_bpc:.4f} ppl={te_ppl:.3f}   "
              f"[{dt:.1f}s]")

        # Save best by boundary TEST bpc
        if te_bpc < best_test_bpc:
            best_test_bpc = te_bpc
            torch.save(model.state_dict(), best_path)
            print(f"Best TEST(boundary) bpc: {best_test_bpc:.4f}  (saved to {best_path})")

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

    # Save a final "last" full checkpoint
    last_ckpt = os.path.join(args.save_dir, "char_lstm_last.pt")
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'epoch': args.epochs,
        'best_bpc': best_test_bpc,
        'vocab': alphabet,
        'emb': model.emb_dim,
        'hidden': model.hidden_dim,
        'layers': model.layers,
    }, last_ckpt)
    print(f"[done] wrote last checkpoint: {last_ckpt}")


if __name__ == "__main__":
    main()
