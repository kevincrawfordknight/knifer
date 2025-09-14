#!/usr/bin/env python3
import math, argparse, os, io, json, torch, random
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

# ---------------- Data ----------------
def load_text(path):
    with io.open(path, 'r', encoding='utf8') as f:
        return f.read()

#def build_vocab():
#   # exactly a..z
#   alphabet = ''.join([chr(ord('a')+i) for i in range(26)])
#   stoi = {c:i for i,c in enumerate(alphabet)}
#   itos = {i:c for c,i in stoi.items()}
#   return stoi, itos

def build_vocab():
    # 27 symbols: BOS/EOS-style boundary + a..z
    alphabet = "#abcdefghijklmnopqrstuvwxyz"
    stoi = {c:i for i,c in enumerate(alphabet)}
    itos = {i:c for c,i in stoi.items()}
    return stoi, itos

def encode(s, stoi):
    return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)

class CharStream(torch.utils.data.IterableDataset):
    def __init__(self, ids, block, drop_last=True):
        self.ids = ids
        self.block = block
        self.drop_last = drop_last
    def __iter__(self):
        # simple sequential TBPTT chunks
        L = len(self.ids)
        i = 0
        while i + self.block < L:
            x = self.ids[i:i+self.block]
            y = self.ids[i+1:i+self.block+1]
            i += self.block
            yield x, y

# ---------------- Model ----------------
class AWDCharLSTM(nn.Module):
    def __init__(self, vocab_size=26, emb=512, hidden=2048, layers=3,
                 p_in=0.2, p_h=0.2, p_out=0.2, tie_weights=False):
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
    def forward(self, x, h=None):
        x = self.drop_in(self.encoder(x))
        out, h = self.lstm(x, h)
        out = self.drop_out(out)
        logits = self.decoder(out)
        return logits, h

# ---------------- LR Schedule ----------------
class WarmupCosine:
    """Per-step warmup + cosine decay to final_mult * base_lr."""
    def __init__(self, optimizer, base_lr, total_steps, warmup_steps=None, final_mult=0.1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total = max(1, int(total_steps))
        self.warmup = int(warmup_steps if warmup_steps is not None else max(100, 0.05 * self.total))
        self.warmup = min(self.warmup, self.total - 1) if self.total > 1 else 0
        self.final_mult = float(final_mult)
        self.step_num = 0
        self._apply_lr(self.base_lr * (0.0 if self.warmup > 0 else 1.0))
    def _lr_at(self, t):
        if t < self.warmup and self.warmup > 0:
            return self.base_lr * (t / self.warmup)
        remain = max(1, self.total - self.warmup)
        tt = min(remain, max(0, t - self.warmup))
        cos = 0.5 * (1.0 + math.cos(math.pi * tt / remain))
        return self.base_lr * (self.final_mult + (1.0 - self.final_mult) * cos)
    def _apply_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
    def step(self):
        self.step_num += 1
        self._apply_lr(self._lr_at(self.step_num))

# ---------------- Train / Eval ----------------
def run_epoch(model, data_iter, optimizer=None, device='cuda', scheduler=None):
    ce = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    total_tok = 0
    model.train(optimizer is not None)
    h = None
    for xb,yb in data_iter:
        xb = xb.to(device); yb = yb.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        # reset TBPTT state if batch size changes (last partial)
        if h is not None and h[0].size(1) != xb.size(0):
            h = None

        logits, h = model(xb, h)
        # detach TBPTT hidden between batches
        h = tuple(s.detach() for s in h)
        loss = ce(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))

        if optimizer is not None:
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_tok  += yb.numel()
    bpc = (total_loss / total_tok) / math.log(2)  # nats -> bits
    ppl = 2**bpc
    return bpc, ppl

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_steps_per_epoch(seq_len, block, bsz, drop_last=True):
    """
    Matches CharStream's chunking rule:
      num_chunks = floor((L-1)/block)
      steps = floor(num_chunks / bsz) if drop_last else ceil(num_chunks / bsz)
    """
    num_chunks = max(0, (seq_len - 1) // block)
    if drop_last:
        steps = max(1, num_chunks // bsz) if num_chunks >= bsz else 1 if num_chunks > 0 else 1
    else:
        steps = max(1, math.ceil(num_chunks / bsz)) if num_chunks > 0 else 1
    return steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--test', required=True)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--block', type=int, default=512) # TBPTT length
    ap.add_argument('--bsz', type=int, default=64)
    ap.add_argument('--emb', type=int, default=512)
    ap.add_argument('--hidden', type=int, default=2048)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--dropin', type=float, default=0.2)
    ap.add_argument('--droph', type=float, default=0.2)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--final_lr_mult', type=float, default=0.1, help='final LR = mult * base LR after cosine')
    ap.add_argument('--save', default='char_lstm.pt', help='path to save best checkpoint')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    stoi, itos = build_vocab()
    train_ids = encode(load_text(args.train).lower(), stoi)
    test_ids  = encode(load_text(args.test).lower(),  stoi)

    # Dataloaders (pack into fixed-length chunks)
    def make_loader(ids, block, bsz, shuffle=False):
        ds = CharStream(ids, block)
        return torch.utils.data.DataLoader(
            ds, batch_size=bsz, shuffle=False, num_workers=0, drop_last=True
        )

    train_loader = make_loader(train_ids, args.block, args.bsz)
    test_loader  = make_loader(test_ids,  args.block, args.bsz)

    model = AWDCharLSTM(
        vocab_size=26,
        emb=args.emb,
        hidden=args.hidden,
        layers=args.layers,
        p_in=args.dropin,
        p_h=args.droph,
        p_out=args.dropout,
        tie_weights=False
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Warmup + cosine LR schedule (per step), avoiding len(dataloader) on IterableDataset
    steps_per_epoch = estimate_steps_per_epoch(len(train_ids), args.block, args.bsz, drop_last=True)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(100, total_steps // 20)  # ~5% warmup, at least 100 steps
    sched = WarmupCosine(opt, base_lr=args.lr, total_steps=total_steps,
                         warmup_steps=warmup_steps, final_mult=args.final_lr_mult)

    # Save config for reload
    model_cfg = dict(vocab_size=26, emb=args.emb, hidden=args.hidden, layers=args.layers,
                     p_in=args.dropin, p_h=args.droph, p_out=args.dropout)

    print(f"#params: {count_params(model):,}")
    best = float('inf')

    # Train
    for e in range(1, args.epochs+1):
        tbpc, tppl = run_epoch(model, train_loader, optimizer=opt, device=device, scheduler=sched)
        ebpc, eppl = run_epoch(model, test_loader, optimizer=None, device=device)
        print(f"epoch {e:02d}  TRAIN bpc={tbpc:.4f} ppl={tppl:.3f}   TEST bpc={ebpc:.4f} ppl={eppl:.3f}")

        if ebpc < best:
            best = ebpc
            torch.save({
                'state_dict': model.state_dict(),
                'config': model_cfg,
            }, args.save)
            # also drop a small sidecar JSON
            try:
                with open(os.path.splitext(args.save)[0] + '.json', 'w') as f:
                    json.dump({'best_test_bpc': float(best), **model_cfg}, f, indent=2)
            except Exception:
                pass

    print(f"Best TEST bpc: {best:.4f}  (saved to {args.save})")

if __name__ == '__main__':
    main()
