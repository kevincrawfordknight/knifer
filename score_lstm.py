#!/usr/bin/env python3
import io, os, re, sys, math, torch, torch.nn as nn

ALPH = 'abcdefghijklmnopqrstuvwxyz'
STOI = {c:i for i,c in enumerate(ALPH)}

class AWDCharLSTM(nn.Module):
    def __init__(self, vocab_size=26, emb=512, hidden=512, layers=3, p_in=0.2, p_h=0.2, p_out=0.2):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, emb)
        self.drop_in  = nn.Dropout(p_in)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=p_h)
        self.drop_out = nn.Dropout(p_out)
        self.decoder = nn.Linear(hidden, vocab_size, bias=False)
    def forward(self, x, h=None):
        x = self.drop_in(self.encoder(x))
        out, h = self.lstm(x, h)
        out = self.drop_out(out)
        logits = self.decoder(out)
        return logits, h

def clean_text(s: str) -> str:
    return re.sub(r'[^a-z]', '', s.lower())

def encode(s: str) -> torch.Tensor:
    # keep only known chars; clean_text already ensured a–z
    return torch.tensor([STOI[c] for c in s if c in STOI], dtype=torch.long)

def safe_torch_load(path, map_location):
    # Prefer weights_only=True if available (PyTorch ≥ 2.2), else fallback
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def load_model(ckpt_path: str, device='cuda'):
    ckpt = safe_torch_load(ckpt_path, map_location=device)
    # Two formats supported:
    # 1) {'state_dict': ..., 'config': {...}}
    # 2) raw state_dict
    if isinstance(ckpt, dict) and 'state_dict' in ckpt and 'config' in ckpt:
        cfg = ckpt['config']
        model = AWDCharLSTM(**cfg).to(device)
        model.load_state_dict(ckpt['state_dict'])
    else:
        # try to infer config (fallback to common defaults you trained with)
        cfg = dict(vocab_size=26, emb=512, hidden=512, layers=3, p_in=0.2, p_h=0.2, p_out=0.2)
        model = AWDCharLSTM(**cfg).to(device)
        model.load_state_dict(ckpt)
    model.eval()
    return model, cfg

@torch.no_grad()
def bpc_for_text(model: nn.Module, text: str, device='cuda', block: int = 1024) -> float:
    ids = encode(clean_text(text)).to(device)
    T = ids.numel()
    if T < 2:
        return float('inf')
    ce_sum = 0.0
    n_tok = 0
    h = None
    # Iterate in blocks; ensure x and y have identical length each step
    pos = 0
    while pos < T - 1:
        # tokens available for targets is (T-1 - pos)
        span = min(block, (T - 1) - pos)
        x = ids[pos : pos + span]       # [span]
        y = ids[pos + 1 : pos + 1 + span]
        pos += span

        # reset hidden if batch size (here 1) changed or starting fresh
        if h is not None and h[0].size(1) != 1:
            h = None

        x = x.unsqueeze(0)  # [1, span]
        y = y.unsqueeze(0)
        logits, h = model(x, h)
        # detach hidden between blocks
        h = tuple(s.detach() for s in h)

        logp = torch.log_softmax(logits, dim=-1)      # [1, span, V]
        ce = nn.functional.nll_loss(
            logp.reshape(-1, logp.size(-1)),
            y.reshape(-1),
            reduction='sum'
        )
        ce_sum += float(ce.item())
        n_tok += int(y.numel())

    nats = ce_sum / n_tok
    bpc = nats / math.log(2)
    return bpc

def main():
    if len(sys.argv) < 3:
        print("Usage: score_lstm.py <model.pt> <textfile_or_->", file=sys.stderr)
        sys.exit(1)
    model_path = sys.argv[1]
    path = sys.argv[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, cfg = load_model(model_path, device=device)

    data = sys.stdin.read() if path == '-' else io.open(path, 'r', encoding='utf8').read()
    bpc = bpc_for_text(model, data, device=device, block=1024)
    ppl = 2.0 ** bpc
    print(f"bpc={bpc:.6f}  ppl={ppl:.6f}")

if __name__ == '__main__':
    main()
