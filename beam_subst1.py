#!/usr/bin/env python3
import argparse, math, heapq, string, torch
import torch.nn.functional as F

# --- import AWDCharLSTM + loader from your scorer (adjust path if needed) ---
from score_lstm import AWDCharLSTM, load_model, clean_text, encode

ALPH = "abcdefghijklmnopqrstuvwxyz"
AI = {c:i for i,c in enumerate(ALPH)}

class BeamState:
    __slots__ = ("logp","pt","map_c2p","map_p2c","h")
    def __init__(self, logp, pt, map_c2p, map_p2c, h):
        self.logp = logp     # cumulative log-prob (base e)
        self.pt = pt         # decoded plaintext so far (string)
        self.map_c2p = map_c2p  # dict cipher->plain
        self.map_p2c = map_p2c  # dict plain->cipher (enforces 1-1)
        self.h = h           # LSTM hidden state tuple

    def __lt__(self, other):  # for heapq
        return self.logp > other.logp  # max-heap via inverted compare

def step_logprobs(model, last_char_idx, h):
    """Given last plaintext char idx (or None for BOS), get next-char logprobs and next hidden.
       We feed a single token and read distribution for the NEXT token."""
    if last_char_idx is None:
        # start with a dummy token: feed 'a' but ignore its output. Instead,
        # we’ll use a zero-length prefix trick by initializing with an all-zero hidden state.
        # Better: use a learned BOS; but we trained without BOS. So we start from a zero h.
        pass
    x = torch.tensor([[last_char_idx]], dtype=torch.long, device=next(model.parameters()).device)
    logits, h2 = model(x, h)
    # logits shape: [1, 1, V]; we want distribution for next step
    return F.log_softmax(logits[0, -1, :], dim=-1), h2  # [V], new hidden

def beam_decode(model, ct, beam_size=400, topk_expand=26, device='cuda'):
    """LM-guided beam over partial keys.
       - When the next cipher char is mapped: force that letter.
       - When unseen: branch over unused plaintext letters (optionally prune with top-k by LM)."""
    model.eval()
    ct = clean_text(ct)
    # initial state: empty plaintext, empty key, h=None
    init = BeamState(logp=0.0, pt="", map_c2p={}, map_p2c={}, h=None)
    beam = [init]

    for pos, c in enumerate(ct):
        new_beam = []
        for st in beam:
            mapped = st.map_c2p.get(c, None)
            if mapped is not None:
                # deterministic next letter
                last_idx = AI[st.pt[-1]] if st.pt else AI['a']  # seed step; letter choice at first step won't matter much
                lp, h2 = step_logprobs(model, last_idx, st.h)
                p_idx = AI[mapped]
                new_logp = st.logp + float(lp[p_idx].item())
                new_beam.append(BeamState(new_logp, st.pt + mapped, st.map_c2p, st.map_p2c, h2))
            else:
                # need to assign a new plaintext letter: branch
                last_idx = AI[st.pt[-1]] if st.pt else AI['a']
                lp, h2_base = step_logprobs(model, last_idx, st.h)
                # consider only unused plaintext letters
                unused = [p for p in ALPH if p not in st.map_p2c]
                # prune by LM top-k among unused (set topk_expand<=26)
                # get candidate indices sorted by logprob
                cand = sorted(unused, key=lambda ch: float(lp[AI[ch]].item()), reverse=True)[:min(topk_expand, len(unused))]
                for p in cand:
                    map_c2p = dict(st.map_c2p); map_p2c = dict(st.map_p2c)
                    map_c2p[c] = p; map_p2c[p] = c
                    p_idx = AI[p]
                    new_logp = st.logp + float(lp[p_idx].item())
                    # IMPORTANT: reuse h2_base; a single input step doesn’t depend on which p we *choose next*,
                    # only the next token probability does. We can carry the same h forward because we fed last_idx once.
                    new_beam.append(BeamState(new_logp, st.pt + p, map_c2p, map_p2c, h2_base))
        # prune to beam_size
        if not new_beam:
            return "", {}
        # partial sort
        new_beam.sort(key=lambda s: s.logp, reverse=True)
        beam = new_beam[:beam_size]

    # pick best completed hypothesis
    best = max(beam, key=lambda s: s.logp)
    return best.pt, best.map_c2p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=400)
    ap.add_argument("--topk", type=int, default=26, help="max unused plaintext letters to branch on when a new cipher letter appears")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)

    ct = open(args.cipherfile, "r", encoding="utf8").read()
    pt, key = beam_decode(model, ct, beam_size=args.beam, topk_expand=args.topk, device=device)

    # crude bpc readout (optional): average per-char log2 prob
    # (We can reuse the model to compute exact bpc, but this is mainly for sanity.)
    print("PLAINTEXT:\n", pt)
    # Print inferred key in a→z cipher order
    print("KEY c->p:")
    print("".join(key.get(c, "?") for c in ALPH))

if __name__ == "__main__":
    main()
