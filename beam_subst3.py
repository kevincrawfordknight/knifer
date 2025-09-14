#!/usr/bin/env python3
import argparse, math, random, torch
import torch.nn.functional as F
from score_lstm import AWDCharLSTM, load_model, clean_text

ALPH = "abcdefghijklmnopqrstuvwxyz"
AI = {c:i for i,c in enumerate(ALPH)}

# English letter prior (rough ETAOIN…) normalized to log-prior in nats
_ENG = "etaoinshrdlcumwfgypbvkjxqz"
_PRI = {c:(27-i) for i,c in enumerate(_ENG)}  # 26..1
S = sum(_PRI.values()); _LP = {c: math.log(_PRI[c]/S) for c in ALPH}

class BeamState:
    __slots__ = ("logp","pt","c2p","p2c","h")
    def __init__(self, logp, pt, c2p, p2c, h):
        self.logp = logp; self.pt = pt; self.c2p = c2p; self.p2c = p2c; self.h = h
    def __lt__(self, other): return self.logp > other.logp  # max-heap behavior

def one_step_logits(model, token_idx, h):
    x = torch.tensor([[token_idx]], dtype=torch.long, device=next(model.parameters()).device)
    logits, h2 = model(x, h)
    return logits[0, -1, :], h2

def beam_decode(model, ct, beam_size=600, topk_expand=12, prior_w=0.15):
    model.eval(); ct = clean_text(ct)
    beam = [BeamState(0.0, "", {}, {}, None)]

    for pos, c in enumerate(ct):
        new_beam = []
        for st in beam:
            mapped = st.c2p.get(c)
            if st.pt == "":
                # First char: branch over ALL letters but randomize order; don't add LM score yet.
                letters = [mapped] if mapped else list(ALPH)
                random.shuffle(letters)
                for p in letters:
                    if p in st.p2c and st.p2c[p] != c: continue
                    c2p = dict(st.c2p); p2c = dict(st.p2c)
                    c2p[c] = p; p2c[p] = c
                    _, h2 = one_step_logits(model, AI[p], st.h)  # set state
                    # add a small unigram prior bonus for first char
                    logp = st.logp + prior_w * _LP[p]
                    new_beam.append(BeamState(logp, st.pt + p, c2p, p2c, h2))
                continue

            # Regular step: score next token by LM; if unmapped, prune to LM top-k (with prior tie-break)
            last_idx = AI[st.pt[-1]]
            prev_logits, h_mid = one_step_logits(model, last_idx, st.h)
            logp_next = F.log_softmax(prev_logits, dim=-1)

            if mapped:
                p_idx = AI[mapped]
                logp = st.logp + float(logp_next[p_idx])
                _, h2 = one_step_logits(model, p_idx, h_mid)
                new_beam.append(BeamState(logp, st.pt + mapped, st.c2p, st.p2c, h2))
            else:
                unused = [p for p in ALPH if p not in st.p2c]
                # rank by LM + small unigram prior
                scored = sorted(
                    unused, key=lambda ch: float(logp_next[AI[ch]]) + prior_w * _LP[ch], reverse=True
                )[:min(topk_expand, len(unused))]
                for p in scored:
                    c2p = dict(st.c2p); p2c = dict(st.p2c)
                    c2p[c] = p; p2c[p] = c
                    p_idx = AI[p]
                    logp = st.logp + float(logp_next[p_idx]) + prior_w * _LP[p]
                    _, h2 = one_step_logits(model, p_idx, h_mid)
                    new_beam.append(BeamState(logp, st.pt + p, c2p, p2c, h2))

        if not new_beam: break
        new_beam.sort(key=lambda s: s.logp, reverse=True)
        beam = new_beam[:beam_size]

    best = max(beam, key=lambda s: s.logp) if beam else BeamState(-1e9,"",{}, {}, None)
    return best.pt, best.c2p

# quick local refinement: try all 2-swaps on key and keep best under full LM
from score_lstm import encode, bpc_for_text
def refine_with_swaps(model, ct, c2p, rounds=2, device='cuda'):
    key = [c2p.get(c, '?') for c in ALPH]
    def apply(k, s): return "".join({ALPH[i]:k[i] for i in range(26)}.get(ch,'') for ch in s)
    pt = apply(key, ct); best_bpc = bpc_for_text(model, pt, device=device, block=1024)
    for _ in range(rounds):
        improved = False
        for i in range(26):
            for j in range(i+1,26):
                k2 = key[:]; k2[i], k2[j] = k2[j], k2[i]
                pt2 = apply(k2, ct)
                b = bpc_for_text(model, pt2, device=device, block=1024)
                if b < best_bpc:
                    key, best_bpc, improved = k2, b, True
        if not improved: break
    return "".join(key), best_bpc, apply(key, ct)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=600)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--prior_w", type=float, default=0.15)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)
    ct = open(args.cipherfile, "r", encoding="utf8").read()

    pt, c2p = beam_decode(model, ct, beam_size=args.beam, topk_expand=args.topk, prior_w=args.prior_w)
    # local refinement under full LM
    key_str, bpc, pt_ref = refine_with_swaps(model, clean_text(ct), c2p, rounds=1, device=device)

    print("PLAINTEXT:\n", pt_ref)
    print("BPC:", bpc)
    print("KEY c->p:\n", key_str)

if __name__ == "__main__":
    main()
