#!/usr/bin/env python3
import argparse, math, random, torch
import torch.nn.functional as F

from score_lstm import AWDCharLSTM, load_model, clean_text, bpc_for_text

ALPH = "abcdefghijklmnopqrstuvwxyz"
AI = {c:i for i,c in enumerate(ALPH)}

# Small English unigram prior (ETAOIN...) to stabilize early choices
_ENG = "etaoinshrdlcumwfgypbvkjxqz"
_PRI = {c:(27-i) for i,c in enumerate(_ENG)}
S = sum(_PRI.values())
LOGP_PRI = {c: math.log(_PRI[c]/S) for c in ALPH}

class BeamState:
    __slots__ = ("logp","pt","c2p","p2c","h")
    def __init__(self, logp, pt, c2p, p2c, h):
        self.logp = logp   # cumulative nats
        self.pt = pt       # plaintext prefix (string)
        self.c2p = c2p     # cipher->plain dict (partial key)
        self.p2c = p2c     # plain->cipher dict
        self.h  = h        # LSTM hidden state tuple
    def __lt__(self, other):  # for heap behavior (max-heap)
        return self.logp > other.logp

def one_step_logits(model, token_idx, h):
    """Feed ONE plaintext token; return logits for next and new hidden."""
    x = torch.tensor([[token_idx]], dtype=torch.long, device=next(model.parameters()).device)
    logits, h2 = model(x, h)
    return logits[0, -1, :], h2

def complete_key(c2p):
    """Return a full cipher->plain permutation (dict) by filling unused letters."""
    used = set(c2p.values())
    unused_plain = [p for p in ALPH if p not in used]
    full = dict(c2p)
    up = iter(unused_plain)
    for c in ALPH:
        if c not in full:
            full[c] = next(up)
    return full

def beam_decode(model, ct, beam_size=600, topk_expand=12, prior_w=0.15):
    """LM-guided beam with 1–1 key constraint. Produces a plaintext char at every step."""
    model.eval()
    ct = clean_text(ct)
    if not ct:
        return "", {c:c for c in ALPH}

    beam = [BeamState(0.0, "", {}, {}, None)]

    for pos, c in enumerate(ct):
        next_beam = []
        for st in beam:
            mapped = st.c2p.get(c)

            # First character: choose plaintext without scoring against BOS; set hidden using the chosen token.
            if st.pt == "":
                letters = [mapped] if mapped else list(ALPH)
                random.shuffle(letters)
                for p in letters:
                    if p in st.p2c and st.p2c[p] != c: 
                        continue
                    c2p = dict(st.c2p); p2c = dict(st.p2c)
                    c2p[c] = p; p2c[p] = c
                    _, h2 = one_step_logits(model, AI[p], st.h)
                    logp = st.logp + prior_w * LOGP_PRI[p]  # small unigram prior only on first step
                    next_beam.append(BeamState(logp, st.pt + p, c2p, p2c, h2))
                continue

            # Positions >= 2: score next char by LM
            last_idx = AI[st.pt[-1]]
            prev_logits, h_mid = one_step_logits(model, last_idx, st.h)
            logp_next = F.log_softmax(prev_logits, dim=-1)

            if mapped is not None:
                p_idx = AI[mapped]
                new_logp = st.logp + float(logp_next[p_idx])
                # advance hidden with chosen token
                _, h2 = one_step_logits(model, p_idx, h_mid)
                next_beam.append(BeamState(new_logp, st.pt + mapped, st.c2p, st.p2c, h2))
            else:
                # Branch over unused plaintext letters; prune by LM+prior
                unused = [p for p in ALPH if p not in st.p2c]
                cand = sorted(
                    unused, key=lambda ch: float(logp_next[AI[ch]]) + prior_w * LOGP_PRI[ch],
                    reverse=True
                )[:min(topk_expand, len(unused))]
                for p in cand:
                    c2p = dict(st.c2p); p2c = dict(st.p2c)
                    c2p[c] = p; p2c[p] = c
                    p_idx = AI[p]
                    new_logp = st.logp + float(logp_next[p_idx]) + prior_w * LOGP_PRI[p]
                    _, h2 = one_step_logits(model, p_idx, h_mid)
                    next_beam.append(BeamState(new_logp, st.pt + p, c2p, p2c, h2))

        if not next_beam:
            break
        next_beam.sort(key=lambda s: s.logp, reverse=True)
        beam = next_beam[:beam_size]

    if not beam:
        return "", {c:c for c in ALPH}

    best = max(beam, key=lambda s: s.logp)
    # Complete key (to avoid any '?' downstream) and return plaintext from the beam directly
    full_c2p = complete_key(best.c2p)
    return best.pt, full_c2p

def refine_swaps(model, ct_clean, c2p, rounds=1, device='cuda'):
    """Local 2-swap refinement under full LM. Only swap cipher letters that appear in ct."""
    present = sorted(set(ct_clean))
    key = [c2p[c] for c in ALPH]  # full permutation (no '?')
    table = {ALPH[i]: key[i] for i in range(26)}
    pt = "".join(table[ch] for ch in ct_clean)
    best_bpc = bpc_for_text(model, pt, device=device, block=1024)
    for _ in range(rounds):
        improved = False
        for i_c in present:
            for j_c in present:
                if i_c >= j_c: 
                    continue
                i, j = AI[i_c], AI[j_c]
                k2 = key[:]; k2[i], k2[j] = k2[j], k2[i]
                t2 = {ALPH[idx]: k2[idx] for idx in range(26)}
                pt2 = "".join(t2[ch] for ch in ct_clean)
                b = bpc_for_text(model, pt2, device=device, block=1024)
                if b < best_bpc:
                    key, best_bpc, improved = k2, b, True
        if not improved:
            break
    final_c2p = {ALPH[i]: key[i] for i in range(26)}
    final_pt = "".join(final_c2p[ch] for ch in ct_clean)
    return final_pt, final_c2p, best_bpc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=800)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--prior_w", type=float, default=0.15)
    ap.add_argument("--refine", type=int, default=1, help="rounds of local 2-swap refinement")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)

    with open(args.cipherfile, "r", encoding="utf8") as f:
        ct_raw = f.read()
    ct = clean_text(ct_raw)

    # Beam decode
    pt_beam, c2p = beam_decode(model, ct, beam_size=args.beam, topk_expand=args.topk, prior_w=args.prior_w)

    # Optional local refinement (never inserts '?', key is full)
    if args.refine > 0:
        pt_final, c2p_final, bpc = refine_swaps(model, ct, c2p, rounds=args.refine, device=device)
        print("PLAINTEXT:\n", pt_final)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(c2p_final[c] for c in ALPH))
    else:
        print("PLAINTEXT:\n", pt_beam)
        # complete and print key anyway
        c2p_full = complete_key(c2p)
        print("KEY c->p:\n", "".join(c2p_full[c] for c in ALPH))

if __name__ == "__main__":
    main()
