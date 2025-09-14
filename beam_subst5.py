#!/usr/bin/env python3
import argparse, math, random, torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# Reuse your scorer's model + utils
from score_lstm import AWDCharLSTM, load_model, clean_text, bpc_for_text

ALPH = "abcdefghijklmnopqrstuvwxyz"
AI = {c:i for i,c in enumerate(ALPH)}

# Small English unigram prior to stabilize very early choices
_ENG = "etaoinshrdlcumwfgypbvkjxqz"
_PRI = {c:(27-i) for i,c in enumerate(_ENG)}
_S = sum(_PRI.values())
LOGP_PRI = {c: math.log(_PRI[c]/_S) for c in ALPH}

class BeamState:
    __slots__ = ("logp", "pt", "c2p", "p2c", "h", "logp_next")
    def __init__(self,
                 logp: float,
                 pt: str,
                 c2p: Dict[str,str],
                 p2c: Dict[str,str],
                 h,                      # LSTM hidden tuple
                 logp_next: torch.Tensor # log-probs (nats) for NEXT token, shape [V] on device
                 ):
        self.logp = logp
        self.pt = pt
        self.c2p = c2p
        self.p2c = p2c
        self.h = h
        self.logp_next = logp_next

    # For heap/ sort: higher logp is better
    def __lt__(self, other): return self.logp > other.logp

def one_step(model, token_idx: int, h):
    """
    Feed ONE plaintext token idx and return:
      - logp_next: log-softmax for NEXT token (nats), shape [V]
      - h2: new hidden after consuming this token
    """
    dev = next(model.parameters()).device
    x = torch.tensor([[token_idx]], dtype=torch.long, device=dev)
    logits, h2 = model(x, h)
    logp_next = F.log_softmax(logits[0, -1, :], dim=-1)
    return logp_next, h2

def complete_key(c2p: Dict[str,str]) -> Dict[str,str]:
    used = set(c2p.values())
    unused_plain = [p for p in ALPH if p not in used]
    full = dict(c2p)
    it = iter(unused_plain)
    for c in ALPH:
        if c not in full:
            full[c] = next(it)
    return full

def beam_decode(model,
                ct_raw: str,
                beam_size: int = 800,
                topk_expand: int = 12,
                prior_w: float = 0.15) -> Tuple[str, Dict[str,str]]:
    """
    LM-guided beam with strict 1–1 substitution constraint.
    Each state caches logp_next for the next token; we never re-feed the previous token.
    """
    model.eval()
    ct = clean_text(ct_raw)
    if not ct:
        return "", {c:c for c in ALPH}

    dev = next(model.parameters()).device

    # FIRST POSITION: choose a plaintext letter, don't score it (no BOS trained).
    # After choosing p0, feed it ONCE to set (h, logp_next) for position 2.
    init_states = []
    # Use a broad prior to diversify; if you already know the first cipher maps, the map will handle it
    first_c = ct[0]
    for p in ALPH:
        c2p = {first_c: p}
        p2c = {p: first_c}
        # set hidden/logits by feeding p
        logp_next, h2 = one_step(model, AI[p], None)
        # tiny unigram prior only at first step
        st = BeamState(logp=prior_w*LOGP_PRI[p],
                       pt=p, c2p=c2p, p2c=p2c, h=h2, logp_next=logp_next)
        init_states.append(st)
    # prune initial beam
    init_states.sort(key=lambda s: s.logp, reverse=True)
    beam = init_states[:beam_size]

    # SUBSEQUENT POSITIONS
    for pos in range(1, len(ct)):
        c = ct[pos]
        next_beam = []
        for st in beam:
            mapped = st.c2p.get(c)

            if mapped is not None:
                # forced plaintext letter; score from cached logp_next
                p_idx = AI[mapped]
                new_logp = st.logp + float(st.logp_next[p_idx].item())
                # now feed the CHOSEN token ONCE to advance (h, logp_next)
                logp_next, h2 = one_step(model, p_idx, st.h)
                next_beam.append(BeamState(new_logp, st.pt + mapped, st.c2p, st.p2c, h2, logp_next))
            else:
                # expand over unused plaintext letters; rank by LM from cached logp_next (plus tiny prior)
                unused = [p for p in ALPH if p not in st.p2c]
                # pick top-k among unused
                scored = sorted(
                    unused,
                    key=lambda ch: float(st.logp_next[AI[ch]].item()) + prior_w*LOGP_PRI[ch],
                    reverse=True
                )[:min(topk_expand, len(unused))]
                for p in scored:
                    c2p = dict(st.c2p); p2c = dict(st.p2c)
                    c2p[c] = p; p2c[p] = c
                    p_idx = AI[p]
                    new_logp = st.logp + float(st.logp_next[p_idx].item()) + prior_w*LOGP_PRI[p]
                    logp_next, h2 = one_step(model, p_idx, st.h)
                    next_beam.append(BeamState(new_logp, st.pt + p, c2p, p2c, h2, logp_next))

        if not next_beam:
            break
        next_beam.sort(key=lambda s: s.logp, reverse=True)
        beam = next_beam[:beam_size]

    if not beam:
        return "", {c:c for c in ALPH}
    best = max(beam, key=lambda s: s.logp)
    return best.pt, complete_key(best.c2p)

def refine_swaps(model, ct_clean: str, c2p: Dict[str,str], rounds: int = 1, device: Optional[str] = None):
    """Optional local 2-swap refinement under the full LM; only swap letters present in ct."""
    present = sorted(set(ct_clean))
    key = [c2p[c] for c in ALPH]
    table = {ALPH[i]: key[i] for i in range(26)}
    pt = "".join(table[ch] for ch in ct_clean)
    best_bpc = bpc_for_text(model, pt, device=device or ('cuda' if torch.cuda.is_available() else 'cpu'), block=1024)
    for _ in range(rounds):
        improved = False
        for i_c in present:
            for j_c in present:
                if i_c >= j_c: continue
                i, j = AI[i_c], AI[j_c]
                k2 = key[:]; k2[i], k2[j] = k2[j], k2[i]
                t2 = {ALPH[idx]: k2[idx] for idx in range(26)}
                pt2 = "".join(t2[ch] for ch in ct_clean)
                b = bpc_for_text(model, pt2, device=device, block=1024)
                if b < best_bpc:
                    key, best_bpc, improved = k2, b, True
        if not improved: break
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
    ap.add_argument("--refine", type=int, default=1, help="rounds of 2-swap refinement")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)

    ct_raw = open(args.cipherfile, "r", encoding="utf8").read()
    ct = clean_text(ct_raw)

    pt_beam, c2p = beam_decode(model, ct, beam_size=args.beam, topk_expand=args.topk, prior_w=args.prior_w)

    if args.refine > 0:
        pt_final, c2p_final, bpc = refine_swaps(model, ct, c2p, rounds=args.refine, device=device)
        print("PLAINTEXT:\n", pt_final)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(c2p_final[c] for c in ALPH))
    else:
        # compute full bpc for the beam plaintext for reference
        bpc = bpc_for_text(model, pt_beam, device=device, block=1024)
        print("PLAINTEXT:\n", pt_beam)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(c2p[c] for c in ALPH))

if __name__ == "__main__":
    main()
