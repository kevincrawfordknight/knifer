#!/usr/bin/env python3
"""
Beam search decoder for simple substitution ciphers using a character LSTM LM.

Additions in this version:
- --prefix : force a plaintext prefix (a..z, nospace) the decode must adhere to.
  The constraint is enforced position-by-position and on the key bijection.
- Correct next-token caching (no double-advance of the LSTM).
- Optional small unigram prior on early steps.
- Optional optimistic lookahead for robustness on short texts.
- Optional local swap refinement under the full LM that respects the prefix.

Usage:
  python beam_subst6_prefix.py char_lstm.pt cipher.txt \
    --beam 2000 --topk 16 --lookahead 2 --alpha 0.7 --gumbel 0.05 --prior_w 0.1 \
    --prefix "iamthe..."
"""

import argparse, math, random, torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# Reuse your scorer's model + utils
from score_lstm import AWDCharLSTM, load_model, clean_text, bpc_for_text

ALPH = "abcdefghijklmnopqrstuvwxyz"
AI = {c:i for i,c in enumerate(ALPH)}

# Unigram prior for stability on very first steps
_ENG = "etaoinshrdlcumwfgypbvkjxqz"
_PRI = {c:(27-i) for i,c in enumerate(_ENG)}
_S = sum(_PRI.values())
LOGP_PRI = {c: math.log(_PRI[c]/_S) for c in ALPH}

class BeamState:
    __slots__ = ("logp","pt","c2p","p2c","h","logp_next","h_heur")
    def __init__(self, logp, pt, c2p, p2c, h, logp_next, h_heur=0.0):
        self.logp = logp                  # true cumulative nats so far
        self.pt = pt                      # plaintext prefix
        self.c2p, self.p2c = c2p, p2c     # partial key (1–1)
        self.h = h                        # LSTM hidden
        self.logp_next = logp_next        # log-softmax for NEXT token (nats)
        self.h_heur = h_heur              # cached heuristic lookahead (nats)
    def key(self):                        # for sorting: true+heur
        return self.logp + self.h_heur

def one_step(model, token_idx: int, h):
    """Feed ONE plaintext token idx and return (logp_next, new_hidden)."""
    dev = next(model.parameters()).device
    x = torch.tensor([[token_idx]], dtype=torch.long, device=dev)
    logits, h2 = model(x, h)
    return F.log_softmax(logits[0, -1, :], dim=-1), h2  # [V], new h

def complete_key(c2p: Dict[str,str]) -> Dict[str,str]:
    """Fill any unmapped cipher letters with remaining plaintext letters (keeps permutation)."""
    used = set(c2p.values())
    unused = [p for p in ALPH if p not in used]
    it = iter(unused)
    full = dict(c2p)
    for c in ALPH:
        if c not in full:
            full[c] = next(it)
    return full

@torch.no_grad()
def greedy_lookahead(model, ct: str, pos: int, st: BeamState, depth: int = 2,
                     prefix: Optional[str] = None) -> float:
    """Optimistic D-step lookahead (nats).
       - If prefix forces a letter at i, use that; if inconsistent with current key, return -inf.
       - If mapped by key, use mapped.
       - Else use LM argmax (optimistic upper bound)."""
    total = 0.0
    h = st.h
    logp_next = st.logp_next
    L = len(ct)
    for d in range(depth):
        i = pos + d
        if i >= L: break
        forced_idx = None
        if prefix and i < len(prefix):
            p_forced = prefix[i]
            # If forced plaintext already taken by a different cipher letter, it's still okay for lookahead
            # as long as the current cipher at i is either mapped to it or unmapped (optimistic).
            forced_idx = AI[p_forced]
            # If current cipher is mapped inconsistently, return very bad heuristic
            c_here = ct[i]
            mapped = st.c2p.get(c_here)
            if mapped is not None and mapped != p_forced:
                return -1e9
            p_idx = forced_idx
        else:
            c_here = ct[i]
            mapped = st.c2p.get(c_here)
            if mapped is not None:
                p_idx = AI[mapped]
            else:
                p_idx = int(torch.argmax(logp_next).item())  # optimistic
        total += float(logp_next[p_idx].item())
        logp_next, h = one_step(model, p_idx, h)
    return total

def beam_decode(model,
                ct_raw: str,
                beam_size: int = 1200,
                topk_expand: int = 12,
                prior_w: float = 0.1,
                lookahead_depth: int = 2,
                alpha: float = 0.7,
                gumbel_scale: float = 0.05,
                prefix: str = "") -> Tuple[str, Dict[str,str]]:
    """
    LM-guided beam with strict 1–1 substitution constraint and optional plaintext prefix.
    Each state caches next-token logprobs; we never re-feed the previous token.
    """
    model.eval()
    ct = clean_text(ct_raw)
    if not ct:
        return "", {c:c for c in ALPH}
    # Clean and cap prefix
    prefix = clean_text(prefix or "")
    if prefix and len(prefix) > len(ct):
        prefix = prefix[:len(ct)]

    # FIRST POSITION
    first_c = ct[0]
    init = []
    if prefix and len(prefix) >= 1:
        # Force the first plaintext letter
        p0 = prefix[0]
        c2p = {first_c: p0}
        p2c = {p0: first_c}
        logp_next, h2 = one_step(model, AI[p0], None)
        st = BeamState(
            logp=prior_w*LOGP_PRI[p0], pt=p0,
            c2p=c2p, p2c=p2c, h=h2, logp_next=logp_next
        )
        st.h_heur = alpha * greedy_lookahead(model, ct, 1, st, lookahead_depth, prefix=prefix)
        init.append(st)
    else:
        # Branch all letters; don't score token itself; set (h, logp_next).
        for p in ALPH:
            c2p = {first_c: p}
            p2c = {p: first_c}
            logp_next, h2 = one_step(model, AI[p], None)
            st = BeamState(
                logp=prior_w*LOGP_PRI[p], pt=p,
                c2p=c2p, p2c=p2c, h=h2, logp_next=logp_next
            )
            st.h_heur = alpha * greedy_lookahead(model, ct, 1, st, lookahead_depth, prefix=prefix)
            init.append(st)

    init.sort(key=lambda s: s.key(), reverse=True)
    beam = init[:beam_size]

    # SUBSEQUENT POSITIONS
    for pos in range(1, len(ct)):
        c = ct[pos]
        forced = prefix[pos] if prefix and pos < len(prefix) else None

        cand_states = []
        for st in beam:
            if forced is not None:
                # Enforce forced plaintext at this position
                # If cipher already mapped inconsistently -> drop
                mapped = st.c2p.get(c)
                if mapped is not None and mapped != forced:
                    continue
                # If forced plaintext already used by a different cipher letter -> drop (bijection)
                if forced in st.p2c and st.p2c[forced] != c:
                    continue
                # Either extend mapping or reuse consistent one
                c2p = dict(st.c2p); p2c = dict(st.p2c)
                if c not in c2p:
                    c2p[c] = forced; p2c[forced] = c
                p_idx = AI[forced]
                new_logp = st.logp + float(st.logp_next[p_idx].item())
                logp_next, h2 = one_step(model, p_idx, st.h)
                ns = BeamState(new_logp, st.pt + forced, c2p, p2c, h2, logp_next)
                ns.h_heur = alpha * greedy_lookahead(model, ct, pos+1, ns, lookahead_depth, prefix=prefix)
                cand_states.append(ns)
            else:
                # No forced letter: proceed as usual
                mapped = st.c2p.get(c)
                if mapped is not None:
                    p_idx = AI[mapped]
                    new_logp = st.logp + float(st.logp_next[p_idx].item())
                    logp_next, h2 = one_step(model, p_idx, st.h)
                    ns = BeamState(new_logp, st.pt + mapped, st.c2p, st.p2c, h2, logp_next)
                    ns.h_heur = alpha * greedy_lookahead(model, ct, pos+1, ns, lookahead_depth, prefix=prefix)
                    cand_states.append(ns)
                else:
                    # rank unused by (true delta + small prior + lookahead + gumbel noise)
                    unused = [p for p in ALPH if p not in st.p2c]
                    deltas = [(p, float(st.logp_next[AI[p]].item())) for p in unused]
                    deltas.sort(key=lambda t: t[1], reverse=True)
                    for p, delta in deltas[:min(topk_expand, len(deltas))]:
                        c2p = dict(st.c2p); p2c = dict(st.p2c)
                        c2p[c] = p; p2c[p] = c
                        p_idx = AI[p]
                        new_logp = st.logp + delta + prior_w*LOGP_PRI[p]
                        logp_next, h2 = one_step(model, p_idx, st.h)
                        ns = BeamState(new_logp, st.pt + p, c2p, p2c, h2, logp_next)
                        ns.h_heur = alpha * greedy_lookahead(model, ct, pos+1, ns, lookahead_depth, prefix=prefix)
                        if gumbel_scale > 0:
                            u = max(1e-9, random.random())
                            g = -math.log(-math.log(u)) * gumbel_scale
                            ns.h_heur += g
                        cand_states.append(ns)

        if not cand_states:
            break
        cand_states.sort(key=lambda s: s.key(), reverse=True)
        beam = cand_states[:beam_size]

    if not beam:
        return "", {c:c for c in ALPH}
    best = max(beam, key=lambda s: s.logp)  # select by true score at the end
    return best.pt, complete_key(best.c2p)

def refine_swaps(model, ct_clean: str, c2p: Dict[str,str], prefix: str = "",
                 rounds: int = 1, device: Optional[str] = None):
    """Local 2-swap refinement under the full LM.
       Respects the prefix by NEVER swapping cipher letters that appear within the prefix span."""
    forced_ciphers = set(ct_clean[:len(prefix)]) if prefix else set()
    present = sorted(set(ct_clean) - forced_ciphers)  # allowed to swap
    key = [c2p[c] for c in ALPH]
    # Build initial plaintext under current key
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
                # ensure prefix is still respected (it will be, since we didn't swap forced ciphers)
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
    ap.add_argument("--beam", type=int, default=1200)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--prior_w", type=float, default=0.1)
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--gumbel", type=float, default=0.05)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--prefix", type=str, default="", help="plaintext prefix (a..z, nospace) that the decode must adhere to")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)

    ct_raw = open(args.cipherfile, "r", encoding="utf8").read()
    ct = clean_text(ct_raw)
    prefix = clean_text(args.prefix or "")
    if prefix and len(prefix) > len(ct):
        prefix = prefix[:len(ct)]

    # Beam decode with prefix constraint
    pt, c2p = beam_decode(
        model, ct, beam_size=args.beam, topk_expand=args.topk,
        prior_w=args.prior_w, lookahead_depth=args.lookahead,
        alpha=args.alpha, gumbel_scale=args.gumbel, prefix=prefix
    )

    # Optional local refinement (still respects prefix)
    if args.refine > 0:
        pt, c2p, bpc = refine_swaps(model, ct, c2p, prefix=prefix, rounds=args.refine, device=device)
    else:
        bpc = bpc_for_text(model, pt, device=device, block=1024)

    print("PLAINTEXT:\n", pt)
    print("BPC:", bpc)
    print("KEY c->p:\n", "".join(c2p[c] for c in ALPH))

if __name__ == "__main__":
    main()
