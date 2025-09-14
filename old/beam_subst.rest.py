#!/usr/bin/env python3
"""
beam_subst8.py — Neural-LM beam search for monoalphabetic substitution with future-cost (A*-style) ranking.

Highlights
- Maintains per-beam hidden state for an AWDCharLSTM (or compatible char LSTM).
- ALWAYS emits a plaintext letter at each position (no '?' in PLAINTEXT).
- A*-style ranking: rank = (g + h_est) / (len^alpha), where
    g      = accumulated log-prob (nats) of chosen plaintext so far
    h_est  = optimistic future log-prob via greedy rollout for K steps + tail bound
    alpha  = length normalization (0 = off; >0 helps when comparing different prefix lengths)
- New-letter branching uses LM top-k with optional unigram prior + tiny Gumbel noise for diversity.
- Optional quick local 2-swap refinement under the full LM to polish the key.

Usage
  python beam_subst8.py char_lstm.pt ciphertext.txt \
    --beam 2000 --topk 16 --lookahead 128 --tail_logp -0.5 \
    --prior_w 0.10 --alpha 0.7 --gumbel 0.05 --refine 1 [--prefix "knownprefix"]

The model is the checkpoint saved by lstm*.py (weights+config). The ciphertext is nospace a..z.
"""
import argparse, math, random, heapq, torch
import torch.nn.functional as F

from score_lstm import AWDCharLSTM, load_model, clean_text, bpc_for_text

ALPH = "abcdefghijklmnopqrstuvwxyz"
AI = {c:i for i,c in enumerate(ALPH)}

# English letter prior (ETAOIN...) -> log prior in nats
_ENG = "etaoinshrdlcumwfgypbvkjxqz"
_PRI = {c:(27-i) for i,c in enumerate(_ENG)}  # 26..1
_S   = sum(_PRI.values())
LOGP_PRI = {c: math.log(_PRI[c]/_S) for c in ALPH}

def gumbel_noise(scale: float) -> float:
    if scale <= 0.0: return 0.0
    u = random.random()
    # Standard Gumbel(0,1) = -ln(-ln(u)); scale β multiplies it
    return -math.log(-math.log(max(1e-12, u))) * scale

class BeamState:
    """
    We store hidden state *before* feeding last_idx.
    At step t:
      - h_prev is the hidden after processing prefix up to y_{t-1} (i.e., state BEFORE feeding last_idx)
      - last_idx is index of y_{t-1} (the last emitted plaintext char)
      - To score y_t, call one_step_logits(last_idx, h_prev) → logits(y_t | prefix), h_curr
      - After choosing y_t = p_idx, next state's (h_prev, last_idx) = (h_curr, p_idx)
    """
    __slots__ = ("g", "pt", "c2p", "p2c", "h_prev", "last_idx")
    def __init__(self, g, pt, c2p, p2c, h_prev, last_idx):
        self.g = g                  # accumulated log-prob in nats (no BOS score)
        self.pt = pt                # plaintext prefix (string)
        self.c2p = c2p              # cipher->plain (partial permutation)
        self.p2c = p2c              # plain->cipher
        self.h_prev = h_prev        # hidden state BEFORE feeding last_idx (None for BOS)
        self.last_idx = last_idx    # index of last emitted plaintext char (int), or None for BOS

def one_step_logits(model, token_idx, h_prev):
    """Feed ONE plaintext token; return logits for NEXT token and new hidden (after this token)."""
    x = torch.tensor([[token_idx]], dtype=torch.long, device=next(model.parameters()).device)
    logits, h_curr = model(x, h_prev)
    return logits[0, -1, :], h_curr  # [V], h_curr

def complete_key(c2p):
    """Complete a partial cipher->plain mapping into a full permutation (dict)."""
    used = set(c2p.values())
    unused_plain = [p for p in ALPH if p not in used]
    full = dict(c2p)
    it = iter(unused_plain)
    for c in ALPH:
        if c not in full:
            full[c] = next(it)
    return full

@torch.no_grad()
def future_upper_bound(model, last_idx, h_prev, cipher_suffix, c2p, K=128, tail_logp=-0.5):
    """
    Optimistic (upper bound) future log-prob (nats) from the current state.
    - Greedy rollout for up to K steps:
        * If cipher char already mapped -> forced letter’s logprob
        * Else -> pick argmax letter (ignores 1-1 constraints; optimistic)
      Each step updates hidden state by feeding the chosen plaintext letter.
    - For the remaining suffix beyond K, add tail_logp per step (optimistic constant).
    """
    if last_idx is None:
        # No token in state yet; estimator would be too weak. Return optimistic tail only.
        return len(cipher_suffix) * tail_logp

    total = 0.0
    steps = 0
    device = next(model.parameters()).device
    lp = None
    h = h_prev
    li = last_idx

    for ch in cipher_suffix:
        # Get distribution for NEXT token y given previous token li, hidden h_prev
        logits_next, h_after_li = one_step_logits(model, li, h)
        logp_next = F.log_softmax(logits_next, dim=-1)

        mapped = c2p.get(ch)
        if mapped is not None:
            p_idx = AI[mapped]
        else:
            # optimistic: pick argmax over all letters (ignore 1-1 constraint)
            p_idx = int(torch.argmax(logp_next).item())

        total += float(logp_next[p_idx].item())
        # Advance hidden and prepare for next step
        _, h_next = one_step_logits(model, p_idx, h_after_li)
        h, li = h_next, p_idx

        steps += 1
        if steps >= K:
            break

    rem = max(0, len(cipher_suffix) - steps)
    total += rem * tail_logp
    return total

def rank_score(g, h_est, length, alpha):
    """Beam ranking score with optional length normalization."""
    denom = (length ** alpha) if alpha > 0 else 1.0
    return (g + h_est) / denom

def apply_key(c2p_full, ct_clean):
    return "".join(c2p_full[ch] for ch in ct_clean)

def select_topk_unused(logp_next, unused, topk, prior_w, gumbel_scale):
    """Return a list of plaintext letters (chars) to branch on, scored by LM+prior+gumbel."""
    scored = []
    for p in unused:
        s = float(logp_next[AI[p]].item()) + prior_w * LOGP_PRI[p] + gumbel_noise(gumbel_scale)
        scored.append((s, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:min(topk, len(scored))]]

def beam_decode(
    model,
    ct,
    beam_size=800,
    topk_expand=12,
    lookahead=128,
    tail_logp=-0.5,
    prior_w=0.10,
    alpha=0.0,
    gumbel=0.0,
    prefix="",
):
    """
    Neural-LM beam for monoalphabetic substitution with A*-style future cost.
    Returns (best_plaintext, best_partial_c2p, best_state_g, best_rank_score).
    """
    model.eval()
    ct = clean_text(ct)
    T = len(ct)
    if T == 0:
        return "", {c: c for c in ALPH}, 0.0, 0.0

    prefix = clean_text(prefix)
    device = next(model.parameters()).device

    # Beam is kept as a list of (rank, counter, BeamState) tuples
    counter = 0
    beam = []  # min-heap on negative rank (we'll push positives and then slice)

    # Seed states for pos=0
    c0 = ct[0]
    seed_states = []
    if prefix:
        # force first plaintext to prefix[0]
        p0 = prefix[0]
        c2p = {c0: p0}; p2c = {p0: c0}
        # No BOS score; apply small unigram prior only
        st = BeamState(g=prior_w*LOGP_PRI[p0], pt=p0, c2p=c2p, p2c=p2c, h_prev=None, last_idx=AI[p0])
        # Rank with future bound
        h_est = future_upper_bound(model, st.last_idx, st.h_prev, ct[1:], st.c2p, K=lookahead, tail_logp=tail_logp)
        r = rank_score(st.g, h_est, len(st.pt), alpha)
        seed_states.append((r, counter, st)); counter += 1
    else:
        # branch over all letters (or a random subset via prior + gumbel)
        letters = list(ALPH)
        random.shuffle(letters)
        for p0 in letters:
            c2p = {c0: p0}; p2c = {p0: c0}
            st = BeamState(g=prior_w*LOGP_PRI[p0], pt=p0, c2p=c2p, p2c=p2c, h_prev=None, last_idx=AI[p0])
            h_est = future_upper_bound(model, st.last_idx, st.h_prev, ct[1:], st.c2p, K=lookahead, tail_logp=tail_logp)
            r = rank_score(st.g, h_est, len(st.pt), alpha)
            seed_states.append((r, counter, st)); counter += 1

    # prune initial seeds
    seed_states.sort(key=lambda x: x[0], reverse=True)
    beam = seed_states[:beam_size]

    # Main loop
    for pos in range(1, T):
        c = ct[pos]
        next_beam = []
        for _, _, st in beam:
            # optional forced plaintext from prefix
            forced_p = prefix[pos] if pos < len(prefix) else None

            # Produce distribution for current position given (h_prev, last_idx)
            logits_next, h_after_last = one_step_logits(model, st.last_idx, st.h_prev)
            logp_next = F.log_softmax(logits_next, dim=-1)

            if c in st.c2p:
                # mapped (forced by key)
                p = st.c2p[c]
                # check 1-1: if plain already bound to a different cipher, skip
                if forced_p and p != forced_p:
                    continue
                p_idx = AI[p]
                g_new = st.g + float(logp_next[p_idx].item())  # no prior on forced
                # advance state for next step
                _, h_next = one_step_logits(model, p_idx, h_after_last)
                st2 = BeamState(g=g_new, pt=st.pt + p, c2p=st.c2p, p2c=st.p2c, h_prev=h_after_last, last_idx=p_idx)
                # Future bound from pos+1
                h_est = future_upper_bound(model, st2.last_idx, st2.h_prev, ct[pos+1:], st2.c2p, K=lookahead, tail_logp=tail_logp)
                r = rank_score(st2.g, h_est, len(st2.pt), alpha)
                next_beam.append((r, counter, st2)); counter += 1
                continue

            # unmapped: branch over unused plaintext letters
            if forced_p:
                # honor forced
                if forced_p in st.p2c and st.p2c[forced_p] != c:
                    continue
                p = forced_p
                p_idx = AI[p]
                g_new = st.g + float(logp_next[p_idx].item()) + prior_w * LOGP_PRI[p]
                c2p = dict(st.c2p); p2c = dict(st.p2c); c2p[c] = p; p2c[p] = c
                _, h_next = one_step_logits(model, p_idx, h_after_last)
                st2 = BeamState(g=g_new, pt=st.pt + p, c2p=c2p, p2c=p2c, h_prev=h_after_last, last_idx=p_idx)
                h_est = future_upper_bound(model, st2.last_idx, st2.h_prev, ct[pos+1:], st2.c2p, K=lookahead, tail_logp=tail_logp)
                r = rank_score(st2.g, h_est, len(st2.pt), alpha)
                next_beam.append((r, counter, st2)); counter += 1
            else:
                unused = [p for p in ALPH if p not in st.p2c]
                cand = select_topk_unused(logp_next, unused, topk_expand, prior_w, gumbel)
                for p in cand:
                    # bind cipher->plain (respect 1-1)
                    if p in st.p2c:  # should not happen after filtering, but guard
                        continue
                    p_idx = AI[p]
                    g_new = st.g + float(logp_next[p_idx].item()) + prior_w * LOGP_PRI[p]
                    c2p = dict(st.c2p); p2c = dict(st.p2c)
                    c2p[c] = p; p2c[p] = c
                    # advance state
                    _, h_next = one_step_logits(model, p_idx, h_after_last)
                    st2 = BeamState(g=g_new, pt=st.pt + p, c2p=c2p, p2c=p2c, h_prev=h_after_last, last_idx=p_idx)
                    # A* future bound
                    h_est = future_upper_bound(model, st2.last_idx, st2.h_prev, ct[pos+1:], st2.c2p, K=lookahead, tail_logp=tail_logp)
                    r = rank_score(st2.g, h_est, len(st2.pt), alpha)
                    next_beam.append((r, counter, st2)); counter += 1

        if not next_beam:
            break
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_size]

    if not beam:
        return "", {c:c for c in ALPH}, -1e9, -1e9

    # Best by true g (optionally length-normalized similarly to rank)
    best_rank, _, best = max(beam, key=lambda x: x[0])
    return best.pt, best.c2p, best.g, best_rank

def refine_swaps(model, ct_clean, c2p_partial, rounds=1, device='cuda'):
    """Local 2-swap refinement under the full LM. Only swaps cipher letters that appear in ct."""
    c2p = complete_key(c2p_partial)
    present = sorted(set(ct_clean))
    key = [c2p[c] for c in ALPH]  # full 26-letter mapping
    # initial score
    pt = apply_key(c2p, ct_clean)
    best_bpc = bpc_for_text(model, pt, device=device, block=1024)
    best_key = key[:]

    for _ in range(max(0, rounds)):
        improved = False
        for i_c in present:
            for j_c in present:
                if i_c >= j_c:
                    continue
                i, j = AI[i_c], AI[j_c]
                k2 = best_key[:]; k2[i], k2[j] = k2[j], k2[i]
                c2p2 = {ALPH[idx]: k2[idx] for idx in range(26)}
                pt2 = apply_key(c2p2, ct_clean)
                bpc = bpc_for_text(model, pt2, device=device, block=1024)
                if bpc < best_bpc:
                    best_bpc, best_key, improved = bpc, k2, True
        if not improved:
            break

    final_c2p = {ALPH[idx]: best_key[idx] for idx in range(26)}
    final_pt = apply_key(final_c2p, ct_clean)
    return final_pt, final_c2p, best_bpc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=800, help="beam size")
    ap.add_argument("--topk", type=int, default=12, help="branch count when a new cipher letter appears")
    ap.add_argument("--lookahead", type=int, default=128, help="greedy rollout steps for future-cost")
    ap.add_argument("--tail_logp", type=float, default=-0.5, help="optimistic per-step log-prob (nats) for remaining tail")
    ap.add_argument("--prior_w", type=float, default=0.10, help="weight on unigram prior (nats)")
    ap.add_argument("--alpha", type=float, default=0.0, help="length normalization in ranking: (g+h)/len^alpha")
    ap.add_argument("--gumbel", type=float, default=0.0, help="Gumbel noise scale for new-letter branching")
    ap.add_argument("--refine", type=int, default=1, help="rounds of local 2-swap refinement")
    ap.add_argument("--prefix", type=str, default="", help="optional known plaintext prefix (nospace a..z)")
    ap.add_argument("--seed", type=int, default=0, help="random seed (for Gumbel / tie breaks)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.set_grad_enabled(False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)

    with open(args.cipherfile, "r", encoding="utf8") as f:
        ct_raw = f.read()
    ct = clean_text(ct_raw)

    # Beam decode (A* ranking)
    pt_beam, c2p_partial, g, rank = beam_decode(
        model, ct,
        beam_size=args.beam,
        topk_expand=args.topk,
        lookahead=args.lookahead,
        tail_logp=args.tail_logp,
        prior_w=args.prior_w,
        alpha=args.alpha,
        gumbel=args.gumbel,
        prefix=args.prefix,
    )

    # Complete + optional local refinement
    if args.refine > 0:
        pt_final, c2p_final, bpc = refine_swaps(model, ct, c2p_partial, rounds=args.refine, device=device)
        print("PLAINTEXT:\n", pt_final)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(c2p_final[c] for c in ALPH))
    else:
        c2p_full = complete_key(c2p_partial)
        print("PLAINTEXT:\n", pt_beam)
        print("BPC:", bpc_for_text(model, pt_beam, device=device, block=1024))
        print("KEY c->p:\n", "".join(c2p_full[c] for c in ALPH))

if __name__ == "__main__":
    main()
