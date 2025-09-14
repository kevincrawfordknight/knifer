#!/usr/bin/env python3
"""
beam_subst_v8.py — Faster neural-LM beam search for monoalphabetic substitution.

This version implements the first 4 efficiency upgrades (no rest-cost yet):
  1) ONE LSTM CALL PER BEAM STATE (defer the "feed chosen token" to next step).
  2) BATCH all beams each step (single forward pass gives all next-token logits).
  3) VECTORIZED top-k branching with masking & priors (no per-child Python sorts).
  4) ARRAY keys (c2p/p2c as length-26 int lists) instead of dicts.

Other behavior matches v7:
  • Optional small unigram prior to stabilize early choices.
  • Optional tiny Gumbel noise to diversify branching.
  • Optional `--prefix` that forces plaintext letters at early positions.
  • Optional local swap refinement under the full LM at the end.

Note: No "lookahead/rest-cost" in v8 (we’ll add that later, as discussed).

Usage:
  python beam_subst_v8.py char_lstm.pt ciphertext.txt \
    --beam 800 --topk 12 --prior_w 0.08 --alpha 0.0 --gumbel 0.02 --refine 1 --seed 0
"""
import argparse, math, random
from typing import List, Tuple

import torch
import torch.nn.functional as F

# Reuse your scorer’s model/loader & utilities
from score_lstm import AWDCharLSTM, load_model, clean_text, bpc_for_text

ALPH = "abcdefghijklmnopqrstuvwxyz"
AI = {c:i for i,c in enumerate(ALPH)}

# English letter prior (ETAOIN...) as log-probs (nats)
_ENG = "etaoinshrdlcumwfgypbvkjxqz"
_PRI = {c:(27-i) for i,c in enumerate(_ENG)}  # 26..1
_S = sum(_PRI.values())
LOGP_PRI = torch.tensor([math.log(_PRI[c]/_S) for c in ALPH])  # [26], nats


def gumbel_noise_like(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        return torch.zeros_like(x)
    # Gumbel(0,1) via -log(-log(U))
    u = torch.rand_like(x)
    return -torch.log(-torch.log(torch.clamp(u, min=1e-12))) * scale


class BeamState:
    """
    Stores:
      g        : accumulated log-prob in nats (no BOS score)
      pt       : plaintext prefix (string)
      c2p/p2c  : array keys (len 26, -1 for unmapped; indices 0..25 for 'a'..'z')
      h_prev   : hidden state BEFORE feeding last_idx (tuple(h,c) or None)
      last_idx : int index of last emitted plaintext char (0..25), or None at BOS
    """
    __slots__ = ("g", "pt", "c2p", "p2c", "h_prev", "last_idx")
    def __init__(self, g: float, pt: str, c2p: List[int], p2c: List[int],
                 h_prev: Tuple[torch.Tensor, torch.Tensor] | None, last_idx: int | None):
        self.g = g
        self.pt = pt
        self.c2p = c2p
        self.p2c = p2c
        self.h_prev = h_prev
        self.last_idx = last_idx


@torch.no_grad()
def batch_one_step(model: AWDCharLSTM,
                   last_idx_batch: torch.Tensor,
                   h_prev_batch: Tuple[torch.Tensor, torch.Tensor] | None):
    """
    One-step batched LSTM:
      Inputs:
        last_idx_batch: [B] int64 of previous tokens (y_{t-1})
        h_prev_batch : tuple(h,c) of shape [L,B,H], or None for zeros
      Returns:
        logits_next: [B,V] logits for y_t
        h_after_last: tuple(h,c) after consuming y_{t-1}, shape [L,B,H]
    """
    device = next(model.parameters()).device
    x = last_idx_batch.view(-1, 1).to(device)  # [B,1]
    logits, h_after_last = model(x, h_prev_batch)  # logits: [B,1,V]
    return logits[:, -1, :], h_after_last  # [B,V], (h,c)


def complete_key(c2p_arr: List[int]) -> List[int]:
    """Fill any -1 slots in c2p with remaining plaintext letters (arbitrary but makes a full perm)."""
    used = set([p for p in c2p_arr if p != -1])
    leftover = [i for i in range(26) if i not in used]
    out = c2p_arr[:]
    it = iter(leftover)
    for ci in range(26):
        if out[ci] == -1:
            out[ci] = next(it)
    return out


def rank_score(g: float, length: int, alpha: float) -> float:
    return g / (length ** alpha) if alpha > 0.0 and length > 0 else g


def apply_key_indices(c2p_idx: List[int], ct_clean: str) -> str:
    """Apply a fully-complete c2p (indices) to ciphertext string => plaintext string."""
    table = [chr(ord('a') + p) for p in c2p_idx]
    return "".join(table[AI[ch]] for ch in ct_clean)


def beam_decode_v8(model: AWDCharLSTM,
                   ct: str,
                   beam_size: int = 800,
                   topk_expand: int = 12,
                   prior_w: float = 0.10,
                   alpha: float = 0.0,
                   gumbel: float = 0.0,
                   prefix: str = "") -> Tuple[str, List[int], float]:
    """
    Fast beam (no future-cost) with:
      • one LSTM call per state, batched
      • vectorized top-k branching
      • array-keys
    Returns (best_plaintext, best_c2p_full_idx, best_g).
    """
    device = next(model.parameters()).device
    model.eval()

    ct = clean_text(ct)
    T = len(ct)
    if T == 0:
        return "", list(range(26)), 0.0

    prefix = clean_text(prefix)
    # Precompute torch prior vector on device
    prior_vec = LOGP_PRI.to(device)

    # --- Initialize beam at pos = 0 (no logits yet; we just choose p0 and set last_idx=p0) ---
    c0 = ct[0]
    c0i = AI[c0]

    seed_states: list[BeamState] = []
    if prefix:
        p0 = prefix[0]
        p0i = AI[p0]
        c2p = [-1]*26; p2c = [-1]*26
        c2p[c0i] = p0i; p2c[p0i] = c0i
        g0 = prior_w * prior_vec[p0i].item()  # small prior on first token
        seed_states.append(BeamState(g=g0, pt=p0, c2p=c2p, p2c=p2c, h_prev=None, last_idx=p0i))
    else:
        # branch over all 26 letters (add gumbel to diversify initial ties)
        noise0 = gumbel_noise_like(prior_vec, gumbel)
        score0 = prior_vec + noise0
        # top-26 is all; but if you want, you can cut to 16–26
        order = torch.argsort(score0, descending=True).tolist()
        for p0i in order:
            c2p = [-1]*26; p2c = [-1]*26
            c2p[c0i] = p0i; p2c[p0i] = c0i
            g0 = prior_w * prior_vec[p0i].item()
            seed_states.append(BeamState(g=g0, pt=chr(ord('a')+p0i),
                                         c2p=c2p, p2c=p2c, h_prev=None, last_idx=p0i))

    # prune seeds to beam_size
    seed_states.sort(key=lambda s: rank_score(s.g, len(s.pt), alpha), reverse=True)
    states = seed_states[:beam_size]

    # --- Main loop over positions 1..T-1 ---
    for pos in range(1, T):
        c = ct[pos]
        ci = AI[c]
        forced_p = prefix[pos] if pos < len(prefix) else None
        forced_pi = AI[forced_p] if forced_p else None

        B = len(states)
        # Build batched inputs for ONE LSTM call
        last_idx_batch = torch.tensor([st.last_idx for st in states], dtype=torch.long, device=device)  # [B]

        # Build batched hidden: at pos==1, all h_prev None (start-of-seq); later, all non-None
        if states[0].h_prev is None:
            h_prev_batch = None
        else:
            # Stack (h,c) across beam dim
            hs = torch.stack([st.h_prev[0] for st in states], dim=1)  # [L,B,H]
            cs = torch.stack([st.h_prev[1] for st in states], dim=1)  # [L,B,H]
            h_prev_batch = (hs, cs)

        # ONE forward pass for all beams
        logits_next, h_after_last = batch_one_step(model, last_idx_batch, h_prev_batch)  # [B,V], (h,c)
        logp_next = F.log_softmax(logits_next, dim=-1)  # [B,V] nats

        # We will generate up to B * topk children, then prune to beam_size
        cand_rank: list[Tuple[float, BeamState]] = []

        # Vector views for hidden rows (we'll attach per child without extra LSTM calls)
        # Each child's h_prev becomes the parent's h_after_last row (deferred feed trick).
        h_rows = [ (h_after_last[0][:,i,:], h_after_last[1][:,i,:]) for i in range(B) ]

        for i, st in enumerate(states):
            row_logp = logp_next[i]  # [V]
            h_row = h_rows[i]

            if st.c2p[ci] != -1:  # cipher letter already mapped → forced plaintext
                pi = st.c2p[ci]
                if forced_pi is not None and pi != forced_pi:
                    continue  # inconsistent with prefix
                g_new = st.g + float(row_logp[pi].item())
                pt_new = st.pt + chr(ord('a')+pi)
                st2 = BeamState(g=g_new, pt=pt_new,
                                c2p=st.c2p, p2c=st.p2c,
                                h_prev=h_row, last_idx=pi)
                cand_rank.append((rank_score(st2.g, len(st2.pt), alpha), st2))
            else:
                # Need to assign a new plaintext letter; honor 1-1 via p2c
                if forced_pi is not None:
                    # Check 1-1 constraint
                    if st.p2c[forced_pi] != -1 and st.p2c[forced_pi] != ci:
                        continue
                    # Use the forced letter only
                    pi_list = [forced_pi]
                else:
                    # Build mask for unused plaintext letters
                    # used_mask[p] = True if mapped already
                    used_mask = torch.tensor([1 if st.p2c[p] != -1 else 0 for p in range(26)],
                                             dtype=torch.bool, device=device)
                    # scores = LM + prior + (optional) gumbel; mask used letters to -inf
                    scores = row_logp + prior_w * prior_vec.to(device)
                    if gumbel > 0.0:
                        scores = scores + gumbel_noise_like(scores, gumbel)
                    scores = scores.masked_fill(used_mask, float('-inf'))
                    k = min(topk_expand, int((~used_mask).sum().item()))
                    if k <= 0:
                        continue
                    vals, idx = torch.topk(scores, k)  # top-k plaintext indices
                    pi_list = idx.tolist()

                for pi in pi_list:
                    # Commit mapping
                    if st.p2c[pi] != -1 and st.p2c[pi] != ci:
                        continue
                    c2p = st.c2p[:]  # copy array
                    p2c = st.p2c[:]
                    c2p[ci] = pi
                    p2c[pi] = ci
                    g_new = st.g + float(row_logp[pi].item())
                    # add small prior only when a NEW letter is assigned (not for forced-by-key repeats)
                    g_new += prior_w * prior_vec[pi].item()
                    pt_new = st.pt + chr(ord('a')+pi)
                    st2 = BeamState(g=g_new, pt=pt_new, c2p=c2p, p2c=p2c, h_prev=h_row, last_idx=pi)
                    cand_rank.append((rank_score(st2.g, len(st2.pt), alpha), st2))

        if not cand_rank:
            break
        # Prune to beam_size by rank
        cand_rank.sort(key=lambda x: x[0], reverse=True)
        states = [st for _, st in cand_rank[:beam_size]]

    if not states:
        return "", list(range(26)), -1e9

    # Choose best by true g (optionally you can apply the same length normalization as rank)
    best = max(states, key=lambda s: s.g)
    c2p_full = complete_key(best.c2p)
    return best.pt, c2p_full, best.g


def refine_swaps(model: AWDCharLSTM,
                 ct_clean: str,
                 c2p_full_idx: List[int],
                 rounds: int = 1,
                 device: str = 'cuda') -> Tuple[str, List[int], float]:
    """Local 2-swap refinement under full LM; only swaps cipher letters present in ct."""
    present = sorted(set(ct_clean))
    present_idx = [AI[c] for c in present]
    key = c2p_full_idx[:]  # 26-length list of ints

    pt = apply_key_indices(key, ct_clean)
    best_bpc = bpc_for_text(model, pt, device=device, block=1024)
    improved = True
    r = 0
    while improved and r < max(0, rounds):
        improved = False
        r += 1
        for i_pos, i_c in enumerate(present_idx):
            for j_c in present_idx[i_pos+1:]:
                if i_c == j_c: continue
                k2 = key[:]
                k2[i_c], k2[j_c] = k2[j_c], k2[i_c]
                pt2 = apply_key_indices(k2, ct_clean)
                b = bpc_for_text(model, pt2, device=device, block=1024)
                if b < best_bpc:
                    key, best_bpc, improved = k2, b, True
    final_pt = apply_key_indices(key, ct_clean)
    return final_pt, key, best_bpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=800, help="beam size")
    ap.add_argument("--topk", type=int, default=12, help="branch count when a new cipher letter appears")
    ap.add_argument("--prior_w", type=float, default=0.10, help="weight on unigram prior (nats)")
    ap.add_argument("--alpha", type=float, default=0.0, help="length normalization: rank = g / len^alpha")
    ap.add_argument("--gumbel", type=float, default=0.0, help="Gumbel noise scale for branching diversity")
    ap.add_argument("--prefix", type=str, default="", help="optional known plaintext prefix (nospace a..z)")
    ap.add_argument("--refine", type=int, default=1, help="rounds of local 2-swap refinement")
    ap.add_argument("--seed", type=int, default=0, help="random seed (affects gumbel)")
    args = ap.parse_args()

    import os
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    random.seed(args.seed)
    torch.set_grad_enabled(False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)

    with open(args.cipherfile, "r", encoding="utf8") as f:
        ct_raw = f.read()
    ct = clean_text(ct_raw)

    # Beam decode (fast, no rest-cost)
    pt_beam, c2p_full, g = beam_decode_v8(
        model, ct,
        beam_size=args.beam,
        topk_expand=args.topk,
        prior_w=args.prior_w,
        alpha=args.alpha,
        gumbel=args.gumbel,
        prefix=args.prefix,
    )

    # Optional local refinement
    if args.refine > 0:
        pt_final, c2p_final, bpc = refine_swaps(model, ct, c2p_full, rounds=args.refine, device=device)
        print("PLAINTEXT:\n", pt_final)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(chr(ord('a')+p) for p in c2p_final))
    else:
        from math import log
        # quick bpc for the beam plaintext
        bpc = bpc_for_text(model, pt_beam, device=device, block=1024)
        print("PLAINTEXT:\n", pt_beam)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(chr(ord('a')+p) for p in c2p_full))


if __name__ == "__main__":
    main()
