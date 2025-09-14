#!/usr/bin/env python3
"""
beam_subst_v9.py — Faster neural-LM beam search for monoalphabetic substitution.

Implements efficiency upgrades:
  1) ONE LSTM CALL PER BEAM STATE (defer "feed chosen token" to next step).
  2) BATCH all beams each step (single forward pass yields next-token logits).
  3) VECTORIZED top-k branching with masking & priors (no per-child Python sorts).
  4) ARRAY keys (c2p/p2c as length-26 int lists) instead of dicts.

Features:
  • --nbest N : print N-best list (one line each: BPC to 3dp, then PLAINTEXT)
  • --pt "<actual-plaintext>": NOT used for decoding; report 0-based index
      where that plaintext prefix falls off the beam (or -1 if never)
  • --prompt "<string>": warm the LM hidden state by reading this string first.
      The prompt does NOT affect scoring and does NOT constrain the key; it
      only initializes the model state so decoding starts “after” the prompt.

No rest-cost / lookahead in this version.

Example:
  python beam_subst_v9.py char_lstm.pt ciphertext.txt \
    --beam 30000 --topk 20 --alpha 0.0 --gumbel 0.02 --prior_w 0.0 \
    --nbest 10 --refine 2 --seed 0 \
    --prompt "thisisthezodiacspeakingmynameis"
"""
import argparse, math, random
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

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
                 h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]],
                 last_idx: Optional[int]):
        self.g = g
        self.pt = pt
        self.c2p = c2p
        self.p2c = p2c
        self.h_prev = h_prev
        self.last_idx = last_idx


@torch.no_grad()
def batch_one_step(model: AWDCharLSTM,
                   last_idx_batch: torch.Tensor,
                   h_prev_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]):
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


@torch.no_grad()
def warm_hidden_with_prompt(model: AWDCharLSTM, prompt: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Feed the prompt through the LM to produce a warm hidden state.
    Returns the hidden (h,c) AFTER the last prompt character, or None if prompt empty.
    Does NOT affect scoring and does NOT constrain cipher->plain mappings.
    """
    prompt = clean_text(prompt)
    if not prompt:
        return None
    device = next(model.parameters()).device
    idxs = torch.tensor([[AI[ch] for ch in prompt]], dtype=torch.long, device=device)  # [1,Tp]
    # Run in a single forward; we only care about the final hidden state
    _, h = model(idxs, None)  # h: (h_n, c_n) with shapes [L, 1, H]
    # Squeeze batch dim so downstream stacking works
    return (h[0].squeeze(1), h[1].squeeze(1))  # ([L,H], [L,H])


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


def beam_decode_v9(model: AWDCharLSTM,
                   ct: str,
                   beam_size: int = 800,
                   topk_expand: int = 12,
                   prior_w: float = 0.10,
                   alpha: float = 0.0,
                   gumbel: float = 0.0,
                   prefix: str = "",
                   pt_actual: str = "",
                   prompt: str = ""):
    """
    Fast beam (no future-cost) with:
      • one LSTM call per state, batched
      • vectorized top-k branching
      • array-keys
      • optional LM-only prompt warm-up
    Returns:
      best_pt, best_c2p_full, best_g, final_states(list), pt_falloff_index(int)
    """
    device = next(model.parameters()).device
    model.eval()

    ct = clean_text(ct)
    T = len(ct)
    if T == 0:
        return "", list(range(26)), 0.0, [], -1

    prefix = clean_text(prefix)
    pt_actual = clean_text(pt_actual)
    if pt_actual and len(pt_actual) != T:
        pt_actual = pt_actual[:T]

    # Precompute prior vector
    prior_vec = LOGP_PRI.to(device)

    # --- LM warm-up with prompt (does not affect score or key) ---
    h_prompt = warm_hidden_with_prompt(model, prompt)  # None or ([L,H],[L,H])

    # --- Initialize beam at pos = 0 ---
    c0i = AI[ct[0]]

    seed_states: List[BeamState] = []
    if prefix:
        p0i = AI[prefix[0]]
        c2p = [-1]*26; p2c = [-1]*26
        c2p[c0i] = p0i; p2c[p0i] = c0i
        g0 = prior_w * prior_vec[p0i].item()
        seed_states.append(BeamState(g=g0, pt=chr(ord('a')+p0i),
                                     c2p=c2p, p2c=p2c, h_prev=h_prompt, last_idx=p0i))
    else:
        noise0 = gumbel_noise_like(prior_vec, gumbel)
        score0 = prior_vec + noise0
        order = torch.argsort(score0, descending=True).tolist()
        for p0i in order:
            c2p = [-1]*26; p2c = [-1]*26
            c2p[c0i] = p0i; p2c[p0i] = c0i
            g0 = prior_w * prior_vec[p0i].item()
            seed_states.append(BeamState(g=g0, pt=chr(ord('a')+p0i),
                                         c2p=c2p, p2c=p2c, h_prev=h_prompt, last_idx=p0i))

    seed_states.sort(key=lambda s: rank_score(s.g, len(s.pt), alpha), reverse=True)
    states = seed_states[:beam_size]

    # Track where the provided PT (if any) falls off the beam
    pt_falloff_index = -1
    if pt_actual:
        alive = any(st.pt == pt_actual[:1] for st in states)
        if not alive:
            pt_falloff_index = 0

    # --- Main loop over positions 1..T-1 ---
    for pos in range(1, T):
        ci = AI[ct[pos]]
        forced_pi = AI[prefix[pos]] if (pos < len(prefix)) else None

        B = len(states)
        last_idx_batch = torch.tensor([st.last_idx for st in states], dtype=torch.long, device=device)

        if states[0].h_prev is None:
            h_prev_batch = None
        else:
            # Stack warm/propagated hidden states across beam
            hs = torch.stack([st.h_prev[0] for st in states], dim=1)  # [L,B,H]
            cs = torch.stack([st.h_prev[1] for st in states], dim=1)  # [L,B,H]
            h_prev_batch = (hs, cs)

        logits_next, h_after_last = batch_one_step(model, last_idx_batch, h_prev_batch)  # [B,V]
        logp_next = F.log_softmax(logits_next, dim=-1)  # [B,V] nats

        cand_rank: List[Tuple[float, BeamState]] = []
        h_rows = [ (h_after_last[0][:,i,:], h_after_last[1][:,i,:]) for i in range(B) ]

        for i, st in enumerate(states):
            row_logp = logp_next[i]
            h_row = h_rows[i]

            if st.c2p[ci] != -1:
                pi = st.c2p[ci]
                if forced_pi is not None and pi != forced_pi:
                    continue
                g_new = st.g + float(row_logp[pi].item())
                pt_new = st.pt + chr(ord('a')+pi)
                st2 = BeamState(g=g_new, pt=pt_new, c2p=st.c2p, p2c=st.p2c, h_prev=h_row, last_idx=pi)
                cand_rank.append((rank_score(st2.g, len(st2.pt), alpha), st2))
            else:
                if forced_pi is not None:
                    if st.p2c[forced_pi] != -1 and st.p2c[forced_pi] != ci:
                        continue
                    pi_list = [forced_pi]
                else:
                    used_mask = torch.tensor([1 if st.p2c[p] != -1 else 0 for p in range(26)],
                                             dtype=torch.bool, device=device)
                    scores = row_logp + prior_w * prior_vec
                    if gumbel > 0.0:
                        scores = scores + gumbel_noise_like(scores, gumbel)
                    scores = scores.masked_fill(used_mask, float('-inf'))
                    k = min(topk_expand, int((~used_mask).sum().item()))
                    if k <= 0:
                        continue
                    _, idx = torch.topk(scores, k)
                    pi_list = idx.tolist()

                for pi in pi_list:
                    if st.p2c[pi] != -1 and st.p2c[pi] != ci:
                        continue
                    c2p = st.c2p[:]; p2c = st.p2c[:]
                    c2p[ci] = pi; p2c[pi] = ci
                    g_new = st.g + float(row_logp[pi].item()) + prior_w * prior_vec[pi].item()
                    pt_new = st.pt + chr(ord('a')+pi)
                    st2 = BeamState(g=g_new, pt=pt_new, c2p=c2p, p2c=p2c, h_prev=h_row, last_idx=pi)
                    cand_rank.append((rank_score(st2.g, len(st2.pt), alpha), st2))

        if not cand_rank:
            break

        cand_rank.sort(key=lambda x: x[0], reverse=True)
        states = [st for _, st in cand_rank[:beam_size]]

        # Update falloff tracking
        if pt_actual and pt_falloff_index == -1:
            alive = any(st.pt == pt_actual[:pos+1] for st in states)
            if not alive:
                pt_falloff_index = pos

    if not states:
        return "", list(range(26)), -1e9, [], pt_falloff_index

    best = max(states, key=lambda s: s.g)
    c2p_full = complete_key(best.c2p)
    return best.pt, c2p_full, best.g, states, pt_falloff_index


def refine_swaps(model: AWDCharLSTM,
                 ct_clean: str,
                 c2p_full_idx: List[int],
                 rounds: int = 1,
                 device: str = 'cuda'):
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
        for ii, i_c in enumerate(present_idx):
            for j_c in present_idx[ii+1:]:
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
    ap.add_argument("--prompt", type=str, default="", help="LM-only warm-up text (nospace a..z); not scored and does not constrain the key")
    ap.add_argument("--refine", type=int, default=1, help="rounds of local 2-swap refinement")
    ap.add_argument("--seed", type=int, default=0, help="random seed (affects gumbel)")
    ap.add_argument("--nbest", type=int, default=10, help="how many final hypotheses to list")
    ap.add_argument("--pt", type=str, default="", help="actual plaintext (for diagnostics only; not used in decoding)")
    args = ap.parse_args()

    # Reproducibility for any PyTorch randomness (e.g., Gumbel noise)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.set_grad_enabled(False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)

    with open(args.cipherfile, "r", encoding="utf8") as f:
        ct_raw = f.read()
    ct = clean_text(ct_raw)

    # Beam decode
    pt_beam, c2p_full, g, end_states, pt_falloff = beam_decode_v9(
        model, ct,
        beam_size=args.beam,
        topk_expand=args.topk,
        prior_w=args.prior_w,
        alpha=args.alpha,
        gumbel=args.gumbel,
        prefix=args.prefix,
        pt_actual=args.pt,
        prompt=args.prompt,
    )

    # ---- N-best list (by true g) ----
    nlist = sorted(end_states, key=lambda s: s.g, reverse=True)[:max(1, args.nbest)]
    for st in nlist:
        bpc = bpc_for_text(model, st.pt, device=device, block=1024)
        print(f"{bpc:.3f} {st.pt}")

    # ---- Optional local refinement of the single best ----
    if args.refine > 0:
        pt_final, c2p_final, bpc = refine_swaps(model, ct, c2p_full, rounds=args.refine, device=device)
        print("PLAINTEXT:\n", pt_final)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(chr(ord('a')+p) for p in c2p_final))
    else:
        bpc = bpc_for_text(model, pt_beam, device=device, block=1024)
        print("PLAINTEXT:\n", pt_beam)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(chr(ord('a')+p) for p in c2p_full))

    # ---- Diagnostic: where does the provided plaintext fall off the beam? ----
    if args.pt:
        print("PT_FALLOFF_INDEX:", pt_falloff)


if __name__ == "__main__":
    main()
