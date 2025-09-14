#!/usr/bin/env python3
"""
beam_subst_v12.py — Prompt-aware, fast beam search for monoalphabetic substitution,
now with an explicit boundary token '#'.

This version fixes seeding at position 0: if the first ciphertext char is '#',
we FORCE '#→#' (no branching), so N-best outputs begin with '#'.

Key features:
  • Vocab is 27 symbols:  "#" + "abcdefghijklmnopqrstuvwxyz"
  • The key fixes '#' → '#' permanently (no branching on '#')
  • Cleaning keeps '#' (and a..z) and drops everything else
  • Prompt-aware scoring: score = logP(prompt internals) + logP(first|prompt) + logP(plaintext|…)
  • N-best is sorted by LM BPC of (prompt + plaintext)
  • Optional --pt diagnostics (report where gold falls off beam)
  • Optional 2-swap local refinement under full LM

No rest-cost / lookahead in this version.

NOTE: This file imports the auto-detecting loader from score2_lstm.py.
"""

import argparse, math, random
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

# Auto-detect loader/scorer (supports 26 or 27 and preserves '#')
from score2_lstm import AWDCharLSTM, load_model

# ------------------------------
# Alphabet & utilities
# ------------------------------

ALPH = "#abcdefghijklmnopqrstuvwxyz"
V = len(ALPH)  # 27
AI = {c: i for i, c in enumerate(ALPH)}
HASH = AI["#"]

# English letter prior (ETAOIN...) for indices 1..26; '#'(0) gets 0
_ENG = "etaoinshrdlcumwfgypbvkjxqz"
_PRI = {c: (27 - i) for i, c in enumerate(_ENG)}  # 26..1
_S = sum(_PRI.values())
LOGP_PRI = torch.zeros(V)
for c in _ENG:
    LOGP_PRI[AI[c]] = math.log(_PRI[c] / _S)  # nats


def keep_text(s: str) -> str:
    """Keep only '#' and a..z (lowercased)."""
    s = (s or "").lower()
    return "".join(ch for ch in s if ch in AI)


def gumbel_noise_like(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0:
        return torch.zeros_like(x)
    u = torch.rand_like(x)
    return -torch.log(-torch.log(torch.clamp(u, min=1e-12))) * scale


# ------------------------------
# One-step batched LM
# ------------------------------

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


# ------------------------------
# Prompt handling (keeps '#')
# ------------------------------

@torch.no_grad()
def prompt_info(model: AWDCharLSTM, prompt: str):
    """
    Compute LM score over the prompt (sum logP(prompt[t] | prompt[:t-1]) for t>=1),
    and the state needed to start decoding right after the prompt:
      - idx_last_prompt : int index of last prompt char (or None if prompt empty)
      - h_before_last   : hidden BEFORE feeding the last prompt char, shape [L,H] (or None if len<=1)
      - h_after_last    : hidden AFTER feeding the last prompt char,  shape [L,H] (or None if prompt empty)
      - logp_next_after_prompt : tensor [V] of log-probs for the FIRST decoded char given the prompt
                                 (or None if prompt empty)
      - g_prompt        : float, LM log-prob sum over prompt internals (t>=1)
    NOTE: we cannot include logP(prompt[0]|BOS) since the model has no BOS token.
    """
    prompt = keep_text(prompt)
    if not prompt:
        return None, None, None, None, 0.0

    device = next(model.parameters()).device
    idxs = [AI[ch] for ch in prompt]
    g_prompt = 0.0
    h_prev_3d = None                  # None or ([L,1,H],[L,1,H])
    h_before_last_3d = None
    idx_last = idxs[-1]

    # score internal transitions and capture state before last char
    for t in range(1, len(idxs)):
        x = torch.tensor([[idxs[t - 1]]], dtype=torch.long, device=device)  # [1,1]
        logits, h_after_3d = model(x, h_prev_3d)  # logits [1,1,V], h_after_3d ([L,1,H],[L,1,H])
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        g_prompt += float(logp[idxs[t]].item())
        if t == len(idxs) - 1:
            # state BEFORE feeding last prompt char = h_after_3d (after feeding idxs[t-1])
            h_before_last_3d = h_after_3d
        h_prev_3d = h_after_3d

    # Now get distribution for the FIRST decoded char given the prompt:
    x_last = torch.tensor([[idx_last]], dtype=torch.long, device=device)
    logits_next, h_after_last_3d = model(x_last, h_before_last_3d)  # [1,1,V], ([L,1,H],[L,1,H])
    logp_next = F.log_softmax(logits_next[0, -1, :], dim=-1)  # [V]

    # Convert stored states to 2-D [L,H] for per-beam storage later
    h_before_last = (h_before_last_3d[0].squeeze(1), h_before_last_3d[1].squeeze(1)) if h_before_last_3d else None
    h_after_last  = (h_after_last_3d[0].squeeze(1),  h_after_last_3d[1].squeeze(1))

    return idx_last, h_before_last, h_after_last, logp_next, g_prompt


# ------------------------------
# Beam state
# ------------------------------

class BeamState:
    """
    Stores:
      g        : accumulated log-prob in nats (includes prompt internals + first char given prompt)
      pt       : plaintext prefix (string)
      c2p/p2c  : array keys (len V, -1 for unmapped; indices 0..V-1 for ALPH)
      h_prev   : hidden state BEFORE feeding last_idx (tuple(h,c) or None); each [L,H]
      last_idx : int index of last emitted plaintext char (0..V-1), or None at BOS
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


def make_empty_key_fixed_hash() -> Tuple[List[int], List[int]]:
    """Create empty permutation with '#' fixed to '#'."""
    c2p = [-1] * V
    p2c = [-1] * V
    c2p[HASH] = HASH
    p2c[HASH] = HASH
    return c2p, p2c


def complete_key(c2p_arr: List[int]) -> List[int]:
    """Fill any -1 slots in c2p with remaining plaintext letters (arbitrary but makes a full perm)."""
    used = set([p for p in c2p_arr if p != -1])
    leftover = [i for i in range(V) if i not in used]
    out = c2p_arr[:]
    it = iter(leftover)
    for ci in range(V):
        if out[ci] == -1:
            out[ci] = next(it)
    return out


def rank_score(g: float, length: int, alpha: float) -> float:
    return g / (length ** alpha) if alpha > 0.0 and length > 0 else g


def apply_key_indices(c2p_idx: List[int], ct_clean: str) -> str:
    """Apply a fully-complete c2p (indices) to ciphertext string => plaintext string."""
    table = [ALPH[p] for p in c2p_idx]
    return "".join(table[AI[ch]] for ch in ct_clean)


# ------------------------------
# BPC on (prompt + plaintext), keeping '#'
# ------------------------------

@torch.no_grad()
def bpc_for_text_local(model: AWDCharLSTM, text: str, device: str = "cuda") -> float:
    s = keep_text(text)
    if len(s) < 2:
        return float("inf")
    idx = {c: i for i, c in enumerate(ALPH)}
    ids = torch.tensor([[idx[ch] for ch in s]], dtype=torch.long, device=device)  # [1,T]
    total_nll = 0.0
    total_tok = 0
    h = None
    Tlen = ids.size(1)
    for t in range(1, Tlen):
        x = ids[:, t - 1 : t]  # [1,1]
        logits, h = model(x, h)  # logits [1,1,V]
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        total_nll += float(-logp[ids[0, t]].item())
        total_tok += 1
    bpc = (total_nll / total_tok) / math.log(2)
    return bpc


# ------------------------------
# Beam search (prompt-aware, 27-char)
# ------------------------------

@torch.no_grad()
def beam_decode_v12(model: AWDCharLSTM,
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
    Beam search (no future-cost) with:
      • batched one-step forward
      • vectorized top-k branching
      • array-keys, with '#' fixed to '#'
      • prompt-aware scoring (g includes prompt internals and P(first|prompt))
    Returns:
      best_pt, best_c2p_full, best_g, final_states(list), pt_falloff_index(int), prompt_clean(str)
    """
    device = next(model.parameters()).device
    model.eval()

    ct = keep_text(ct)
    T = len(ct)
    if T == 0:
        return "", list(range(V)), 0.0, [], -1, keep_text(prompt)

    prefix = keep_text(prefix)
    pt_actual = keep_text(pt_actual)
    if pt_actual and len(pt_actual) != T:
        pt_actual = pt_actual[:T]

    prior_vec = LOGP_PRI.to(device)

    # Prompt preparation
    idx_last_prompt, h_before_last, h_after_last, logp_next_after_prompt, g_prompt = prompt_info(model, prompt)
    prompt_clean = keep_text(prompt)

    # Initialize beam at pos = 0 (FIXED '#' if ct[0] == '#')
    c0i = AI[ct[0]]
    seed_states: List[BeamState] = []

    if prefix:
        p0i = AI[prefix[0]]
        # Validate # constraints at pos 0
        if c0i == HASH and p0i != HASH:
            raise ValueError("Prefix[0] must be '#' because ciphertext[0] is '#'.")
        if c0i != HASH and p0i == HASH:
            raise ValueError("Prefix[0] cannot be '#' because ciphertext[0] is not '#'.")
        c2p, p2c = make_empty_key_fixed_hash()
        c2p[c0i] = p0i; p2c[p0i] = c0i
        if logp_next_after_prompt is not None:
            g0 = g_prompt + float(logp_next_after_prompt[p0i].item()) + prior_w * float(prior_vec[p0i].item())
            h0 = h_after_last
        else:
            g0 = prior_w * float(prior_vec[p0i].item())
            h0 = None
        seed_states.append(BeamState(g=g0, pt=ALPH[p0i],
                                     c2p=c2p, p2c=p2c, h_prev=h0, last_idx=p0i))
    else:
        if c0i == HASH:
            # FORCE '#→#' at start, no branching
            c2p, p2c = make_empty_key_fixed_hash()
            p0i = HASH
            if logp_next_after_prompt is not None:
                g0 = g_prompt + float(logp_next_after_prompt[p0i].item())
                h0 = h_after_last
            else:
                g0 = 0.0
                h0 = None
            seed_states.append(BeamState(g=g0, pt=ALPH[p0i],
                                         c2p=c2p, p2c=p2c, h_prev=h0, last_idx=p0i))
        else:
            # enumerate all possible first letters except '#'
            if logp_next_after_prompt is not None:
                base_scores = logp_next_after_prompt.clone()
            else:
                base_scores = torch.zeros(V, device=device)
            scores0 = base_scores + prior_w * prior_vec
            if gumbel > 0.0:
                scores0 = scores0 + gumbel_noise_like(scores0, gumbel)
            order = torch.argsort(scores0, descending=True).tolist()
            for p0i in order:
                if p0i == HASH:
                    continue  # cannot map non-# cipher to '#'
                c2p, p2c = make_empty_key_fixed_hash()
                c2p[c0i] = p0i; p2c[p0i] = c0i
                if logp_next_after_prompt is not None:
                    g0 = g_prompt + float(base_scores[p0i].item()) + prior_w * float(prior_vec[p0i].item())
                    h0 = h_after_last
                else:
                    g0 = prior_w * float(prior_vec[p0i].item())
                    h0 = None
                seed_states.append(BeamState(g=g0, pt=ALPH[p0i],
                                             c2p=c2p, p2c=p2c, h_prev=h0, last_idx=p0i))

    seed_states.sort(key=lambda s: rank_score(s.g, len(s.pt), alpha), reverse=True)
    states = seed_states[:beam_size]

    # Track where provided PT falls off
    pt_falloff_index = -1
    if pt_actual:
        alive = any(st.pt == pt_actual[:1] for st in states)
        if not alive:
            pt_falloff_index = 0

    # Main loop
    for pos in range(1, T):
        ci = AI[ct[pos]]
        forced_pi = AI[prefix[pos]] if (pos < len(prefix)) else None
        if ci != HASH and forced_pi == HASH:
            # invalid constraint: non-# cipher forced to '#'
            continue

        B = len(states)
        last_idx_batch = torch.tensor([st.last_idx for st in states], dtype=torch.long, device=device)

        if states[0].h_prev is None:
            h_prev_batch = None
        else:
            # Stack per-beam 2-D states [L,H] into [L,B,H]
            hs = torch.stack([st.h_prev[0] for st in states], dim=1)
            cs = torch.stack([st.h_prev[1] for st in states], dim=1)
            h_prev_batch = (hs, cs)

        logits_next, h_after_last_batch = batch_one_step(model, last_idx_batch, h_prev_batch)
        logp_next = F.log_softmax(logits_next, dim=-1)

        cand_rank: List[Tuple[float, BeamState]] = []
        h_rows = [(h_after_last_batch[0][:, i, :], h_after_last_batch[1][:, i, :]) for i in range(B)]

        for i, st in enumerate(states):
            row_logp = logp_next[i]
            h_row = h_rows[i]

            # If current cipher is '#', mapping is fixed to '#'
            if ci == HASH:
                pi = HASH
                if forced_pi is not None and pi != forced_pi:
                    continue
                g_new = st.g + float(row_logp[pi].item())
                pt_new = st.pt + ALPH[pi]
                st2 = BeamState(g=g_new, pt=pt_new, c2p=st.c2p, p2c=st.p2c, h_prev=h_row, last_idx=pi)
                cand_rank.append((rank_score(st2.g, len(st2.pt), alpha), st2))
                continue

            # If already mapped, no branching
            if st.c2p[ci] != -1:
                pi = st.c2p[ci]
                if forced_pi is not None and pi != forced_pi:
                    continue
                g_new = st.g + float(row_logp[pi].item())
                pt_new = st.pt + ALPH[pi]
                st2 = BeamState(g=g_new, pt=pt_new, c2p=st.c2p, p2c=st.p2c, h_prev=h_row, last_idx=pi)
                cand_rank.append((rank_score(st2.g, len(st2.pt), alpha), st2))
            else:
                # Need to select a new plaintext letter
                if forced_pi is not None:
                    if forced_pi == HASH:
                        continue  # cannot map non-# cipher to '#'
                    if st.p2c[forced_pi] != -1 and st.p2c[forced_pi] != ci:
                        continue
                    pi_list = [forced_pi]
                else:
                    used_mask = torch.tensor([st.p2c[p] != -1 for p in range(V)],
                                             dtype=torch.bool, device=device)
                    used_mask[HASH] = True  # never choose '#'
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
                    if pi == HASH:
                        continue
                    if st.p2c[pi] != -1 and st.p2c[pi] != ci:
                        continue
                    c2p = st.c2p[:]; p2c = st.p2c[:]
                    c2p[ci] = pi; p2c[pi] = ci
                    g_new = st.g + float(row_logp[pi].item()) + prior_w * float(prior_vec[pi].item())
                    pt_new = st.pt + ALPH[pi]
                    st2 = BeamState(g=g_new, pt=pt_new, c2p=c2p, p2c=p2c, h_prev=h_row, last_idx=pi)
                    cand_rank.append((rank_score(st2.g, len(st2.pt), alpha), st2))

        if not cand_rank:
            break

        cand_rank.sort(key=lambda x: x[0], reverse=True)
        states = [st for _, st in cand_rank[:beam_size]]

        if pt_actual and pt_falloff_index == -1:
            alive = any(st.pt == pt_actual[:pos + 1] for st in states)
            if not alive:
                pt_falloff_index = pos

    if not states:
        return "", list(range(V)), -1e9, [], pt_falloff_index, prompt_clean

    best = max(states, key=lambda s: s.g)
    c2p_full = complete_key(best.c2p)
    return best.pt, c2p_full, best.g, states, pt_falloff_index, prompt_clean


# ------------------------------
# Local refinement (under prompt+pt objective)
# ------------------------------

@torch.no_grad()
def refine_swaps(model: AWDCharLSTM,
                 ct_clean: str,
                 c2p_full_idx: List[int],
                 prompt: str = "",
                 rounds: int = 1,
                 device: str = 'cuda'):
    """Local 2-swap refinement under the full LM; objective is BPC on (prompt + plaintext)."""
    present = sorted(set(ct_clean))
    present_idx = [AI[c] for c in present]
    present_idx = [i for i in present_idx if i != HASH]  # never swap '#'
    key = c2p_full_idx[:]  # V-length list of ints

    def full_bpc(pt: str) -> float:
        return bpc_for_text_local(model, keep_text(prompt) + pt, device=device)

    pt = apply_key_indices(key, ct_clean)
    best_bpc = full_bpc(pt)
    improved = True
    r = 0
    while improved and r < max(0, rounds):
        improved = False
        r += 1
        for ii, i_c in enumerate(present_idx):
            for j_c in present_idx[ii + 1:]:
                if i_c == j_c:
                    continue
                k2 = key[:]
                k2[i_c], k2[j_c] = k2[j_c], k2[i_c]
                pt2 = apply_key_indices(k2, ct_clean)
                b = full_bpc(pt2)
                if b < best_bpc:
                    key, best_bpc, improved = k2, b, True
    final_pt = apply_key_indices(key, ct_clean)
    return final_pt, key, best_bpc


# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=800, help="beam size")
    ap.add_argument("--topk", type=int, default=12, help="branch count when a new cipher letter appears")
    ap.add_argument("--prior_w", type=float, default=0.10, help="weight on unigram prior (nats)")
    ap.add_argument("--alpha", type=float, default=0.0, help="length normalization: rank = g / len^alpha")
    ap.add_argument("--gumbel", type=float, default=0.0, help="Gumbel noise scale for branching diversity")
    ap.add_argument("--prefix", type=str, default="", help="optional known plaintext prefix (uses '#', a..z)")
    ap.add_argument("--prompt", type=str, default="", help="LM-only warm-up text (uses '#', a..z); contributes to score and initializes state; does not constrain the key")
    ap.add_argument("--refine", type=int, default=1, help="rounds of local 2-swap refinement")
    ap.add_argument("--seed", type=int, default=0, help="random seed (affects gumbel)")
    ap.add_argument("--nbest", type=int, default=10, help="how many final hypotheses to list")
    ap.add_argument("--pt", type=str, default="", help="actual plaintext (for diagnostics only; not used in decoding)")
    args = ap.parse_args()

    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)  # supports 26 or 27; your model should be 27 for '#'

    with open(args.cipherfile, "r", encoding="utf8") as f:
        ct_raw = f.read()
    ct = keep_text(ct_raw)

    # Decode
    pt_beam, c2p_full, g, end_states, pt_falloff, prompt_clean = beam_decode_v12(
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

    # N-best (sorted by BPC on prompt+pt), one line each: "<bpc:.3f> <plaintext>"
    scored = []
    for st in end_states:
        bpc = bpc_for_text_local(model, prompt_clean + st.pt, device=device)
        scored.append((bpc, st.pt))
    scored.sort(key=lambda x: x[0])
    for bpc, pt in scored[:max(1, args.nbest)]:
        print(f"{bpc:.3f} {pt}")

    # Optional refine of the single best under prompt+pt objective
    if args.refine > 0:
        pt_final, c2p_final, bpc = refine_swaps(model, ct, c2p_full, prompt=prompt_clean, rounds=args.refine, device=device)
        print("PLAINTEXT:\n", pt_final)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(ALPH[p] for p in c2p_final))
    else:
        bpc = bpc_for_text_local(model, prompt_clean + pt_beam, device=device)
        print("PLAINTEXT:\n", pt_beam)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(ALPH[p] for p in c2p_full))

    if args.pt:
        print("PT_FALLOFF_INDEX:", pt_falloff)


if __name__ == "__main__":
    main()
