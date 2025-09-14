#!/usr/bin/env python3
"""
beam_subst13.py — Memory-tight beam search for monoalphabetic substitution (# + a..z).

Major RAM reductions vs v12:
  • Plaintext prefixes stored via backpointers (no per-beam strings)
  • c2p/p2c stored as array('b') (int8) instead of Python lists
  • N-best BPC computed only for top-N beams
  • Gold falloff check tracked as a boolean flag

Dependencies:
  • score2_lstm.py (auto-detecting 26/27 loader). We assume 27-char model for '#'.

"""

import argparse, math, random, heapq
from typing import List, Tuple, Optional
from array import array

import torch
import torch.nn.functional as F

from score2_lstm import AWDCharLSTM, load_model  # loader keeps '#'

# ---------------- Alphabet ----------------

ALPH = "#abcdefghijklmnopqrstuvwxyz"
V = len(ALPH)  # 27
AI = {c: i for i, c in enumerate(ALPH)}
HASH = AI["#"]

# Prior (ETAOIN) in nats for letters only; '#'(0) gets 0
_ENG = "etaoinshrdlcumwfgypbvkjxqz"
_PRI = {c: (27 - i) for i, c in enumerate(_ENG)}  # 26..1
_S = sum(_PRI.values())
LOGP_PRI = torch.zeros(V)
for c in _ENG:
    LOGP_PRI[AI[c]] = math.log(_PRI[c] / _S)

def keep_text(s: str) -> str:
    s = (s or "").lower()
    return "".join(ch for ch in s if ch in AI)

def gumbel_noise_like(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0.0: return torch.zeros_like(x)
    u = torch.rand_like(x)
    return -torch.log(-torch.log(torch.clamp(u, min=1e-12))) * scale

# --------------- Backpointer store ---------------

class PTNodes:
    """
    Compact plaintext prefix store:
      prev[i] : parent node id (int), -1 for root
      ch[i]   : int8 plaintext index at this node (0..26)
    """
    __slots__ = ("prev", "ch")
    def __init__(self):
        self.prev = array('i')   # parent node id
        self.ch   = array('b')   # 0..26
    def add(self, parent: int, ch_idx: int) -> int:
        self.prev.append(parent)
        self.ch.append(ch_idx)
        return len(self.prev) - 1

def reconstruct(nodes: PTNodes, node_id: int) -> str:
    out = []
    while node_id != -1:
        out.append(ALPH[nodes.ch[node_id]])
        node_id = nodes.prev[node_id]
    out.reverse()
    return "".join(out)

# --------------- One-step batched LM ---------------

@torch.no_grad()
def batch_one_step(model: AWDCharLSTM,
                   last_idx_batch: torch.Tensor,
                   h_prev_batch: Optional[Tuple[torch.Tensor, torch.Tensor]]):
    device = next(model.parameters()).device
    x = last_idx_batch.view(-1, 1).to(device)  # [B,1]
    logits, h_after = model(x, h_prev_batch)   # logits [B,1,V]
    return logits[:, -1, :], h_after           # [B,V], (h,c) [L,B,H]

# --------------- Prompt handling ---------------

@torch.no_grad()
def prompt_info(model: AWDCharLSTM, prompt: str):
    prompt = keep_text(prompt)
    if not prompt:
        return None, None, None, None, 0.0

    device = next(model.parameters()).device
    idxs = [AI[ch] for ch in prompt]
    g_prompt = 0.0
    h_prev_3d = None
    h_before_last_3d = None
    idx_last = idxs[-1]

    for t in range(1, len(idxs)):
        x = torch.tensor([[idxs[t-1]]], dtype=torch.long, device=device)
        logits, h_after_3d = model(x, h_prev_3d)
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        g_prompt += float(logp[idxs[t]].item())
        if t == len(idxs) - 1:
            h_before_last_3d = h_after_3d
        h_prev_3d = h_after_3d

    x_last = torch.tensor([[idx_last]], dtype=torch.long, device=device)
    logits_next, h_after_last_3d = model(x_last, h_before_last_3d)
    logp_next = F.log_softmax(logits_next[0, -1, :], dim=-1)

    h_before_last = (h_before_last_3d[0].squeeze(1), h_before_last_3d[1].squeeze(1)) if h_before_last_3d else None
    h_after_last  = (h_after_last_3d[0].squeeze(1),  h_after_last_3d[1].squeeze(1))

    return idx_last, h_before_last, h_after_last, logp_next, g_prompt

# --------------- Beam state ---------------

class BeamState:
    """
    Memory-lean hypothesis:
      g        : accumulated log-prob (includes prompt + first|prompt)
      node     : node id in PTNodes (backpointer head)
      c2p/p2c  : array('b') length V, -1 or 0..26
      h_prev   : per-beam hidden [L,H] tuple (views into last batch tensor)
      last_idx : int (0..26) last plaintext index
      match_g  : bool, whether prefix equals gold so far (only if --pt used)
      length   : int, plaintext length so far
    """
    __slots__ = ("g","node","c2p","p2c","h_prev","last_idx","match_g","length")
    def __init__(self, g: float, node: int, c2p: array, p2c: array,
                 h_prev: Optional[Tuple[torch.Tensor, torch.Tensor]],
                 last_idx: int, match_g: bool, length: int):
        self.g = g
        self.node = node
        self.c2p = c2p
        self.p2c = p2c
        self.h_prev = h_prev
        self.last_idx = last_idx
        self.match_g = match_g
        self.length = length

def make_empty_key_fixed_hash() -> Tuple[array, array]:
    c2p = array('b', [-1]*V)
    p2c = array('b', [-1]*V)
    c2p[HASH] = HASH
    p2c[HASH] = HASH
    return c2p, p2c

def complete_key(c2p_arr: array) -> List[int]:
    used = {p for p in c2p_arr if p != -1}
    leftover = [i for i in range(V) if i not in used]
    out = list(c2p_arr)
    it = iter(leftover)
    for ci in range(V):
        if out[ci] == -1:
            out[ci] = next(it)
    return out

def rank_score(g: float, length: int, alpha: float) -> float:
    return g / (length ** alpha) if alpha > 0.0 and length > 0 else g

# --------------- BPC for (prompt + pt) ---------------

@torch.no_grad()
def bpc_for_text_local(model: AWDCharLSTM, text: str, device: str = "cuda") -> float:
    s = keep_text(text)
    if len(s) < 2: return float("inf")
    idx = {c: i for i, c in enumerate(ALPH)}
    ids = torch.tensor([[idx[ch] for ch in s]], dtype=torch.long, device=device)
    total_nll = 0.0; total_tok = 0
    h = None
    T = ids.size(1)
    for t in range(1, T):
        x = ids[:, t-1:t]
        logits, h = model(x, h)
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        total_nll += float(-logp[ids[0, t]].item())
        total_tok += 1
    return (total_nll / total_tok) / math.log(2)

# --------------- Beam search ---------------

@torch.no_grad()
def beam_decode_v13(model: AWDCharLSTM,
                    ct: str,
                    beam_size: int = 800,
                    topk_expand: int = 12,
                    prior_w: float = 0.10,
                    alpha: float = 0.0,
                    gumbel: float = 0.0,
                    prefix: str = "",
                    pt_actual: str = "",
                    prompt: str = ""):
    device = next(model.parameters()).device
    model.eval()

    ct = keep_text(ct); T = len(ct)
    if T == 0:
        return "", list(range(V)), 0.0, [], -1, keep_text(prompt)

    prefix = keep_text(prefix)
    pt_actual = keep_text(pt_actual)
    gold_chars = [AI[c] for c in pt_actual] if pt_actual else None

    prior_vec = LOGP_PRI.to(device)

    # Prompt
    _, h_before_last, h_after_last, logp_next_after_prompt, g_prompt = prompt_info(model, prompt)
    prompt_clean = keep_text(prompt)

    # Node store
    nodes = PTNodes()

    # Seed at pos 0 (respect '#' forcing)
    c0i = AI[ct[0]]
    seed_states: List[BeamState] = []

    def make_state(p0i: int, c2p, p2c, h0, g0, match_g: bool):
        node0 = nodes.add(-1, p0i)
        return BeamState(g=g0, node=node0, c2p=c2p, p2c=p2c,
                         h_prev=h0, last_idx=p0i, match_g=match_g, length=1)

    if prefix:
        p0i = AI[prefix[0]]
        if (c0i == HASH) != (p0i == HASH):
            raise ValueError("Prefix[0] must match '#' status of ciphertext[0].")
        c2p, p2c = make_empty_key_fixed_hash()
        c2p[c0i] = p0i; p2c[p0i] = c0i
        if logp_next_after_prompt is not None:
            g0 = g_prompt + float(logp_next_after_prompt[p0i].item()) + prior_w * float(prior_vec[p0i].item())
            h0 = h_after_last
        else:
            g0 = prior_w * float(prior_vec[p0i].item()); h0 = None
        match0 = (gold_chars is not None and len(gold_chars) > 0 and p0i == gold_chars[0])
        seed_states.append(make_state(p0i, c2p, p2c, h0, g0, match0))
    else:
        if c0i == HASH:
            c2p, p2c = make_empty_key_fixed_hash()
            p0i = HASH
            if logp_next_after_prompt is not None:
                g0 = g_prompt + float(logp_next_after_prompt[p0i].item()); h0 = h_after_last
            else:
                g0 = 0.0; h0 = None
            match0 = (gold_chars is not None and len(gold_chars) > 0 and p0i == gold_chars[0])
            seed_states.append(make_state(p0i, c2p, p2c, h0, g0, match0))
        else:
            base_scores = logp_next_after_prompt.clone() if logp_next_after_prompt is not None else torch.zeros(V, device=device)
            scores0 = base_scores + prior_w * prior_vec
            if gumbel > 0.0: scores0 = scores0 + gumbel_noise_like(scores0, gumbel)
            order = torch.argsort(scores0, descending=True).tolist()
            for p0i in order:
                if p0i == HASH: continue
                c2p, p2c = make_empty_key_fixed_hash()
                c2p[c0i] = p0i; p2c[p0i] = c0i
                if logp_next_after_prompt is not None:
                    g0 = g_prompt + float(base_scores[p0i].item()) + prior_w * float(prior_vec[p0i].item())
                    h0 = h_after_last
                else:
                    g0 = prior_w * float(prior_vec[p0i].item()); h0 = None
                match0 = (gold_chars is not None and len(gold_chars) > 0 and p0i == gold_chars[0])
                seed_states.append(make_state(p0i, c2p, p2c, h0, g0, match0))

    seed_states.sort(key=lambda s: rank_score(s.g, s.length, alpha), reverse=True)
    states = seed_states[:beam_size]

    pt_falloff_index = -1
    if gold_chars:
        alive = any(st.match_g for st in states)
        if not alive: pt_falloff_index = 0

    # Main loop
    for pos in range(1, T):
        ci = AI[ct[pos]]
        forced_pi = AI[prefix[pos]] if (pos < len(prefix)) else None
        if ci != HASH and forced_pi == HASH:
            continue

        B = len(states)
        last_idx_batch = torch.tensor([st.last_idx for st in states], dtype=torch.long, device=device)

        if states[0].h_prev is None:
            h_prev_batch = None
        else:
            hs = torch.stack([st.h_prev[0] for st in states], dim=1)
            cs = torch.stack([st.h_prev[1] for st in states], dim=1)
            h_prev_batch = (hs, cs)

        logits_next, h_after_last_batch = batch_one_step(model, last_idx_batch, h_prev_batch)
        logp_next = F.log_softmax(logits_next, dim=-1)

        cand_rank: List[Tuple[float, BeamState]] = []
        # create per-beam views (do NOT clone; views are fine)
        h_rows = [(h_after_last_batch[0][:, i, :], h_after_last_batch[1][:, i, :]) for i in range(B)]

        for i, st in enumerate(states):
            row_logp = logp_next[i]
            h_row = h_rows[i]

            if ci == HASH:
                pi = HASH
                if forced_pi is not None and pi != forced_pi: continue
                g_new = st.g + float(row_logp[pi].item())
                match_new = st.match_g and (gold_chars is not None and pos < len(gold_chars) and pi == gold_chars[pos]) if gold_chars else False
                node2 = nodes.add(st.node, pi)
                st2 = BeamState(g=g_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                h_prev=h_row, last_idx=pi, match_g=match_new, length=st.length+1)
                cand_rank.append((rank_score(st2.g, st2.length, alpha), st2))
                continue

            # mapped already?
            mapped = st.c2p[ci]
            if mapped != -1:
                pi = int(mapped)
                if forced_pi is not None and pi != forced_pi: continue
                g_new = st.g + float(row_logp[pi].item())
                match_new = st.match_g and (gold_chars is not None and pos < len(gold_chars) and pi == gold_chars[pos]) if gold_chars else False
                node2 = nodes.add(st.node, pi)
                st2 = BeamState(g=g_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                h_prev=h_row, last_idx=pi, match_g=match_new, length=st.length+1)
                cand_rank.append((rank_score(st2.g, st2.length, alpha), st2))
            else:
                # assign a new plaintext letter
                if forced_pi is not None:
                    if forced_pi == HASH: continue
                    if st.p2c[forced_pi] != -1 and st.p2c[forced_pi] != ci: continue
                    pi_list = [forced_pi]
                else:
                    # mask used plaintext letters
                    scores = row_logp + prior_w * LOGP_PRI.to(device)
                    if gumbel > 0.0: scores = scores + gumbel_noise_like(scores, gumbel)
                    # disallow already-used and '#' (always)
                    used_idx = [p for p in range(V) if st.p2c[p] != -1]
                    if HASH not in used_idx: used_idx.append(HASH)
                    scores[used_idx] = float('-inf')
                    k = min(topk_expand, max(0, V - len(used_idx)))
                    if k <= 0: continue
                    _, idx = torch.topk(scores, k)
                    pi_list = idx.tolist()

                for pi in pi_list:
                    if pi == HASH: continue
                    if st.p2c[pi] != -1 and st.p2c[pi] != ci: continue
                    c2p = array('b', st.c2p)
                    p2c = array('b', st.p2c)
                    c2p[ci] = pi; p2c[pi] = ci
                    g_new = st.g + float(row_logp[pi].item()) + prior_w * float(LOGP_PRI[pi].item())
                    match_new = st.match_g and (gold_chars is not None and pos < len(gold_chars) and pi == gold_chars[pos]) if gold_chars else (gold_chars is not None and pos < len(gold_chars) and pi == gold_chars[pos])
                    node2 = nodes.add(st.node, pi)
                    st2 = BeamState(g=g_new, node=node2, c2p=c2p, p2c=p2c,
                                    h_prev=h_row, last_idx=pi, match_g=match_new, length=st.length+1)
                    cand_rank.append((rank_score(st2.g, st2.length, alpha), st2))

        if not cand_rank: break

        cand_rank.sort(key=lambda x: x[0], reverse=True)
        states = [st for _, st in cand_rank[:beam_size]]

        if gold_chars and pt_falloff_index == -1:
            alive = any(st.match_g for st in states)
            if not alive: pt_falloff_index = pos

    if not states:
        return "", list(range(V)), -1e9, [], pt_falloff_index, prompt_clean

    # Best by raw g
    best = max(states, key=lambda s: s.g)
    best_pt = reconstruct(nodes, best.node)
    c2p_full = complete_key(best.c2p)
    return best_pt, c2p_full, best.g, (states, nodes), pt_falloff_index, prompt_clean

# --------------- N-best + refinement ---------------

@torch.no_grad()
def bpc_for_text(model: AWDCharLSTM, prompt: str, pt: str, device: str) -> float:
    return bpc_for_text_local(model, keep_text(prompt) + keep_text(pt), device=device)

@torch.no_grad()
def refine_swaps(model: AWDCharLSTM,
                 ct_clean: str,
                 c2p_full_idx: List[int],
                 prompt: str = "",
                 rounds: int = 1,
                 device: str = 'cuda'):
    present = sorted(set(ct_clean))
    present_idx = [AI[c] for c in present if c != '#']
    key = c2p_full_idx[:]

    def full_bpc(pt: str) -> float:
        return bpc_for_text_local(model, keep_text(prompt) + pt, device=device)

    def apply_key(k: List[int]) -> str:
        table = [ALPH[p] for p in k]
        return "".join(table[AI[ch]] for ch in ct_clean)

    pt = apply_key(key); best_bpc = full_bpc(pt)
    for _ in range(max(0, rounds)):
        improved = False
        for ii, i_c in enumerate(present_idx):
            for j_c in present_idx[ii+1:]:
                if i_c == j_c: continue
                k2 = key[:]
                k2[i_c], k2[j_c] = k2[j_c], k2[i_c]
                b = full_bpc(apply_key(k2))
                if b < best_bpc:
                    key, best_bpc, improved = k2, b, True
        if not improved: break
    return apply_key(key), key, best_bpc

# --------------- Main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=800)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--prior_w", type=float, default=0.10)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--gumbel", type=float, default=0.0)
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--pt", type=str, default="")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_grad_enabled(False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)  # should be 27-char for '#'

    with open(args.cipherfile, "r", encoding="utf8") as f:
        ct_raw = f.read()
    ct = keep_text(ct_raw)

    best_pt, c2p_full, g, pack, pt_falloff, prompt_clean = beam_decode_v13(
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
    states, nodes = pack

    # N-best: pick top args.nbest by raw g, then score (prompt+pt) for those only
    topk_states = heapq.nlargest(max(1, args.nbest), states, key=lambda s: s.g)
    scored = []
    for st in topk_states:
        pt = reconstruct(nodes, st.node)
        bpc = bpc_for_text(model, prompt_clean, pt, device=device)
        scored.append((bpc, pt))
    scored.sort(key=lambda x: x[0])
    for bpc, pt in scored:
        print(f"{bpc:.3f} {pt}")

    # Optional refine
    if args.refine > 0:
        pt_final, c2p_final, bpc = refine_swaps(model, ct, c2p_full, prompt=prompt_clean, rounds=args.refine, device=device)
        print("PLAINTEXT:\n", pt_final)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(ALPH[p] for p in c2p_final))
    else:
        bpc = bpc_for_text(model, prompt_clean, best_pt, device=device)
        print("PLAINTEXT:\n", best_pt)
        print("BPC:", bpc)
        print("KEY c->p:\n", "".join(ALPH[p] for p in c2p_full))

    if args.pt:
        print("PT_FALLOFF_INDEX:", pt_falloff)

if __name__ == "__main__":
    main()
