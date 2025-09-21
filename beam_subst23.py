#!/usr/bin/env python3
"""
beam_subst23.py — Language-agnostic beam search for monoalphabetic and homophonic substitution.
Based on beam_subst22.py.
"""
import argparse, math, random, heapq, io, warnings
from typing import List, Tuple, Optional
from array import array

import torch
import torch.nn.functional as F
import torch.nn as nn

# ---- minimal model/loader ----

class AWDCharLSTM(nn.Module):
    def __init__(self, vocab_size=27, emb=512, hidden=512, layers=3,
                 p_in=0.0, p_h=0.0, p_out=0.0, tie_weights=False):
        super().__init__()
        self.encoder = nn.Embedding(vocab_size, emb)
        self.drop_in  = nn.Dropout(p_in)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=p_h)
        self.drop_out = nn.Dropout(p_out)
        self.decoder = nn.Linear(hidden, vocab_size, bias=False)
        if tie_weights:
            assert emb == hidden, "tie_weights requires emb==hidden"
            self.decoder.weight = self.encoder.weight
        self.vocab_size = vocab_size
        self.emb_dim = emb
        self.hidden_dim = hidden
        self.layers = layers
    def forward(self, x, h=None):
        x = self.drop_in(self.encoder(x))
        out, h = self.lstm(x, h)
        out = self.drop_out(out)
        logits = self.decoder(out)
        return logits, h

def safe_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return torch.load(path, map_location=device)

def _alphabet_for_vocab_fallback(vsz: int) -> str:
    """Fallback for old checkpoints without saved vocab."""
    if vsz == 26: return "abcdefghijklmnopqrstuvwxyz"
    if vsz == 27: return "#abcdefghijklmnopqrstuvwxyz"
    raise ValueError(f"Unsupported vocab size {vsz}")

def _infer_layers_from_state(sd) -> int:
    layers = set()
    for k in sd.keys():
        if k.startswith("lstm.weight_ih_l"):
            try:
                layers.add(int(k.split("lstm.weight_ih_l", 1)[1].split('.')[0]))
            except Exception:
                pass
    if layers: return max(layers) + 1
    for k in sd.keys():
        if k.startswith("lstm.weight_hh_l"):
            try:
                layers.add(int(k.split("lstm.weight_hh_l", 1)[1].split('.')[0]))
            except Exception:
                pass
    return 3

def load_model(ckpt_path: str, device: str = "cuda"):
    """Load model and extract alphabet from checkpoint."""
    ckpt = safe_load(ckpt_path, device)

    # Extract state dict - handle multiple formats
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith(('encoder.', 'lstm.', 'decoder.')) for k in ckpt.keys()):
        # Old format: direct state dict
        sd = ckpt
    else:
        sd = ckpt

    # Get model dimensions
    if "encoder.weight" in sd:
        vocab_size, emb = sd["encoder.weight"].shape
    else:
        for k in sd:
            if k.endswith("encoder.weight"):
                vocab_size, emb = sd[k].shape; break
        else:
            raise KeyError("encoder.weight not found")
    if "decoder.weight" in sd:
        _, hidden = sd["decoder.weight"].shape
    else:
        for k in sd:
            if k.startswith("lstm.weight_hh_l0"):
                hidden = sd[k].shape[1]; break
        else:
            raise KeyError("decoder.weight not found")
    layers = _infer_layers_from_state(sd)

    # Extract alphabet from checkpoint, with fallback for old checkpoints
    if isinstance(ckpt, dict) and 'vocab' in ckpt:
        alphabet = ckpt['vocab']
        print("Using alphabet from checkpoint")
    else:
        alphabet = _alphabet_for_vocab_fallback(vocab_size)
        print(f"Warning: No alphabet in checkpoint, falling back to {alphabet}")

    # Extract character priors if available
    char_priors = None
    if isinstance(ckpt, dict) and 'char_priors' in ckpt:
        char_priors = ckpt['char_priors']

    model = AWDCharLSTM(vocab_size=vocab_size, emb=emb, hidden=hidden, layers=layers,
                        p_in=0.0, p_h=0.0, p_out=0.0, tie_weights=False).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, alphabet, char_priors

# ---- prior P0 helpers ----

def clean_text(s: str, alphabet: str) -> str:
    s = (s or "").lower()
    allow = set(alphabet)
    return "".join(ch for ch in s if ch in allow)

def prior_from_text(text: str, alphabet: str, smoothing: float = 1e-6) -> torch.Tensor:
    s = clean_text(text, alphabet)
    counts = {c: 0.0 for c in alphabet}
    for ch in s:
        counts[ch] += 1.0
    total = sum(counts.values()) + smoothing * len(alphabet)
    probs = torch.tensor([(counts[c] + smoothing) / total for c in alphabet], dtype=torch.float32)
    return probs

def prior_uniform(alphabet: str) -> torch.Tensor:
    return torch.full((len(alphabet),), 1.0/len(alphabet), dtype=torch.float32)

# ---- backpointers ----

class PTNodes:
    __slots__ = ("prev","ch")
    def __init__(self):
        self.prev = array('i')
        self.ch   = array('b')
    def add(self, parent: int, ch_idx: int) -> int:
        self.prev.append(parent)
        self.ch.append(ch_idx)
        return len(self.prev) - 1

def reconstruct(nodes: PTNodes, node_id: int, alphabet: str) -> str:
    out = []
    while node_id != -1:
        out.append(alphabet[nodes.ch[node_id]])
        node_id = nodes.prev[node_id]
    out.reverse()
    return "".join(out)

# ---- prompt ----

@torch.no_grad()
def run_prompt(model: AWDCharLSTM, prompt: str, alphabet: str):
    """Feed prompt to get hidden state and next-char logprobs after last prompt char."""
    device = next(model.parameters()).device
    idx = {c: i for i,c in enumerate(alphabet)}
    s = clean_text(prompt, alphabet)
    if not s:
        return None, None
    h = None
    for t in range(1, len(s)):
        x = torch.tensor([[idx[s[t-1]]]], dtype=torch.long, device=device)
        _, h = model(x, h)
    x_last = torch.tensor([[idx[s[-1]]]], dtype=torch.long, device=device)
    logits, h_after = model(x_last, h)
    logp_next = F.log_softmax(logits[0, -1, :], dim=-1)  # [V]
    h_after = (h_after[0].squeeze(1), h_after[1].squeeze(1))
    return logp_next, h_after

# ---- homophonic support ----


def complete_homophonic_key(c2p: array, V_cipher: int, V: int):
    """Complete partial homophonic key by mapping unmapped cipher symbols to plaintext 0"""
    complete_c2p = list(c2p)
    for ci in range(V_cipher):
        if complete_c2p[ci] == -1:
            complete_c2p[ci] = 0  # Map to first plaintext symbol
    return complete_c2p

# ---- beam state ----

class BeamState:
    __slots__ = ("g","h","node","c2p","p2c","h_prev","last_idx","length","rest")
    def __init__(self, g: float, h: float, node: int, c2p: array, p2c,
                 h_prev, last_idx: int, length: int, rest: str = ""):
        self.g = g
        self.h = h  # rest-cost heuristic
        self.node = node
        self.c2p = c2p
        self.p2c = p2c  # array for monoalphabetic, list of sets for homophonic
        self.h_prev = h_prev
        self.last_idx = last_idx
        self.length = length
        self.rest = rest  # sampled continuation text

# ---- rest-cost heuristic ----


def compute_rest_cost(model, alphabet: str, ct: str, current_pos: int,
                     h_state, c2p_arr: array, cipher_A2I: dict, device, M: int = 0, last_char_idx: int = 0, p2c=None, homophonic_limit: int = 1) -> tuple:
    """
    Compute rest-cost heuristic h for remaining cipher text.
    If M > 0: Sample M characters greedily and return total negative logprob
    If M = 0: Return 0
    Returns: (cost, predicted_text)
    """
    if M <= 0:
        return 0.0, ""

    # LSTM lookahead sampling
    remaining_length = len(ct) - current_pos
    if remaining_length <= 0:
        return 0.0, ""

    # Limit lookahead to M characters
    lookahead_len = min(M, remaining_length)

    # Prepare LSTM state
    if h_state is None:
        h_current = None
    else:
        # Convert single hidden state to batch format expected by model
        h_current = (h_state[0].unsqueeze(1).contiguous(), h_state[1].unsqueeze(1).contiguous())

    total_cost = 0.0
    sampled_text = ""

    # Find alphabet indices for a-z (skip #)
    A2I = {c: i for i, c in enumerate(alphabet)}

    # Find available plaintext characters (not at homophonic limit)
    if p2c is not None:
        candidate_indices = []
        for c in alphabet:
            if c != '#':
                idx = A2I[c]
                if len(p2c[idx]) < homophonic_limit:
                    candidate_indices.append(idx)
    else:
        candidate_indices = [A2I[c] for c in alphabet if c != '#']

    with torch.no_grad():
        for i in range(lookahead_len):
            # Feed the last character to get BOTH logits and updated state
            last_input = torch.tensor([[last_char_idx]], device=device)
            logits, h_current = model(last_input, h_current)  # Update h_current here
            logp = F.log_softmax(logits[0, -1], dim=-1)

            # Check if next cipher symbol is already mapped
            next_cipher_pos = current_pos + i
            if next_cipher_pos < len(ct):
                next_cipher_char = ct[next_cipher_pos]
                if next_cipher_char in cipher_A2I:
                    cipher_idx = cipher_A2I[next_cipher_char]
                    if cipher_idx < len(c2p_arr) and c2p_arr[cipher_idx] != -1:
                        # Use committed plaintext character
                        best_idx = c2p_arr[cipher_idx]
                        best_logp = logp[best_idx].item()
                    else:
                        # Not committed - find best among a-z candidates
                        best_logp = float('-inf')
                        best_idx = candidate_indices[0]  # fallback
                        for idx in candidate_indices:
                            if logp[idx].item() > best_logp:
                                best_logp = logp[idx].item()
                                best_idx = idx
                else:
                    # Cipher char not in vocab - find best among a-z candidates
                    best_logp = float('-inf')
                    best_idx = candidate_indices[0]  # fallback
                    for idx in candidate_indices:
                        if logp[idx].item() > best_logp:
                            best_logp = logp[idx].item()
                            best_idx = idx
            else:
                # Past end of cipher text - find best among a-z candidates
                best_logp = float('-inf')
                best_idx = candidate_indices[0]  # fallback
                for idx in candidate_indices:
                    if logp[idx].item() > best_logp:
                        best_logp = logp[idx].item()
                        best_idx = idx

            # Add to total cost (already negative log prob)
            total_cost += best_logp

            # Add character to sampled text
            sampled_text += alphabet[best_idx]

            # Update last_char_idx for next iteration (no additional model call)
            last_char_idx = best_idx

    return total_cost, sampled_text

# ---- key helpers ----


def make_empty_homophonic_key(V_cipher: int, V: int, hash_cipher_idx: int, hash_idx: int):
    """Create empty homophonic key with bidirectional # constraint."""
    c2p = array('b', [-1] * V_cipher)
    p2c = [set() for _ in range(V)]

    # Enforce bidirectional # constraint: # cipher symbol only maps to # plaintext and vice versa
    if hash_cipher_idx >= 0 and hash_idx >= 0:
        c2p[hash_cipher_idx] = hash_idx
        p2c[hash_idx].add(hash_cipher_idx)

    return c2p, p2c


def complete_homophonic_key(c2p_arr: array, V_cipher: int, V: int) -> List[int]:
    """Complete a homophonic cipher key by filling in missing mappings with unused plaintext symbols."""
    used = {p for p in c2p_arr if p != -1}
    leftover = [i for i in range(V) if i not in used]
    unmapped_count = sum(1 for ci in range(V_cipher) if c2p_arr[ci] == -1)

    if unmapped_count > len(leftover):
        # Calculate more precise minimum needed
        used_count = len(used)
        # We need enough capacity for all cipher symbols
        total_capacity_needed = V_cipher
        # Each plaintext can hold up to homophonic_limit cipher symbols
        # So we need: homophonic_limit * V >= V_cipher
        min_homophonic = math.ceil(V_cipher / V)
        raise ValueError(
            f"Cannot complete cipher key: {V_cipher} cipher symbols but only {V} plaintext symbols available.\n"
            f"Current mapping uses {used_count} plaintext symbols, leaving {len(leftover)} unused.\n"
            f"But {unmapped_count} cipher symbols still need mapping.\n"
            f"Try increasing --homophonic to at least {min_homophonic} to allow multiple cipher symbols per plaintext character."
        )

    out = list(c2p_arr)
    it = iter(leftover)
    for ci in range(V_cipher):
        if out[ci] == -1:
            out[ci] = next(it)
    return out

# ---- micro-batched forward ----

@torch.no_grad()
def one_step_micro(model: AWDCharLSTM, states: List[BeamState], micro: int):
    device = next(model.parameters()).device
    B = len(states)
    row_logps = [None]*B
    h_rows = [None]*B
    for start in range(0, B, micro):
        end = min(B, start+micro)
        batch = states[start:end]
        last_idx = torch.tensor([st.last_idx for st in batch], dtype=torch.long, device=device)
        if batch[0].h_prev is None:
            h_prev_batch = None
        else:
            hs = torch.stack([st.h_prev[0] for st in batch], dim=1)
            cs = torch.stack([st.h_prev[1] for st in batch], dim=1)
            h_prev_batch = (hs, cs)
        logits, h_after = model(last_idx.view(-1,1), h_prev_batch)
        logp = F.log_softmax(logits[:, -1, :], dim=-1)
        for j in range(end-start):
            row_logps[start+j] = logp[j]
            h_rows[start+j] = (h_after[0][:, j, :], h_after[1][:, j, :])
    return row_logps, h_rows

# ---- rank ----

def rank_score(g: float, length: int, alpha: float) -> float:
    """Final scoring for reporting - uses g only"""
    return g / (length**alpha) if (alpha > 0.0 and length > 0) else g

def rank_score_astar(g: float, h: float, length: int, alpha: float) -> float:
    """A* ranking for beam pruning - uses g + h"""
    total_cost = g + h
    return total_cost / (length**alpha) if (alpha > 0.0 and length > 0) else total_cost

# ---- bpc rescoring consistent with boundary policy ----

@torch.no_grad()
def conditional_bpc(model: AWDCharLSTM, alphabet: str, prompt: str, text: str, p0_log: Optional[torch.Tensor], device: str):
    s_prompt = clean_text(prompt, alphabet)
    s_text   = clean_text(text, alphabet)
    if len(s_text) == 0: return float("inf")
    idx = {c: i for i,c in enumerate(alphabet)}
    h = None
    if len(s_prompt) >= 1:
        for t in range(1, len(s_prompt)):
            x = torch.tensor([[idx[s_prompt[t-1]]]], dtype=torch.long, device=device)
            _, h = model(x, h)
        lastp = torch.tensor([[idx[s_prompt[-1]]]], dtype=torch.long, device=device)
        logits, h = model(lastp, h)
        logp = F.log_softmax(logits[0,-1,:], dim=-1)
        total_nll = float(-logp[idx[s_text[0]]].item())
    else:
        if p0_log is None:
            raise ValueError("p0_log is required when prompt is empty for rescoring")
        total_nll = float(-p0_log[idx[s_text[0]]].item())
    # Process subsequent characters
    for t in range(1, len(s_text)):
        x = torch.tensor([[idx[s_text[t-1]]]], dtype=torch.long, device=device)
        logits, h = model(x, h)
        logp = F.log_softmax(logits[0,-1,:], dim=-1)
        total_nll += float(-logp[idx[s_text[t]]].item())
    return (total_nll / len(s_text)) / math.log(2)

# ---- beam search (with prefix enforcement) ----

@torch.no_grad()
def beam_decode(model: AWDCharLSTM,
                ct_raw: str,
                alphabet: str,
                beam_size: int,
                micro: int,
                topk_expand: int,
                prior_w: float,
                alpha: float,
                gumbel: float,
                prefix: str,
                prompt: str,
                p0_log: Optional[torch.Tensor],
                char_priors: Optional[torch.Tensor],
                pt_debug: str = "",
                homophonic_limit: int = 0,
                incremental: bool = False,
                restcost: int = 0):

    device = next(model.parameters()).device
    V = len(alphabet)
    A2I = {c:i for i,c in enumerate(alphabet)}
    HASH = A2I["#"] if alphabet[0] == "#" else None

    # Cipher text will be processed based on mode (homophonic vs monoalphabetic)

    # Clean and prepare debug plaintext
    pt_debug_clean = clean_text(pt_debug, alphabet)
    pt_falloff_index = -1

    pref = clean_text(prefix, alphabet)
    pref_len = len(pref)

    nodes = PTNodes()
    def add_node(parent, ch_idx): return nodes.add(parent, ch_idx)

    # Unified cipher processing - preserve all cipher characters
    # Note: homophonic_limit=1 represents simple substitution
    ct_raw_clean = ct_raw.strip()
    cipher_vocab = "".join(sorted(set(ct_raw_clean)))
    V_cipher = len(cipher_vocab)
    cipher_A2I = {c: i for i, c in enumerate(cipher_vocab)}
    hash_cipher_idx = cipher_A2I.get("#", -1)
    # Update ct to use raw cipher text
    ct = ct_raw_clean

    # Check for empty cipher text
    if len(ct) == 0:
        return "", list(range(V_cipher)), 0.0, ([], PTNodes()), -1

    # Check for prefix/ciphertext mismatch with #
    if pref_len > 0:
        ct_starts_with_hash = (HASH is not None and len(ct) > 0 and ct[0] == "#")
        prefix_starts_with_hash = (len(pref) > 0 and pref[0] == "#")
        if ct_starts_with_hash and not prefix_starts_with_hash:
            print("WARNING: Ciphertext starts with '#' but prefix doesn't. Consider using --prefix '#...'")
        elif not ct_starts_with_hash and prefix_starts_with_hash:
            print("WARNING: Prefix starts with '#' but ciphertext doesn't. This will likely fail.")

    logp_after_prompt, h_after_prompt = run_prompt(model, prompt, alphabet)

    # ---- seeding (t=0) ----
    c0 = ct[0]
    c0i = cipher_A2I[c0]
    seed_states: List[BeamState] = []

    def add_seed(p0i: int, g0: float, h0):
        node0 = add_node(-1, p0i)
        c2p, p2c = make_empty_homophonic_key(V_cipher, V, hash_cipher_idx, HASH if HASH is not None else -1)
        # Assign first cipher symbol if not #
        if hash_cipher_idx == -1 or c0i != hash_cipher_idx:
            c2p[c0i] = p0i
            p2c[p0i].add(c0i)
        h0_rest, rest0 = compute_rest_cost(model, alphabet, ct, 1, h0, c2p, cipher_A2I, device, restcost, p0i, p2c, homophonic_limit)  # After first character
        st = BeamState(g=g0, h=h0_rest, node=node0, c2p=c2p, p2c=p2c, h_prev=h0, last_idx=p0i, length=1, rest=rest0)
        seed_states.append(st)

    force_first = (pref_len > 0) and (0 < pref_len)
    if force_first:
        # Force to prefix[0]
        forced_pi = A2I[pref[0]]
        if HASH is not None and hash_cipher_idx >= 0 and c0i == hash_cipher_idx and forced_pi != HASH:
            pass  # impossible - cipher # must map to plaintext #
        else:
            if logp_after_prompt is not None:
                g0 = float(logp_after_prompt[forced_pi].item())
                h0 = h_after_prompt
            else:
                if p0_log is None: raise ValueError("Need P0 when no prompt is provided")
                g0 = float(p0_log[forced_pi].item())
                h0 = None
            add_seed(forced_pi, g0, h0)
    else:
        # normal boundary policy (possibly fixed '#')
        if hash_cipher_idx >= 0 and c0i == hash_cipher_idx:
            p0i = HASH
            if logp_after_prompt is not None:
                g0 = float(logp_after_prompt[p0i].item()); h0 = h_after_prompt
            else:
                if p0_log is None: raise ValueError("Need P0 when no prompt is provided")
                g0 = float(p0_log[p0i].item()); h0 = None
            add_seed(p0i, g0, h0)
        else:
            if logp_after_prompt is not None:
                base = logp_after_prompt.clone()
                if gumbel > 0.0:
                    u = torch.rand_like(base); base = base - torch.log(-torch.log(u.clamp_min(1e-12))) * gumbel
                if HASH is not None: base[HASH] = float('-inf')
                order = torch.argsort(base, descending=True).tolist()
                for p0i in order:
                    add_seed(p0i, float(base[p0i].item()), h_after_prompt)
            else:
                if p0_log is None: raise ValueError("Need P0 when no prompt is provided")
                base = p0_log.clone()
                if HASH is not None: base[HASH] = float('-inf')
                if gumbel > 0.0:
                    u = torch.rand_like(base); base = base - torch.log(-torch.log(u.clamp_min(1e-12))) * gumbel
                order = torch.argsort(base, descending=True).tolist()
                for p0i in order:
                    add_seed(p0i, float(base[p0i].item()), None)

    if not seed_states:
        return "", list(range(V)), -1e9, ([], nodes), -1
    seed_states.sort(key=lambda s: rank_score_astar(s.g, s.h, s.length, alpha), reverse=True)
    states = seed_states[:beam_size]

    # Show initial character if incremental output requested
    if incremental and states:
        if incremental == "rest":
            best_st = max(states, key=lambda s: s.g + s.h)
        else:
            best_st = max(states, key=lambda s: s.g)
        best_pt_so_far = reconstruct(nodes, best_st.node, alphabet)

        # Calculate BPC for initial character
        best_bpc = conditional_bpc(model, alphabet, prompt, best_pt_so_far,
                                  p0_log if prompt == "" else None, str(device))

        # Show PT BPC if debugging with ground truth
        if pt_debug_clean:
            target_len = 1
            if target_len <= len(pt_debug_clean):
                pt_prefix = pt_debug_clean[:target_len]
                pt_bpc = conditional_bpc(model, alphabet, prompt, pt_prefix,
                                       p0_log if prompt == "" else None, str(device))
                print(f"bpc={best_bpc:.3f} (pt bpc={pt_bpc:.3f}) {best_pt_so_far} s.g={best_st.g:.3f} s.h={best_st.h:.3f} s.rest={best_st.rest}")
            else:
                print(f"bpc={best_bpc:.3f} {best_pt_so_far} s.g={best_st.g:.3f} s.h={best_st.h:.3f} s.rest={best_st.rest}")
        else:
            print(f"bpc={best_bpc:.3f} {best_pt_so_far} s.g={best_st.g:.3f} s.h={best_st.h:.3f} s.rest={best_st.rest}")

    # ---- main ----
    for pos in range(1, len(ct)):
        ci = cipher_A2I[ct[pos]]
        row_logps, h_rows = one_step_micro(model, states, micro)
        candidates: List[Tuple[float, BeamState]] = []
        # Collect rest-cost requests for batching
        rest_cost_requests = []
        candidate_data = []  # Store data needed to create BeamState objects

        # Are we inside the prefix window at this pos?
        inside_prefix = (pref_len > 0) and (pos < pref_len)
        forced_pi = A2I[pref[pos]] if inside_prefix else None

        for i, st in enumerate(states):
            row = row_logps[i]
            hrow = h_rows[i]

            # If ciphertext is '#', force '#'
            if hash_cipher_idx >= 0 and ci == hash_cipher_idx:
                pi = HASH
                # mapping consistency
                if st.c2p[ci] not in (-1, pi) or (ci not in st.p2c[pi]):
                    continue
                node2 = nodes.add(st.node, pi)
                g_new = st.g + float(row[pi].item())
                h_new, rest_new = compute_rest_cost(model, alphabet, ct, pos+1, hrow, st.c2p, cipher_A2I, device, restcost, pi, st.p2c, homophonic_limit)  # Rest cost after this position
                st2 = BeamState(g=g_new, h=h_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                h_prev=hrow, last_idx=pi, length=st.length+1, rest=rest_new)
                candidates.append((rank_score_astar(st2.g, st2.h, st2.length, alpha), st2))
                continue

            # If we're inside prefix, force the plaintext symbol
            if inside_prefix:
                pi = forced_pi
                # Enforce bidirectional # constraint
                if hash_cipher_idx >= 0 and HASH >= 0:
                    if ci == hash_cipher_idx and pi != HASH:
                        continue  # # cipher symbol can only map to # plaintext
                    if pi == HASH and ci != hash_cipher_idx:
                        continue  # # plaintext can only come from # cipher symbol
                elif HASH >= 0 and pi == HASH:
                    # If no # in cipher, no cipher symbol should map to # plaintext
                    continue

                # mapping consistency
                if st.c2p[ci] not in (-1, pi) or (st.c2p[ci] != -1 and st.c2p[ci] != pi):
                    continue
                # Check homophonic limit
                if st.c2p[ci] == -1 and len(st.p2c[pi]) >= homophonic_limit:
                    continue  # Skip - already at max homophones
                # apply mapping if new
                if st.c2p[ci] == -1:
                    c2p_new = array('b', st.c2p)
                    p2c_new = [s.copy() for s in st.p2c]
                    c2p_new[ci] = pi
                    p2c_new[pi].add(ci)
                    node2 = nodes.add(st.node, pi)
                    g_new = st.g + float(row[pi].item())
                    h_new, rest_new = compute_rest_cost(model, alphabet, ct, pos+1, hrow, c2p_new, cipher_A2I, device, restcost, pi, p2c_new, homophonic_limit)  # Rest cost after this position
                    st2 = BeamState(g=g_new, h=h_new, node=node2, c2p=c2p_new, p2c=p2c_new,
                                    h_prev=hrow, last_idx=pi, length=st.length+1, rest=rest_new)
                    candidates.append((rank_score_astar(st2.g, st2.h, st2.length, alpha), st2))
                else:
                    # Already mapped
                    node2 = nodes.add(st.node, pi)
                    g_new = st.g + float(row[pi].item())
                    h_new, rest_new = compute_rest_cost(model, alphabet, ct, pos+1, hrow, st.c2p, cipher_A2I, device, restcost, pi, st.p2c, homophonic_limit)  # Rest cost after this position
                    st2 = BeamState(g=g_new, h=h_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                    h_prev=hrow, last_idx=pi, length=st.length+1, rest=rest_new)
                    candidates.append((rank_score_astar(st2.g, st2.h, st2.length, alpha), st2))
                continue

            # Normal expansion (not in prefix window)
            # Homophonic expansion
            mapped = st.c2p[ci]
            if mapped != -1:
                # Cipher symbol already mapped
                pi = int(mapped)
                node2 = nodes.add(st.node, pi)
                g_new = st.g + float(row[pi].item())
                h_new, rest_new = compute_rest_cost(model, alphabet, ct, pos+1, hrow, st.c2p, cipher_A2I, device, restcost, pi, st.p2c, homophonic_limit)  # Rest cost after this position
                st2 = BeamState(g=g_new, h=h_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                h_prev=hrow, last_idx=pi, length=st.length+1, rest=rest_new)
                candidates.append((rank_score_astar(st2.g, st2.h, st2.length, alpha), st2))
            else:
                # Cipher symbol not yet mapped - choose plaintext assignment
                scores = row.clone()
                if prior_w > 0.0 and char_priors is not None:
                    eta = char_priors.to(scores.device)
                    scores = scores + prior_w * torch.log(eta.clamp_min(1e-30))

                # Try multiple plaintext assignments
                _, idxs = torch.topk(scores, min(topk_expand, V))
                for pi in idxs.tolist():
                    # Enforce bidirectional # constraint
                    if hash_cipher_idx >= 0 and HASH >= 0:
                        if ci == hash_cipher_idx and pi != HASH:
                            continue  # # cipher symbol can only map to # plaintext
                        if pi == HASH and ci != hash_cipher_idx:
                            continue  # # plaintext can only come from # cipher symbol
                    elif HASH >= 0 and pi == HASH:
                        # If no # in cipher, no cipher symbol should map to # plaintext
                        continue

                    # Check homophonic limit
                    if len(st.p2c[pi]) >= homophonic_limit:
                        continue  # Skip - already at max homophones

                    c2p_new = array('b', st.c2p)
                    p2c_new = [s.copy() for s in st.p2c]
                    c2p_new[ci] = pi
                    p2c_new[pi].add(ci)
                    node2 = nodes.add(st.node, pi)
                    g_new = st.g + float(row[pi].item())
                    h_new, rest_new = compute_rest_cost(model, alphabet, ct, pos+1, hrow, c2p_new, cipher_A2I, device, restcost, pi, p2c_new, homophonic_limit)  # Rest cost after this position
                    st2 = BeamState(g=g_new, h=h_new, node=node2, c2p=c2p_new, p2c=p2c_new,
                                    h_prev=hrow, last_idx=pi, length=st.length+1, rest=rest_new)
                    candidates.append((rank_score_astar(st2.g, st2.h, st2.length, alpha), st2))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)


        # Check if pt_debug still on beam
        if pt_debug_clean and pt_falloff_index == -1:
            target_len = pos + 1
            if target_len <= len(pt_debug_clean):
                target_prefix = pt_debug_clean[:target_len]
                found_on_beam = False
                for _, st in candidates[:beam_size]:
                    pt_so_far = reconstruct(nodes, st.node, alphabet)
                    if pt_so_far == target_prefix:
                        found_on_beam = True
                        break
                if not found_on_beam:
                    pt_falloff_index = pos
                    print(f"*** True plaintext fell off beam at position {pos} ***")

        states = [st for _, st in candidates[:beam_size]]

        # Show incremental triangle output if requested
        if incremental and states:
            if incremental == "rest":
                best_st = max(states, key=lambda s: s.g + s.h)
            else:
                best_st = max(states, key=lambda s: s.g)
            best_pt_so_far = reconstruct(nodes, best_st.node, alphabet)

            # Calculate BPC for current best hypothesis
            best_bpc = conditional_bpc(model, alphabet, prompt, best_pt_so_far,
                                      p0_log if prompt == "" else None, str(device))

            # No bracketed letter in main output anymore
            beam_debug = ""

            # Show PT BPC if debugging with ground truth
            if pt_debug_clean:
                target_len = pos + 1
                if target_len <= len(pt_debug_clean):
                    pt_prefix = pt_debug_clean[:target_len]
                    pt_bpc = conditional_bpc(model, alphabet, prompt, pt_prefix,
                                           p0_log if prompt == "" else None, str(device))
                    print(f"bpc={best_bpc:.3f} (pt bpc={pt_bpc:.3f}) {best_pt_so_far} s.g={best_st.g:.3f} s.h={best_st.h:.3f} s.rest={best_st.rest}")
                else:
                    print(f"bpc={best_bpc:.3f} {best_pt_so_far} s.g={best_st.g:.3f} s.h={best_st.h:.3f} s.rest={best_st.rest}")
            else:
                print(f"bpc={best_bpc:.3f} {best_pt_so_far} s.g={best_st.g:.3f} s.h={best_st.h:.3f} s.rest={best_st.rest}")

        if device.type == "cuda" and (pos % 32 == 0):
            torch.cuda.empty_cache()

    if not states:
        return "", complete_homophonic_key(array('b', [-1]*V_cipher), V_cipher, V), -1e9, ([], nodes), -1

    best = max(states, key=lambda s: s.g)
    best_pt = reconstruct(nodes, best.node, alphabet)
    c2p_full = complete_homophonic_key(best.c2p, V_cipher, V)
    return best_pt, c2p_full, best.g, (states, nodes), pt_falloff_index

# ---- CLI ----

def read_all(path: str) -> str:
    if path == "-" or path == "/dev/stdin":
        return sys.stdin.read()
    with io.open(path, "r", encoding="utf8", errors="ignore") as f:
        return f.read()

def read_text_or_file(text_or_path: str) -> str:
    """Read text from a file path or return the string directly.

    If the input looks like a file path (exists as a file), read from file.
    Otherwise, treat it as a literal string.
    """
    if not text_or_path:
        return ""

    # Check if it's a file that exists
    import os
    if os.path.isfile(text_or_path):
        try:
            with io.open(text_or_path, "r", encoding="utf8", errors="ignore") as f:
                return f.read().strip()
        except Exception:
            # If file reading fails, treat as literal string
            pass

    # Treat as literal string
    return text_or_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=800)
    ap.add_argument("--micro", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=26)
    ap.add_argument("--prior_w", type=float, default=0.10, help="character prior weight for NEW assignments at t>0")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--gumbel", type=float, default=0.0)
    ap.add_argument("--prefix", type=str, default="", help="force initial plaintext characters (filename or quoted string, include # if needed)")
    ap.add_argument("--prompt", type=str, default="", help="warm-start text for language model (filename or quoted string)")
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--pt", type=str, default="", help="true plaintext for debugging (filename or quoted string, include # if present)")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp32")
    ap.add_argument("--prior_text", type=str, default="")
    ap.add_argument("--prior", choices=["data","uniform"], default="data")
    ap.add_argument("--smooth", type=float, default=1e-6)
    ap.add_argument("--homophonic", type=int, default=1, help="max cipher symbols per plaintext (1=simple substitution)")
    ap.add_argument("--incremental", nargs="?", const="g", choices=["g", "rest"], help="show incremental triangle output during beam search (g=use s.g only, rest=use s.g+s.h)")
    ap.add_argument("--restcost", type=int, default=0, help="use LSTM lookahead for rest-cost heuristic with M character window (0=simple log heuristic)")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Don't seed random for rest-cost heuristic randomness
    torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    torch.set_grad_enabled(False)

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model, alphabet, saved_priors = load_model(args.model, device=str(device))
    if use_cuda and args.dtype != "fp32":
        if args.dtype == "fp16": model = model.to(dtype=torch.float16)
        elif args.dtype == "bf16": model = model.to(dtype=torch.bfloat16)

    # Determine character priors
    if args.prior_text and os.path.exists(args.prior_text):
        p0 = prior_from_text(read_all(args.prior_text), alphabet, smoothing=args.smooth)
        print("Using character priors from", args.prior_text)
    elif saved_priors is not None and args.prior == "data":
        p0 = saved_priors
        print("Using character priors from model checkpoint")
    elif args.prior == "uniform":
        p0 = prior_uniform(alphabet)
        print("Using uniform character priors")
    else:
        # Default: try saved priors, fall back to uniform
        if saved_priors is not None:
            p0 = saved_priors
            print("Using character priors from model checkpoint")
        else:
            p0 = prior_uniform(alphabet)
            print("Using uniform character priors (no training data available)")

    p0_log = torch.log(p0.clamp_min(1e-30)).to(device)


    # Process text/file arguments
    prefix_text = read_text_or_file(args.prefix)
    prompt_text = read_text_or_file(args.prompt)
    pt_text = read_text_or_file(args.pt)

    ct = read_all(args.cipherfile)
    try:
        best_pt, c2p_full, g_raw, pack, pt_falloff = beam_decode(
            model, ct, alphabet,
            beam_size=args.beam, micro=args.micro, topk_expand=args.topk,
            prior_w=args.prior_w, alpha=args.alpha, gumbel=args.gumbel,
            prefix=prefix_text, prompt=prompt_text,
            p0_log=p0_log if prompt_text == "" else None,
            char_priors=p0, pt_debug=pt_text, homophonic_limit=args.homophonic,
            incremental=args.incremental, restcost=args.restcost
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    states, nodes = pack

    # N-best (rescored with same boundary policy)
    topk_states = heapq.nlargest(max(1, args.nbest), states, key=lambda s: s.g)
    scored = []
    for st in topk_states:
        pt = reconstruct(nodes, st.node, alphabet)
        bpc = conditional_bpc(model, alphabet, args.prompt, pt,
                              p0_log if args.prompt == "" else None, device=str(device))
        scored.append((bpc, pt))
    scored.sort(key=lambda x: x[0])

    print(f"\nTop {len(scored)} results:")
    for i, (bpc, pt) in enumerate(scored):
        print(f"{i+1:2d}. bpc={bpc:6.3f} {pt}")

    if scored:
        # Display key with cipher alphabet above plaintext symbols
        # Reorder so plaintext line is alphabetical
        if args.homophonic > 0:
            # For homophonic mode, get cipher vocabulary from the cipher file
            cipher_text = read_all(args.cipherfile).strip()
            cipher_alphabet = "".join(sorted(set(cipher_text)))
            if len(c2p_full) == len(cipher_alphabet):
                # Create cipher->plaintext pairs and sort by plaintext
                pairs = [(cipher_alphabet[i], alphabet[c2p_full[i]]) for i in range(len(cipher_alphabet))]
                pairs.sort(key=lambda x: x[1])  # Sort by plaintext symbol
                cipher_line = "".join(pair[0] for pair in pairs)
                plaintext_line = "".join(pair[1] for pair in pairs)
                print("KEY c->p:")
                print(cipher_line)
                print(plaintext_line)
            else:
                print("KEY c->p:\n", "".join(alphabet[p] for p in c2p_full))
        else:
            # For monoalphabetic mode, use standard alphabet
            # Create cipher->plaintext pairs and sort by plaintext
            pairs = [(alphabet[i], alphabet[c2p_full[i]]) for i in range(len(alphabet))]
            pairs.sort(key=lambda x: x[1])  # Sort by plaintext symbol
            cipher_line = "".join(pair[0] for pair in pairs)
            plaintext_line = "".join(pair[1] for pair in pairs)
            print("KEY c->p:")
            print(cipher_line)
            print(plaintext_line)

    if args.pt:
        if pt_falloff >= 0:
            print(f"True plaintext fell off beam at position {pt_falloff}")
        else:
            print("True plaintext stayed on beam throughout search")

if __name__ == "__main__":
    import sys, os
    main()
