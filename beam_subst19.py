#!/usr/bin/env python3
"""
beam_subst17.py — Language-agnostic beam search for monoalphabetic and homophonic substitution.
Based on beam_subst16.py with homophonic cipher support.
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

def make_empty_homophonic_key(V_cipher: int, V: int, HASH_CIPHER: int, HASH: int):
    """Create empty homophonic key: c2p[ci] = pi, p2c[pi] = set of cipher indices"""
    c2p = array('b', [-1] * V_cipher)
    p2c = [set() for _ in range(V)]

    # Fix # mapping if both are defined
    if HASH_CIPHER >= 0 and HASH >= 0:
        c2p[HASH_CIPHER] = HASH
        p2c[HASH].add(HASH_CIPHER)

    return c2p, p2c

def complete_homophonic_key(c2p: array, V_cipher: int, V: int):
    """Complete partial homophonic key by mapping unmapped cipher symbols to plaintext 0"""
    complete_c2p = list(c2p)
    for ci in range(V_cipher):
        if complete_c2p[ci] == -1:
            complete_c2p[ci] = 0  # Map to first plaintext symbol
    return complete_c2p

# ---- beam state ----

class BeamState:
    __slots__ = ("g","node","c2p","p2c","h_prev","last_idx","length")
    def __init__(self, g: float, node: int, c2p: array, p2c,
                 h_prev, last_idx: int, length: int):
        self.g = g
        self.node = node
        self.c2p = c2p
        self.p2c = p2c  # array for monoalphabetic, list of sets for homophonic
        self.h_prev = h_prev
        self.last_idx = last_idx
        self.length = length

# ---- key helpers ----

def make_empty_key_fixed_hash(V: int, hash_idx: int) -> Tuple[array, array]:
    c2p = array('b', [-1]*V)
    p2c = array('b', [-1]*V)
    if 0 <= hash_idx < V:
        c2p[hash_idx] = hash_idx
        p2c[hash_idx] = hash_idx
    return c2p, p2c

def make_empty_homophonic_key(V_cipher: int, V: int, HASH_CIPHER: int, HASH: int):
    """Create empty homophonic key with bidirectional # constraint."""
    c2p = array('b', [-1] * V_cipher)
    p2c = [set() for _ in range(V)]

    # Enforce bidirectional # constraint: # cipher symbol only maps to # plaintext and vice versa
    if HASH_CIPHER >= 0 and HASH >= 0:
        c2p[HASH_CIPHER] = HASH
        p2c[HASH].add(HASH_CIPHER)

    return c2p, p2c

def complete_key(c2p_arr: array, V: int) -> List[int]:
    used = {p for p in c2p_arr if p != -1}
    leftover = [i for i in range(V) if i not in used]
    out = list(c2p_arr)
    it = iter(leftover)
    for ci in range(V):
        if out[ci] == -1:
            out[ci] = next(it)
    return out

def complete_homophonic_key(c2p_arr: array, V_cipher: int, V: int) -> List[int]:
    """Complete a homophonic cipher key by filling in missing mappings with unused plaintext symbols."""
    used = {p for p in c2p_arr if p != -1}
    leftover = [i for i in range(V) if i not in used]
    out = list(c2p_arr)
    it = iter(leftover)
    for ci in range(V_cipher):
        if out[ci] == -1:
            out[ci] = next(it) if leftover else 0  # fallback to 'a' if no symbols left
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
    return g / (length**alpha) if (alpha > 0.0 and length > 0) else g

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
                incremental: bool = False):

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

    # Check for prefix/ciphertext mismatch with #
    if pref_len > 0:
        ct_starts_with_hash = (HASH is not None and len(ct) > 0 and ct[0] == "#")
        prefix_starts_with_hash = (len(pref) > 0 and pref[0] == "#")
        if ct_starts_with_hash and not prefix_starts_with_hash:
            print("WARNING: Ciphertext starts with '#' but prefix doesn't. Consider using --prefix '#...'")
        elif not ct_starts_with_hash and prefix_starts_with_hash:
            print("WARNING: Prefix starts with '#' but ciphertext doesn't. This will likely fail.")

    nodes = PTNodes()
    def add_node(parent, ch_idx): return nodes.add(parent, ch_idx)

    # Determine cipher mode
    homophonic = homophonic_limit > 0
    if homophonic:
        # For homophonic mode, preserve case distinctions in cipher text
        ct_raw_clean = ct_raw.strip()
        cipher_vocab = "".join(sorted(set(ct_raw_clean)))
        V_cipher = len(cipher_vocab)
        cipher_A2I = {c: i for i, c in enumerate(cipher_vocab)}
        HASH_CIPHER = cipher_A2I.get("#", -1)
        # Update ct to use raw cipher text
        ct = ct_raw_clean
    else:
        # For monoalphabetic mode, also preserve all cipher characters
        ct_raw_clean = ct_raw.strip()
        cipher_vocab = "".join(sorted(set(ct_raw_clean)))
        V_cipher = len(cipher_vocab)
        cipher_A2I = {c: i for i, c in enumerate(cipher_vocab)}
        HASH_CIPHER = cipher_A2I.get("#", -1)
        # Update ct to use raw cipher text
        ct = ct_raw_clean

    # Check for empty cipher text
    if len(ct) == 0:
        return "", list(range(V_cipher)), 0.0, ([], PTNodes()), -1

    logp_after_prompt, h_after_prompt = run_prompt(model, prompt, alphabet)

    # ---- seeding (t=0) ----
    c0 = ct[0]
    c0i = cipher_A2I[c0]
    seed_states: List[BeamState] = []

    def add_seed(p0i: int, g0: float, h0):
        node0 = add_node(-1, p0i)
        if homophonic:
            c2p, p2c = make_empty_homophonic_key(V_cipher, V, HASH_CIPHER, HASH if HASH is not None else -1)
            # Assign first cipher symbol if not #
            if HASH_CIPHER == -1 or c0i != HASH_CIPHER:
                c2p[c0i] = p0i
                p2c[p0i].add(c0i)
            st = BeamState(g=g0, node=node0, c2p=c2p, p2c=p2c, h_prev=h0, last_idx=p0i, length=1)
        else:
            c2p, p2c = make_empty_key_fixed_hash(V, HASH if HASH is not None else -1)
            # set mapping for c0 -> p0, unless it's the fixed '#'
            if HASH is None or c0i != HASH:
                if c2p[c0i] != -1 and c2p[c0i] != p0i:  # conflict
                    return
                if p2c[p0i] != -1 and p2c[p0i] != c0i:  # conflict
                    return
                c2p[c0i] = p0i
                p2c[p0i] = c0i
            st = BeamState(g=g0, node=node0, c2p=c2p, p2c=p2c, h_prev=h0, last_idx=p0i, length=1)
        seed_states.append(st)

    force_first = (pref_len > 0) and (0 < pref_len)
    if force_first:
        # Force to prefix[0]
        forced_pi = A2I[pref[0]]
        if (not homophonic) and HASH is not None and c0i == HASH and forced_pi != HASH:
            pass  # impossible
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
        if (homophonic and HASH_CIPHER >= 0 and c0i == HASH_CIPHER) or ((not homophonic) and HASH_CIPHER >= 0 and c0i == HASH_CIPHER):
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
    seed_states.sort(key=lambda s: rank_score(s.g, s.length, alpha), reverse=True)
    states = seed_states[:beam_size]

    # Show initial character if incremental output requested
    if incremental and states:
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
                print(f"bpc={best_bpc:.3f} (pt bpc={pt_bpc:.3f}) {best_pt_so_far}")
            else:
                print(f"bpc={best_bpc:.3f} {best_pt_so_far}")
        else:
            print(f"bpc={best_bpc:.3f} {best_pt_so_far}")

    # ---- main ----
    for pos in range(1, len(ct)):
        ci = cipher_A2I[ct[pos]]
        row_logps, h_rows = one_step_micro(model, states, micro)
        candidates: List[Tuple[float, BeamState]] = []

        # Are we inside the prefix window at this pos?
        inside_prefix = (pref_len > 0) and (pos < pref_len)
        forced_pi = A2I[pref[pos]] if inside_prefix else None

        for i, st in enumerate(states):
            row = row_logps[i]
            hrow = h_rows[i]

            # If ciphertext is '#', force '#'
            if homophonic:
                if HASH_CIPHER >= 0 and ci == HASH_CIPHER:
                    pi = HASH
                    # mapping consistency
                    if st.c2p[ci] not in (-1, pi) or (ci not in st.p2c[pi]):
                        continue
                    node2 = nodes.add(st.node, pi)
                    g_new = st.g + float(row[pi].item())
                    st2 = BeamState(g=g_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                    h_prev=hrow, last_idx=pi, length=st.length+1)
                    candidates.append((rank_score(st2.g, st2.length, alpha), st2))
                    continue
            else:
                if HASH_CIPHER >= 0 and ci == HASH_CIPHER:
                    pi = HASH
                    # mapping must be consistent
                    if st.c2p[ci] not in (-1, pi) or (st.p2c[pi] not in (-1, ci)):
                        continue
                    node2 = nodes.add(st.node, pi)
                    g_new = st.g + float(row[pi].item())
                    st2 = BeamState(g=g_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                    h_prev=hrow, last_idx=pi, length=st.length+1)
                    candidates.append((rank_score(st2.g, st2.length, alpha), st2))
                    continue

            # If we're inside prefix, force the plaintext symbol
            if inside_prefix:
                pi = forced_pi
                if homophonic:
                    # Enforce bidirectional # constraint
                    if HASH_CIPHER >= 0 and HASH >= 0:
                        if ci == HASH_CIPHER and pi != HASH:
                            continue  # # cipher symbol can only map to # plaintext
                        if pi == HASH and ci != HASH_CIPHER:
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
                        st2 = BeamState(g=g_new, node=node2, c2p=c2p_new, p2c=p2c_new,
                                       h_prev=hrow, last_idx=pi, length=st.length+1)
                        candidates.append((rank_score(st2.g, st2.length, alpha), st2))
                    else:
                        # Already mapped
                        node2 = nodes.add(st.node, pi)
                        g_new = st.g + float(row[pi].item())
                        st2 = BeamState(g=g_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                       h_prev=hrow, last_idx=pi, length=st.length+1)
                        candidates.append((rank_score(st2.g, st2.length, alpha), st2))
                else:
                    # Enforce # constraint for monoalphabetic mode
                    if HASH >= 0 and pi == HASH:
                        if HASH_CIPHER >= 0:
                            # If # exists in cipher, only # cipher can map to # plaintext
                            if ci != HASH_CIPHER:
                                continue
                        else:
                            # If no # in cipher, no cipher symbol can map to # plaintext
                            continue

                    # mapping consistency
                    if st.c2p[ci] not in (-1, pi) or (st.p2c[pi] not in (-1, ci)):
                        continue
                    # apply mapping if new
                    if st.c2p[ci] == -1:
                        c2p = array('b', st.c2p); p2c = array('b', st.p2c)
                        c2p[ci] = pi; p2c[pi] = ci
                    else:
                        c2p = st.c2p; p2c = st.p2c
                    node2 = nodes.add(st.node, pi)
                    g_new = st.g + float(row[pi].item())  # no prior_w here
                    st2 = BeamState(g=g_new, node=node2, c2p=c2p, p2c=p2c,
                                    h_prev=hrow, last_idx=pi, length=st.length+1)
                    candidates.append((rank_score(st2.g, st2.length, alpha), st2))
                continue

            # Normal expansion (not in prefix window)
            if homophonic:
                # Homophonic expansion
                mapped = st.c2p[ci]
                if mapped != -1:
                    # Cipher symbol already mapped
                    pi = int(mapped)
                    node2 = nodes.add(st.node, pi)
                    g_new = st.g + float(row[pi].item())
                    st2 = BeamState(g=g_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                    h_prev=hrow, last_idx=pi, length=st.length+1)
                    candidates.append((rank_score(st2.g, st2.length, alpha), st2))
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
                        if HASH_CIPHER >= 0 and HASH >= 0:
                            if ci == HASH_CIPHER and pi != HASH:
                                continue  # # cipher symbol can only map to # plaintext
                            if pi == HASH and ci != HASH_CIPHER:
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
                        st2 = BeamState(g=g_new, node=node2, c2p=c2p_new, p2c=p2c_new,
                                       h_prev=hrow, last_idx=pi, length=st.length+1)
                        candidates.append((rank_score(st2.g, st2.length, alpha), st2))
            else:
                # Monoalphabetic expansion
                mapped = st.c2p[ci]
                if mapped != -1:
                    pi = int(mapped)
                    node2 = nodes.add(st.node, pi)
                    g_new = st.g + float(row[pi].item())
                    st2 = BeamState(g=g_new, node=node2, c2p=st.c2p, p2c=st.p2c,
                                    h_prev=hrow, last_idx=pi, length=st.length+1)
                    candidates.append((rank_score(st2.g, st2.length, alpha), st2))
                else:
                    scores = row.clone()
                    if prior_w > 0.0 and char_priors is not None:
                        # Use training data character priors instead of English etaoin
                        eta = char_priors.to(scores.device)
                        scores = scores + prior_w * torch.log(eta.clamp_min(1e-30))
                    used = [p for p in range(V) if st.p2c[p] != -1]
                    # Only # cipher symbol can map to # plaintext
                    if HASH >= 0:
                        if HASH_CIPHER >= 0:
                            # If # exists in cipher, only # cipher can map to # plaintext
                            if ci != HASH_CIPHER:
                                used.append(HASH)
                        else:
                            # If no # in cipher, no cipher symbol can map to # plaintext
                            used.append(HASH)
                    scores[used] = float('-inf')
                    k = min(topk_expand, max(0, V - len(used)))
                    if k <= 0: continue
                    _, idxs = torch.topk(scores, k)
                    for pi in idxs.tolist():
                        c2p = array('b', st.c2p); p2c = array('b', st.p2c)
                        c2p[ci] = pi; p2c[pi] = ci
                        node2 = nodes.add(st.node, pi)
                        g_new = st.g + float(row[pi].item())
                        st2 = BeamState(g=g_new, node=node2, c2p=c2p, p2c=p2c,
                                        h_prev=hrow, last_idx=pi, length=st.length+1)
                        candidates.append((rank_score(st2.g, st2.length, alpha), st2))

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
            best_st = max(states, key=lambda s: s.g)
            best_pt_so_far = reconstruct(nodes, best_st.node, alphabet)

            # Calculate BPC for current best hypothesis
            best_bpc = conditional_bpc(model, alphabet, prompt, best_pt_so_far,
                                      p0_log if prompt == "" else None, str(device))

            # Show PT BPC if debugging with ground truth
            if pt_debug_clean:
                target_len = pos + 1
                if target_len <= len(pt_debug_clean):
                    pt_prefix = pt_debug_clean[:target_len]
                    pt_bpc = conditional_bpc(model, alphabet, prompt, pt_prefix,
                                           p0_log if prompt == "" else None, str(device))
                    print(f"bpc={best_bpc:.3f} (pt bpc={pt_bpc:.3f}) {best_pt_so_far}")
                else:
                    print(f"bpc={best_bpc:.3f} {best_pt_so_far}")
            else:
                print(f"bpc={best_bpc:.3f} {best_pt_so_far}")

        if device.type == "cuda" and (pos % 32 == 0):
            torch.cuda.empty_cache()

    if not states:
        if homophonic:
            return "", complete_homophonic_key(array('b', [-1]*V_cipher), V_cipher, V), -1e9, ([], nodes), -1
        else:
            return "", list(range(V)), -1e9, ([], nodes), -1

    best = max(states, key=lambda s: s.g)
    best_pt = reconstruct(nodes, best.node, alphabet)
    if homophonic:
        c2p_full = complete_homophonic_key(best.c2p, V_cipher, V)
    else:
        c2p_full = complete_key(best.c2p, V)
    return best_pt, c2p_full, best.g, (states, nodes), pt_falloff_index

# ---- CLI ----

def read_all(path: str) -> str:
    if path == "-" or path == "/dev/stdin":
        return sys.stdin.read()
    with io.open(path, "r", encoding="utf8", errors="ignore") as f:
        return f.read()

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
    ap.add_argument("--prefix", type=str, default="", help="force initial plaintext characters (include # if needed)")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--refine", type=int, default=0)
    ap.add_argument("--pt", type=str, default="", help="true plaintext for debugging (include # if present)")
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp32")
    ap.add_argument("--prior_text", type=str, default="")
    ap.add_argument("--prior", choices=["data","uniform"], default="data")
    ap.add_argument("--smooth", type=float, default=1e-6)
    ap.add_argument("--homophonic", type=int, default=0, help="max cipher symbols per plaintext (0=monoalphabetic)")
    ap.add_argument("--incremental", action="store_true", help="show incremental triangle output during beam search")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(0)
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


    ct = read_all(args.cipherfile)
    best_pt, c2p_full, g_raw, pack, pt_falloff = beam_decode(
        model, ct, alphabet,
        beam_size=args.beam, micro=args.micro, topk_expand=args.topk,
        prior_w=args.prior_w, alpha=args.alpha, gumbel=args.gumbel,
        prefix=args.prefix, prompt=args.prompt,
        p0_log=p0_log if args.prompt == "" else None,
        char_priors=p0, pt_debug=args.pt, homophonic_limit=args.homophonic,
        incremental=args.incremental
    )
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
