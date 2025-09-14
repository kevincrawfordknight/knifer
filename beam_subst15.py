#!/usr/bin/env python3
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

def _alphabet_for_vocab(vsz: int) -> str:
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
    ckpt = safe_load(ckpt_path, device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
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
    vocab = _alphabet_for_vocab(vocab_size)
    model = AWDCharLSTM(vocab_size=vocab_size, emb=emb, hidden=hidden, layers=layers,
                        p_in=0.0, p_h=0.0, p_out=0.0, tie_weights=False).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, vocab

# ---- prior P0 helpers ----

_ENG_ORDER = "etaoinshrdlcumwfgypbvkjxqz"

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

def prior_etaoin(alphabet: str) -> torch.Tensor:
    scores = {}
    if alphabet[0] == "#":
        scores["#"] = 0.5
        letters = alphabet[1:]
    else:
        letters = alphabet
    for rank, ch in enumerate(_ENG_ORDER[::-1], start=1):
        if ch in letters: scores[ch] = rank
    for ch in letters:
        scores.setdefault(ch, 1.0)
    total = sum(scores[c] for c in scores)
    return torch.tensor([scores[c]/total for c in alphabet], dtype=torch.float32)

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

# ---- beam state ----

class BeamState:
    __slots__ = ("g","node","c2p","p2c","h_prev","last_idx","length")
    def __init__(self, g: float, node: int, c2p: array, p2c: array,
                 h_prev, last_idx: int, length: int):
        self.g = g
        self.node = node
        self.c2p = c2p
        self.p2c = p2c
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

def complete_key(c2p_arr: array, V: int) -> List[int]:
    used = {p for p in c2p_arr if p != -1}
    leftover = [i for i in range(V) if i not in used]
    out = list(c2p_arr)
    it = iter(leftover)
    for ci in range(V):
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
    x0 = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
    _, h = model(x0, h)
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
                p0_log: Optional[torch.Tensor]):

    device = next(model.parameters()).device
    V = len(alphabet)
    A2I = {c:i for i,c in enumerate(alphabet)}
    HASH = A2I["#"] if alphabet[0] == "#" else None

    ct = clean_text(ct_raw, alphabet)
    if len(ct) == 0:
        return "", list(range(V)), 0.0, ([], PTNodes())

    # how many leading '#' to skip before applying prefix
    lead_hash = 0
    if HASH is not None:
        while lead_hash < len(ct) and ct[lead_hash] == "#":
            lead_hash += 1

    pref = clean_text(prefix, alphabet)
    pref_len = len(pref)

    nodes = PTNodes()
    def add_node(parent, ch_idx): return nodes.add(parent, ch_idx)

    logp_after_prompt, h_after_prompt = run_prompt(model, prompt, alphabet)

    # ---- seeding (t=0) ----
    c0i = A2I[ct[0]]
    seed_states: List[BeamState] = []

    def add_seed(p0i: int, g0: float, h0):
        node0 = add_node(-1, p0i)
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

    force_first = (pref_len > 0) and (0 >= lead_hash) and (0 - lead_hash < pref_len)
    if force_first:
        # Force to prefix[0 - lead_hash] == prefix[0]
        forced_pi = A2I[pref[0]]
        if HASH is not None and c0i == HASH and forced_pi != HASH:
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
        if HASH is not None and c0i == HASH:
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
        return "", list(range(V)), -1e9, ([], nodes)
    seed_states.sort(key=lambda s: rank_score(s.g, s.length, alpha), reverse=True)
    states = seed_states[:beam_size]

    # ---- main ----
    for pos in range(1, len(ct)):
        ci = A2I[ct[pos]]
        row_logps, h_rows = one_step_micro(model, states, micro)
        candidates: List[Tuple[float, BeamState]] = []

        # Are we inside the prefix window at this pos?
        inside_prefix = (pref_len > 0) and (pos >= lead_hash) and ((pos - lead_hash) < pref_len)
        forced_pi = A2I[pref[pos - lead_hash]] if inside_prefix else None

        for i, st in enumerate(states):
            row = row_logps[i]
            hrow = h_rows[i]

            # If ciphertext is '#', force '#'
            if HASH is not None and ci == HASH:
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
                if prior_w > 0.0:
                    eta = prior_etaoin(alphabet).to(scores.device)
                    scores = scores + prior_w * torch.log(eta.clamp_min(1e-30))
                used = [p for p in range(V) if st.p2c[p] != -1]
                if HASH is not None: used.append(HASH)
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
        states = [st for _, st in candidates[:beam_size]]
        if device.type == "cuda" and (pos % 32 == 0):
            torch.cuda.empty_cache()

    if not states:
        return "", list(range(V)), -1e9, ([], nodes)

    best = max(states, key=lambda s: s.g)
    best_pt = reconstruct(nodes, best.node, alphabet)
    c2p_full = complete_key(best.c2p, V)
    return best_pt, c2p_full, best.g, (states, nodes)

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
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--prior_w", type=float, default=0.10, help="heuristic letter prior for NEW assignments at t>0")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--gumbel", type=float, default=0.0)
    ap.add_argument("--prefix", type=str, default="")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--nbest", type=int, default=10)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp32")
    ap.add_argument("--prior_text", type=str, default="")
    ap.add_argument("--prior", choices=["etaoin","uniform"], default="etaoin")
    ap.add_argument("--smooth", type=float, default=1e-6)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
    torch.set_grad_enabled(False)

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model, alphabet = load_model(args.model, device=str(device))
    if use_cuda and args.dtype != "fp32":
        if args.dtype == "fp16": model = model.to(dtype=torch.float16)
        elif args.dtype == "bf16": model = model.to(dtype=torch.bfloat16)

    # Prior P0 (used only if no prompt at boundary)
    if args.prior_text:
        p0 = prior_from_text(read_all(args.prior_text), alphabet, smoothing=args.smooth)
    else:
        p0 = prior_etaoin(alphabet) if args.prior == "etaoin" else prior_uniform(alphabet)
    p0_log = torch.log(p0.clamp_min(1e-30)).to(device)

    ct = read_all(args.cipherfile)
    best_pt, c2p_full, g_raw, pack = beam_decode(
        model, ct, alphabet,
        beam_size=args.beam, micro=args.micro, topk_expand=args.topk,
        prior_w=args.prior_w, alpha=args.alpha, gumbel=args.gumbel,
        prefix=args.prefix, prompt=args.prompt,
        p0_log=p0_log if args.prompt == "" else None
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
    for bpc, pt in scored:
        print(f"{bpc:.3f} {pt}")

    # Output best
    best_pt = scored[0][1]
    bpc_best = scored[0][0]
    print("PLAINTEXT:\n", best_pt)
    print("BPC:", bpc_best)
    print("KEY c->p:\n", "".join(alphabet[p] for p in c2p_full))

if __name__ == "__main__":
    main()
