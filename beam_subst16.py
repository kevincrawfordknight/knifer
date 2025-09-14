#!/usr/bin/env python3
"""
beam_subst16.py — Language-agnostic beam search for monoalphabetic substitution.
Removes English bias, uses data-driven character priors from training.
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

    # Extract state dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
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

    # Extract alphabet from checkpoint - no hardcoded assumptions
    if isinstance(ckpt, dict) and 'vocab' in ckpt:
        alphabet = ckpt['vocab']
    else:
        raise ValueError("No alphabet found in checkpoint. Please use lstm6.py which saves alphabet info.")

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

# ---- beam structures ----

class PTNodes:
    def __init__(self):
        self.parent = array('i')   # node_id -> parent_node_id
        self.ch = array('i')       # node_id -> char_index
        self.depth = array('i')    # node_id -> depth_in_tree

    def new_node(self, parent_id: int, char_idx: int, depth: int) -> int:
        node_id = len(self.parent)
        self.parent.append(parent_id)
        self.ch.append(char_idx)
        self.depth.append(depth)
        return node_id

    def root(self) -> int:
        return self.new_node(-1, -1, 0)

class BeamState:
    def __init__(self, node: int, c2p: array, h_state, logp: float, gumbel: float = 0.0):
        self.node = node
        self.c2p = c2p
        self.h_state = h_state
        self.logp = logp
        self.gumbel = gumbel

    @property
    def priority(self):
        return self.logp + self.gumbel

def reconstruct(nodes: PTNodes, node_id: int, alphabet: str) -> str:
    out = []
    while node_id >= 0 and nodes.parent[node_id] >= 0:
        out.append(alphabet[nodes.ch[node_id]])
        node_id = nodes.parent[node_id]
    return "".join(reversed(out))

def run_prompt(model: AWDCharLSTM, prompt: str, alphabet: str):
    """Run prompt through model to warm up hidden state."""
    device = next(model.parameters()).device
    idx = {c: i for i,c in enumerate(alphabet)}
    s = clean_text(prompt, alphabet)
    if not s:
        h = None
        logp = 0.0
    else:
        ids = torch.tensor([idx[c] for c in s], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, h = model(ids)
        logp = F.log_softmax(logits[0], dim=-1).sum().item()
    return logp, h

def read_all(path: str) -> str:
    with io.open(path, "r", encoding="utf8", errors="ignore") as f:
        return f.read()

def conditional_bpc(model: AWDCharLSTM, alphabet: str, prompt: str, text: str, p0_log: Optional[torch.Tensor], device: str):
    s_prompt = clean_text(prompt, alphabet)
    s_text   = clean_text(text, alphabet)

    idx = {c: i for i,c in enumerate(alphabet)}

    with torch.no_grad():
        h = None
        if s_prompt:
            ids = torch.tensor([idx[c] for c in s_prompt], dtype=torch.long).unsqueeze(0).to(device)
            _, h = model(ids, h)

        if not s_text:
            return 0.0

        total_logp = 0.0
        for i, ch in enumerate(s_text):
            ch_idx = idx[ch]
            if i == 0 and p0_log is not None:
                logp_ch = p0_log[ch_idx].item()
            else:
                if h is None:
                    # First char, no prompt
                    dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
                    logits, h = model(dummy_input)
                    logp_ch = F.log_softmax(logits[0, 0], dim=-1)[ch_idx].item()
                else:
                    # Use previous hidden state
                    dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
                    logits, h = model(dummy_input, h)
                    logp_ch = F.log_softmax(logits[0, 0], dim=-1)[ch_idx].item()

            total_logp += logp_ch

            # Update hidden state with actual character
            ch_input = torch.tensor([[ch_idx]], dtype=torch.long).to(device)
            _, h = model(ch_input, h)

    return -total_logp / (len(s_text) * math.log(2))

def beam_subst_decode(model: AWDCharLSTM,
                     ct_raw: str,
                     alphabet: str,
                     beam_size: int = 1000,
                     nbest: int = 10,
                     topk: int = 26,
                     gumbel: float = 0.0,
                     alpha: float = 0.0,
                     prior_w: float = 0.0,
                     prefix: str = "",
                     prompt: str = "",
                     p0_prior: Optional[torch.Tensor] = None,
                     refine: int = 0,
                     pt_debug: str = "",
                     dtype=None) -> Tuple[List[BeamState], PTNodes]:

    device = str(next(model.parameters()).device)
    if dtype: model = model.to(dtype)

    V = len(alphabet)
    A2I = {c:i for i,c in enumerate(alphabet)}
    HASH = A2I.get("#", None)

    ct = clean_text(ct_raw, alphabet)
    if not ct:
        return [], PTNodes()

    T = len(ct)
    nodes = PTNodes()
    root_id = nodes.root()

    # Handle prefix constraint
    pref = clean_text(prefix, alphabet)
    P = len(pref)

    # Run prompt to get initial hidden state
    logp_after_prompt, h_after_prompt = run_prompt(model, prompt, alphabet)

    # Convert p0_prior to log space if provided
    p0_log = None
    if p0_prior is not None:
        p0_log = torch.log(p0_prior.clamp(min=1e-10)).to(device)

    # Initialize beam with empty partial key
    c2p_init = array('i', [-1] * V)  # ciphertext_char -> plaintext_char
    if P > 0:
        # Apply prefix constraints
        for i, pc in enumerate(pref):
            cc = ct[i]
            if A2I[cc] in range(V) and A2I[pc] in range(V):
                c2p_init[A2I[cc]] = A2I[pc]

    init_state = BeamState(
        node=root_id,
        c2p=c2p_init,
        h_state=h_after_prompt,
        logp=logp_after_prompt,
        gumbel=random.gauss(0, gumbel) if gumbel > 0 else 0.0
    )

    beam = [init_state]

    for t in range(P, T):
        cc = ct[t]
        cc_idx = A2I.get(cc, -1)
        if cc_idx == -1:
            continue  # skip unknown characters

        next_beam = []

        for st in beam:
            if st.c2p[cc_idx] != -1:
                # Already mapped, extend with known mapping
                pt_char = st.c2p[cc_idx]

                # Get LM score
                if st.h_state is None:
                    dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
                    logits, new_h = model(dummy_input)
                else:
                    dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
                    logits, new_h = model(dummy_input, st.h_state)

                lm_logp = F.log_softmax(logits[0, 0], dim=-1)[pt_char].item()

                # Update hidden state with actual character
                ch_input = torch.tensor([[pt_char]], dtype=torch.long).to(device)
                _, final_h = model(ch_input, st.h_state)

                total_logp = st.logp + lm_logp

                new_node = nodes.new_node(st.node, pt_char, nodes.depth[st.node] + 1)
                new_state = BeamState(
                    node=new_node,
                    c2p=st.c2p,
                    h_state=final_h,
                    logp=total_logp,
                    gumbel=st.gumbel
                )
                next_beam.append(new_state)
            else:
                # Try all possible mappings
                candidates = []
                used_chars = set(st.c2p[i] for i in range(V) if st.c2p[i] != -1)

                for pt_idx in range(V):
                    if pt_idx in used_chars:
                        continue  # already used
                    if HASH is not None and pt_idx == HASH:
                        continue  # skip hash symbol in plaintext

                    # Get LM score
                    if st.h_state is None:
                        dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
                        logits, new_h = model(dummy_input)
                    else:
                        dummy_input = torch.zeros(1, 1, dtype=torch.long).to(device)
                        logits, new_h = model(dummy_input, st.h_state)

                    lm_logp = F.log_softmax(logits[0, 0], dim=-1)[pt_idx].item()

                    # Add first-character prior if applicable
                    if t == 0 and p0_log is not None:
                        prior_logp = p0_log[pt_idx].item()
                        total_logp = st.logp + prior_logp
                    else:
                        total_logp = st.logp + lm_logp

                    candidates.append((total_logp, pt_idx, new_h))

                # Keep top-k candidates
                candidates.sort(reverse=True, key=lambda x: x[0])
                for score, pt_idx, new_h in candidates[:topk]:
                    # Update hidden state with actual character
                    ch_input = torch.tensor([[pt_idx]], dtype=torch.long).to(device)
                    _, final_h = model(ch_input, st.h_state)

                    new_c2p = array('i', st.c2p)
                    new_c2p[cc_idx] = pt_idx

                    new_node = nodes.new_node(st.node, pt_idx, nodes.depth[st.node] + 1)
                    new_state = BeamState(
                        node=new_node,
                        c2p=new_c2p,
                        h_state=final_h,
                        logp=score,
                        gumbel=st.gumbel + (random.gauss(0, gumbel) if gumbel > 0 else 0.0)
                    )
                    next_beam.append(new_state)

        # Keep top beam_size states
        next_beam.sort(reverse=True, key=lambda x: x.priority)
        beam = next_beam[:beam_size]

        if not beam:
            break

    # Return top nbest results
    beam.sort(reverse=True, key=lambda x: x.priority)
    return beam[:nbest], nodes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Path to LSTM model (.pt file)")
    ap.add_argument("ciphertext", help="Path to ciphertext file")

    ap.add_argument("--beam", type=int, default=1000, help="Beam size")
    ap.add_argument("--nbest", type=int, default=10, help="Number of best results to return")
    ap.add_argument("--topk", type=int, default=26, help="Top-k candidates per position")

    ap.add_argument("--gumbel", type=float, default=0.0, help="Gumbel noise for exploration")
    ap.add_argument("--alpha", type=float, default=0.0, help="Length normalization")
    ap.add_argument("--prior_w", type=float, default=0.0, help="Prior weight")

    ap.add_argument("--prefix", type=str, default="", help="Known plaintext prefix")
    ap.add_argument("--prompt", type=str, default="", help="LM warm-up prompt")

    ap.add_argument("--prior_text", type=str, default="", help="Text file to compute character priors from")
    ap.add_argument("--prior", choices=["data", "uniform"], default="data", help="Prior type: data-driven or uniform")
    ap.add_argument("--smooth", type=float, default=1e-6, help="Smoothing for text-based priors")

    ap.add_argument("--refine", type=int, default=0, help="Refinement iterations")
    ap.add_argument("--pt", type=str, default="", help="True plaintext for debugging")

    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if args.dtype == "fp16": dtype = torch.float16
    elif args.dtype == "bf16": dtype = torch.bfloat16
    else: dtype = None

    print(f"Loading model from {args.model}")
    model, alphabet, saved_priors = load_model(args.model, device=device)
    print(f"Alphabet: {alphabet} ({len(alphabet)} chars)")

    # Load ciphertext
    with io.open(args.ciphertext, "r", encoding="utf8", errors="ignore") as f:
        ct_text = f.read()

    # Determine character priors
    if args.prior_text and os.path.exists(args.prior_text):
        p0 = prior_from_text(read_all(args.prior_text), alphabet, smoothing=args.smooth)
        print("Using character priors from", args.prior_text)
    elif saved_priors is not None:
        p0 = saved_priors
        print("Using character priors from model checkpoint")
    elif args.prior == "uniform":
        p0 = prior_uniform(alphabet)
        print("Using uniform character priors")
    else:
        # Default: use saved priors if available, otherwise uniform
        if saved_priors is not None:
            p0 = saved_priors
            print("Using character priors from model checkpoint")
        else:
            p0 = prior_uniform(alphabet)
            print("Using uniform character priors (no training data available)")

    p0 = p0.to(device)

    results, nodes = beam_subst_decode(
        model, ct_text, alphabet,
        beam_size=args.beam, nbest=args.nbest, topk=args.topk,
        gumbel=args.gumbel, alpha=args.alpha, prior_w=args.prior_w,
        prefix=args.prefix, prompt=args.prompt, p0_prior=p0,
        refine=args.refine, pt_debug=args.pt, dtype=dtype
    )

    print(f"\nTop {len(results)} results:")
    for i, st in enumerate(results):
        pt = reconstruct(nodes, st.node, alphabet)
        bpc = conditional_bpc(model, alphabet, args.prompt, pt,
                             torch.log(p0.clamp(min=1e-10)) if args.prior != "uniform" else None, device)
        print(f"{i+1:2d}. logp={st.logp:8.2f} bpc={bpc:6.3f} |{pt}|")

    if results:
        best = results[0]
        c2p_full = [best.c2p[i] if best.c2p[i] != -1 else i for i in range(len(alphabet))]
        print("KEY c->p:\n", "".join(alphabet[p] for p in c2p_full))

if __name__ == "__main__":
    import os
    main()