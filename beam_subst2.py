#!/usr/bin/env python3
import argparse, math, heapq, string, torch
import torch.nn.functional as F

from score_lstm import AWDCharLSTM, load_model, clean_text

ALPH = "abcdefghijklmnopqrstuvwxyz"
AI = {c:i for i,c in enumerate(ALPH)}

class BeamState:
    __slots__ = ("logp","pt","map_c2p","map_p2c","h")
    def __init__(self, logp, pt, map_c2p, map_p2c, h):
        self.logp = logp         # cumulative log-prob in nats
        self.pt = pt             # plaintext so far (string)
        self.map_c2p = map_c2p   # cipher->plain dict
        self.map_p2c = map_p2c   # plain->cipher dict
        self.h = h               # LSTM hidden state tuple

    def __lt__(self, other):  # for heapq (max-heap via inverted compare)
        return self.logp > other.logp

def one_step_logits(model, token_idx, h):
    """Feed one plaintext token (int) and return (logits, new_hidden)."""
    x = torch.tensor([[token_idx]], dtype=torch.long, device=next(model.parameters()).device)
    logits, h2 = model(x, h)
    return logits[0, -1, :], h2  # [V], new hidden

def beam_decode(model, ct, beam_size=400, topk_expand=26, device='cuda'):
    """
    LM-guided beam over partial keys for monoalphabetic substitution.
    - If next cipher char is already mapped, force that plaintext letter.
    - If unseen, branch over currently-unused plaintext letters; prune to top-k by LM.
    Scoring: we *don’t* score the very first char (no BOS in training); we feed it to set hidden,
    then score from the second output onward.
    """
    model.eval()
    ct = clean_text(ct)

    # Start with empty hypothesis
    init = BeamState(logp=0.0, pt="", map_c2p={}, map_p2c={}, h=None)
    beam = [init]

    for pos, c in enumerate(ct):
        new_beam = []
        for st in beam:
            mapped = st.map_c2p.get(c, None)
            if st.pt == "":
                # First character: assign (or force) a plaintext letter, but DO NOT add logprob.
                if mapped is not None:
                    p = mapped
                    p_idx = AI[p]
                    # update hidden by feeding p
                    _, h_next = one_step_logits(model, p_idx, st.h)  # logits ignored here
                    new_beam.append(BeamState(st.logp, st.pt + p, st.map_c2p, st.map_p2c, h_next))
                else:
                    # Branch over unused plaintext letters (all letters at start), optionally prune by a flat prior.
                    # We can’t use LM to score first char (no BOS); just expand all or a subset (topk = 26 means all).
                    cand = ALPH[:topk_expand]
                    for p in cand:
                        map_c2p = dict(st.map_c2p); map_p2c = dict(st.map_p2c)
                        map_c2p[c] = p; map_p2c[p] = c
                        p_idx = AI[p]
                        _, h_next = one_step_logits(model, p_idx, st.h)  # set state
                        new_beam.append(BeamState(st.logp, st.pt + p, map_c2p, map_p2c, h_next))
                continue  # done with first-char handling

            # For positions >= 2, rely on LM next-token distribution given previous plaintext char
            last_idx = AI[st.pt[-1]]
            prev_logits, h_after_prev = one_step_logits(model, last_idx, st.h)
            logp_next = F.log_softmax(prev_logits, dim=-1)  # [V], nats

            if mapped is not None:
                p_idx = AI[mapped]
                new_logp = st.logp + float(logp_next[p_idx].item())
                # IMPORTANT: feed the *chosen* token to advance hidden
                _, h_next = one_step_logits(model, p_idx, h_after_prev)
                new_beam.append(BeamState(new_logp, st.pt + mapped, st.map_c2p, st.map_p2c, h_next))
            else:
                # Branch over unused plaintext letters; prune by LM top-k among the unused set
                unused = [p for p in ALPH if p not in st.map_p2c]
                # pick top-k by LM score restricted to unused
                cand = sorted(unused, key=lambda ch: float(logp_next[AI[ch]].item()), reverse=True)[:min(topk_expand, len(unused))]
                for p in cand:
                    map_c2p = dict(st.map_c2p); map_p2c = dict(st.map_p2c)
                    map_c2p[c] = p; map_p2c[p] = c
                    p_idx = AI[p]
                    new_logp = st.logp + float(logp_next[p_idx].item())
                    # advance hidden with the chosen token
                    _, h_next = one_step_logits(model, p_idx, h_after_prev)
                    new_beam.append(BeamState(new_logp, st.pt + p, map_c2p, map_p2c, h_next))

        if not new_beam:
            break
        # prune beam
        new_beam.sort(key=lambda s: s.logp, reverse=True)
        beam = new_beam[:beam_size]

    if not beam:
        return "", {}
    best = max(beam, key=lambda s: s.logp)
    return best.pt, best.map_c2p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--beam", type=int, default=400)
    ap.add_argument("--topk", type=int, default=12, help="unused plaintext letters to consider when a new cipher letter appears")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)

    with open(args.cipherfile, "r", encoding="utf8") as f:
        ct = f.read()

    pt, key = beam_decode(model, ct, beam_size=args.beam, topk_expand=args.topk, device=device)
    print("PLAINTEXT:\n", pt)
    print("KEY c->p:\n", "".join(key.get(c, "?") for c in ALPH))

if __name__ == "__main__":
    main()
