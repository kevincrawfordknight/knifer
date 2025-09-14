#!/usr/bin/env python3
import argparse, random, string, math, torch
from score_lstm import AWDCharLSTM, load_model, clean_text, encode, bpc_for_text

ALPH = "abcdefghijklmnopqrstuvwxyz"

def random_key():
    p = list(ALPH)
    random.shuffle(p)
    return "".join(p)  # maps cipher 'a..z' -> plain p[0..25]

def key_apply(key, ct):
    # ct is nospace a..z; decode by substituting per-letter
    table = {ALPH[i]: key[i] for i in range(26)}
    return "".join(table.get(c, "") for c in ct)

def key_swap(key, i, j):
    k = list(key)
    k[i], k[j] = k[j], k[i]
    return "".join(k)

def hill_climb(model, ct, iters=20000, temp=1.0, restarts=10, seed=0, device='cuda'):
    rng = random.Random(seed)
    best_global = None
    best_score = float("inf")
    for r in range(restarts):
        key = random_key()
        pt = key_apply(key, ct)
        score = bpc_for_text(model, pt, device=device, block=1024)  # lower is better
        curr_key, curr_score = key, score
        if curr_score < best_score:
            best_global, best_score = curr_key, curr_score
        for t in range(1, iters+1):
            i, j = rng.randrange(26), rng.randrange(26)
            if i == j: continue
            cand_key = key_swap(curr_key, i, j)
            cand_pt = key_apply(cand_key, ct)
            cand_score = bpc_for_text(model, cand_pt, device=device, block=1024)
            # accept if better, or with simulated annealing probability
            if cand_score < curr_score or rng.random() < math.exp((curr_score - cand_score)/max(1e-6, temp)):
                curr_key, curr_score = cand_key, cand_score
                if curr_score < best_score:
                    best_global, best_score = curr_key, curr_score
            # mild cooling
            if t % 1000 == 0: temp *= 0.95
        # new restart
    return best_global, best_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("cipherfile")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = load_model(args.model, device=device)
    ct = open(args.cipherfile, "r", encoding="utf8").read().lower()
    ct = "".join(c for c in ct if c in ALPH)  # nospace letters only

    key, score = hill_climb(model, ct, iters=args.iters, restarts=args.restarts, temp=args.temp, seed=args.seed, device=device)
    pt = key_apply(key, ct)
    print("BEST_BPC:", score)
    print("KEY (cipher a..z -> plain):", key)
    print("PLAINTEXT:\n", pt[:5000])

if __name__ == "__main__":
    main()
