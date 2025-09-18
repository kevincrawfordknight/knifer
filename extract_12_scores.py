#!/usr/bin/env python3
"""
Extract the 12 incremental scores for #thegoodnews from both methods
"""
import torch
import torch.nn.functional as F
from beam_subst17 import load_model, clean_text, conditional_bpc

def extract_12_scores():
    """Extract incremental scores for #thegoodnews (12 chars)"""
    model_path = "charlarge_pile80_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, char_priors = load_model(model_path, device=str(device))

    if char_priors is not None:
        p0_log = torch.log(char_priors.clamp_min(1e-30)).to(device)
    else:
        p0_log = None

    text = "#thegoodnews"
    s_text = clean_text(text, alphabet)
    idx = {c: i for i, c in enumerate(alphabet)}

    print(f"Extracting 12 incremental scores for: '{text}'")
    print()

    # === BEAM SEARCH SCORES (from triangle output) ===
    print("=== BEAM SEARCH SCORES (from actual run) ===")

    # From the beam search output, when triangle shows true plaintext:
    # The key observation is that the triangle shows both beam BPC and fresh BPC
    # When they match, it means beam search selected the correct hypothesis
    beam_triangle_scores = [
        # These are the BPC values when triangle output shows #thegoodnews...
        # Need to extract from actual run, but based on pattern seen:
        5.03,    # #
        # Will extract remaining 11 from the beam output when it appears
    ]

    print("Need to extract from actual beam run when #thegoodnews... appears in triangle")
    print()

    # === CONDITIONAL_BPC SCORES ===
    print("=== CONDITIONAL_BPC INCREMENTAL SCORES ===")

    # Calculate incremental cumulative scores
    cumulative_scores = []

    for length in range(1, len(s_text) + 1):
        prefix = s_text[:length]
        bpc = conditional_bpc(model, alphabet, "", prefix, p0_log, str(device))
        cumulative_log_prob = -bpc * length * torch.log(torch.tensor(2.0)).item()
        cumulative_scores.append(cumulative_log_prob)
        print(f"Length {length:2d} '{prefix}': BPC={bpc:.6f}, cumulative_score={cumulative_log_prob:.6f}")

    # Calculate incremental scores (difference between consecutive cumulative scores)
    print(f"\n=== INCREMENTAL SCORES (conditional_bpc) ===")
    incremental_scores = []

    for i in range(len(cumulative_scores)):
        if i == 0:
            incremental = cumulative_scores[0]
        else:
            incremental = cumulative_scores[i] - cumulative_scores[i-1]
        incremental_scores.append(incremental)
        print(f"Char {i} '{s_text[i]}': incremental_score={incremental:.6f}")

    print(f"\n=== SUMMARY FOR COMPARISON ===")
    print("Need beam search incremental scores to compare with these conditional_bpc scores:")
    for i, score in enumerate(incremental_scores):
        print(f"Position {i} '{s_text[i]}': conditional_bpc_incremental={score:.6f}")

if __name__ == "__main__":
    extract_12_scores()