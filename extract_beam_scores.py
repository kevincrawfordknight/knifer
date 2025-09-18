#!/usr/bin/env python3
"""
Extract actual incremental beam scores from beam_subst17 for #thegoodnews
"""
import torch
import torch.nn.functional as F
from beam_subst17 import load_model, clean_text, conditional_bpc

def extract_actual_scores():
    """Extract actual beam scores and compare with conditional_bpc"""
    model_path = "charlarge_pile80_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, char_priors = load_model(model_path, device=str(device))

    if char_priors is not None:
        p0_log = torch.log(char_priors.clamp_min(1e-30)).to(device)
    else:
        p0_log = None

    text = "#thegoodnews"
    s_text = clean_text(text, alphabet)

    print(f"Extracting scores for: '{text}' (no prompt)")
    print(f"Cleaned text: '{s_text}'")
    print()

    # === METHOD 1: Run beam search and trace the true plaintext path ===
    print("=== RUNNING BEAM SEARCH TO EXTRACT ACTUAL SCORES ===")
    print("Need to run: python beam_subst17.py charlarge_pile80_last.pt test-data/test.ct.goodnews.txt --pt '#thegoodnews' --beam 500")
    print("And extract scores from debug output...")
    print()

    # === METHOD 2: Calculate conditional_bpc incremental scores ===
    print("=== CONDITIONAL_BPC INCREMENTAL SCORES ===")

    idx = {c: i for i, c in enumerate(alphabet)}

    # Calculate total first
    total_bpc = conditional_bpc(model, alphabet, "", text, p0_log, str(device))
    print(f"Total conditional_bpc: {total_bpc:.6f}")
    print()

    # Calculate incrementally
    h = None
    total_nll = 0.0
    cumulative_scores = []

    # First character from P0
    if p0_log is not None:
        first_nll = float(-p0_log[idx[s_text[0]]].item())
        first_logp = -first_nll
    else:
        raise ValueError("Need P0 for first character")

    total_nll += first_nll
    cumulative_scores.append(-total_nll)  # Cumulative log prob (negative NLL)
    print(f"After char 0 '{s_text[0]}': score={first_logp:.6f}, cumulative={cumulative_scores[-1]:.6f}")

    # Feed first character
    x0 = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
    _, h = model(x0, h)

    # Subsequent characters
    for i in range(1, len(s_text)):
        x = torch.tensor([[idx[s_text[i-1]]]], dtype=torch.long, device=device)
        logits, h = model(x, h)
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        char_nll = float(-logp[idx[s_text[i]]].item())
        char_logp = -char_nll

        total_nll += char_nll
        cumulative_scores.append(-total_nll)  # Cumulative log prob
        print(f"After char {i} '{s_text[i]}': score={char_logp:.6f}, cumulative={cumulative_scores[-1]:.6f}")

    print()
    print("=== CONDITIONAL_BPC CUMULATIVE SCORES ===")
    for i, score in enumerate(cumulative_scores):
        prefix = s_text[:i+1]
        bpc = conditional_bpc(model, alphabet, "", prefix, p0_log, str(device))
        expected_score = -bpc * (i+1) * torch.log(torch.tensor(2.0)).item()
        print(f"Length {i+1} '{prefix}': cumulative={score:.6f}, expected={expected_score:.6f}, BPC={bpc:.6f}")

if __name__ == "__main__":
    extract_actual_scores()