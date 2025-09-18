#!/usr/bin/env python3
"""
Debug exactly what beam search does step by step for #i
"""
import torch
import torch.nn.functional as F
from beam_subst17 import load_model, clean_text, run_prompt

def debug_exact_beam():
    """Simulate exactly what beam search does for the first two characters"""
    model_path = "charlarge_pile80_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, char_priors = load_model(model_path, device=str(device))

    prompt = "#isaythat"
    text = "#i"

    s_prompt = clean_text(prompt, alphabet)
    s_text = clean_text(text, alphabet)
    idx = {c: i for i, c in enumerate(alphabet)}

    print(f"Simulating beam search for: '{text}' with prompt: '{prompt}'")

    # === STEP 1: Get prompt result (exactly like beam search) ===
    logp_after_prompt, h_after_prompt = run_prompt(model, prompt, alphabet)

    # === STEP 2: Score first character (like beam seeding) ===
    pi0 = idx[s_text[0]]  # '#'
    g0 = float(logp_after_prompt[pi0].item())
    print(f"First char '{s_text[0]}' score: {g0:.6f}")

    # === STEP 3: What beam search ACTUALLY does (WRONG) ===
    print("\n=== BEAM SEARCH METHOD (scoring 'i' after prompt, not after '#') ===")

    # Use hidden state from after prompt to score second char (WRONG!)
    last_idx = torch.tensor([pi0], dtype=torch.long, device=device)  # [1] for batch_size=1
    if h_after_prompt is None:
        h_prev_batch = None
    else:
        hs = h_after_prompt[0].detach().unsqueeze(1)  # Add batch dim
        cs = h_after_prompt[1].detach().unsqueeze(1)  # Add batch dim
        h_prev_batch = (hs, cs)

    # Model call - this scores 'i' given hidden state after 't', not after '#'
    logits, h_after = model(last_idx.view(-1,1), h_prev_batch)
    logp = F.log_softmax(logits[:, -1, :], dim=-1)

    pi1 = idx[s_text[1]]  # 'i'
    g1_wrong = float(logp[0, pi1].item())
    print(f"Second char '{s_text[1]}' score (WRONG method): {g1_wrong:.6f}")

    total_beam_wrong = g0 + g1_wrong
    print(f"Total beam score (WRONG): {total_beam_wrong:.6f}")

    # === STEP 4: What conditional_bpc does (CORRECT) ===
    print("\n=== CONDITIONAL_BPC METHOD (scoring 'i' after feeding '#') ===")

    # First feed the first character to get updated hidden state
    x0 = torch.tensor([[pi0]], dtype=torch.long, device=device)
    # Need to add batch dimension to hidden state for model call
    h_batch = (h_after_prompt[0].unsqueeze(1), h_after_prompt[1].unsqueeze(1))
    _, h_after_first = model(x0, h_batch)

    # Then score second character using updated hidden state
    x_prev = torch.tensor([[pi0]], dtype=torch.long, device=device)  # '#'
    logits, h_after_second = model(x_prev, h_after_first)
    logp_correct = F.log_softmax(logits[0, -1, :], dim=-1)

    g1_correct = float(logp_correct[pi1].item())
    print(f"Second char '{s_text[1]}' score (CORRECT method): {g1_correct:.6f}")

    total_correct = g0 + g1_correct
    print(f"Total score (CORRECT): {total_correct:.6f}")

    print(f"\n=== COMPARISON ===")
    print(f"Beam search total: {total_beam_wrong:.6f}")
    print(f"Conditional_bpc total: {total_correct:.6f}")
    print(f"Expected from simple debug: -8.161994")
    print(f"Beam vs correct difference: {total_beam_wrong - total_correct:.6f}")
    print(f"This explains the scoring corruption!")

if __name__ == "__main__":
    debug_exact_beam()