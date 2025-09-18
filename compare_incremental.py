#!/usr/bin/env python3
"""
Compare 12 incremental beam scores vs conditional_bpc scores for #thegoodnews
"""
import torch
import torch.nn.functional as F
from beam_subst17 import load_model, clean_text, conditional_bpc

def compare_incremental():
    """Compare beam vs conditional_bpc incremental scoring for #thegoodnews"""
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

    print(f"Comparing incremental scores for: '{text}' (no prompt)")
    print(f"Cleaned text: '{s_text}'")
    print(f"Length: {len(s_text)} characters")
    print()

    # === METHOD 1: Simulate beam search incremental scoring ===
    print("=== BEAM SEARCH METHOD (wrong hidden state management) ===")
    beam_scores = []
    beam_total = 0.0

    # Start with prior for first character
    if p0_log is not None:
        first_score = float(p0_log[idx[s_text[0]]].item())
    else:
        # If no priors, need to get from model with empty state
        h = None
        x = torch.tensor([[idx['#']]], dtype=torch.long, device=device)  # dummy
        logits, h = model(x, h)
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        first_score = float(logp[idx[s_text[0]]].item())

    beam_scores.append(first_score)
    beam_total += first_score
    print(f"Char 0 '{s_text[0]}': {first_score:.6f} (cumulative: {beam_total:.6f})")

    # For subsequent characters, simulate beam search behavior
    # This is tricky - let me check exactly how beam search works...
    # Actually, let me use P0 priors for first char and then model for rest
    h = None

    # Process each character using the SAME hidden state pattern as beam search
    for i in range(1, len(s_text)):
        if i == 1:
            # Second character: score using hidden state after first char
            x = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
            _, h = model(x, h)
            # Now score next character
            x_prev = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
            logits, h_new = model(x_prev, h)
            # BUT beam search doesn't use h_new, it keeps using h!
            # Actually, let me think about this more carefully...

        # Actually, let me step back and think about what beam search actually does
        # The issue is that beam search scores all expansions from one state simultaneously
        # Let me simulate that pattern

        if i == 1:
            # After processing first character, get state
            x = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
            _, h = model(x, h)

        # Score character i using current hidden state h
        x_prev = torch.tensor([[idx[s_text[i-1]]]], dtype=torch.long, device=device)
        logits, h_next = model(x_prev, h)
        logp = F.log_softmax(logits[0, -1, :], dim=-1)
        score = float(logp[idx[s_text[i]]].item())

        beam_scores.append(score)
        beam_total += score
        print(f"Char {i} '{s_text[i]}': {score:.6f} (cumulative: {beam_total:.6f})")

        # Update hidden state for next iteration
        h = h_next

    print(f"Total beam score: {beam_total:.6f}")
    beam_bpc = -beam_total / (len(s_text) * torch.log(torch.tensor(2.0)).item())
    print(f"Beam BPC: {beam_bpc:.6f}")

    # === METHOD 2: Conditional BPC incremental scoring ===
    print(f"\n=== CONDITIONAL_BPC METHOD (correct hidden state management) ===")

    # Calculate total first
    total_bpc = conditional_bpc(model, alphabet, "", text, p0_log, str(device))
    print(f"Total conditional_bpc: {total_bpc:.6f}")

    # Now calculate incrementally to see individual scores
    h = None
    total_nll = 0.0
    cond_scores = []

    # First character from P0
    if p0_log is not None:
        first_nll = float(-p0_log[idx[s_text[0]]].item())
    else:
        # This case shouldn't happen with our model, but handle it
        raise ValueError("Need P0 for first character")

    cond_scores.append(-first_nll)  # Convert back to log prob
    total_nll += first_nll
    print(f"Char 0 '{s_text[0]}': {-first_nll:.6f} (cumulative nll: {total_nll:.6f})")

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

        cond_scores.append(char_logp)
        total_nll += char_nll
        print(f"Char {i} '{s_text[i]}': {char_logp:.6f} (cumulative nll: {total_nll:.6f})")

    manual_bpc = (total_nll / len(s_text)) / torch.log(torch.tensor(2.0)).item()
    print(f"Manual BPC: {manual_bpc:.6f}")

    # === COMPARISON ===
    print(f"\n=== DETAILED COMPARISON ===")
    print("Pos Char   Beam Score   Cond Score   Difference")
    print("--- ----   ----------   ----------   ----------")
    total_diff = 0.0
    for i in range(len(s_text)):
        diff = beam_scores[i] - cond_scores[i]
        total_diff += diff
        print(f"{i:3d} '{s_text[i]}'   {beam_scores[i]:10.6f}   {cond_scores[i]:10.6f}   {diff:10.6f}")

    print(f"\nTotal score difference: {total_diff:.6f}")
    print(f"BPC difference: {beam_bpc - total_bpc:.6f}")

if __name__ == "__main__":
    compare_incremental()