#!/usr/bin/env python3
"""
Debug exactly how beam search accumulates scores vs what conditional_bpc does
"""
import torch
import torch.nn.functional as F
from beam_subst17 import load_model, conditional_bpc, clean_text

def trace_beam_accumulation():
    """Trace step-by-step score accumulation for #i"""
    model_path = "charlarge_pile80_last.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, char_priors = load_model(model_path, device=str(device))

    if char_priors is not None:
        p0_log = torch.log(char_priors.clamp_min(1e-30)).to(device)
    else:
        p0_log = None

    prompt = "#isaythat"
    text = "#i"

    s_prompt = clean_text(prompt, alphabet)
    s_text = clean_text(text, alphabet)
    idx = {c: i for i, c in enumerate(alphabet)}

    print(f"Tracing beam accumulation for: '{text}' with prompt: '{prompt}'")
    print(f"Cleaned prompt: '{s_prompt}', cleaned text: '{s_text}'")

    # === STEP 1: Simulate beam search scoring ===
    print(f"\n=== BEAM SEARCH SIMULATION ===")

    # Process prompt (like run_prompt)
    h = None
    print(f"Processing prompt: {s_prompt}")
    for t in range(1, len(s_prompt)):
        prev_char = s_prompt[t-1]
        x = torch.tensor([[idx[prev_char]]], dtype=torch.long, device=device)
        _, h = model(x, h)
        print(f"  Fed prompt char '{prev_char}' to model")

    # Get logits after last prompt char 't'
    x_last = torch.tensor([[idx[s_prompt[-1]]]], dtype=torch.long, device=device)  # 't'
    logits, h = model(x_last, h)
    logp_next = F.log_softmax(logits[0, -1, :], dim=-1)

    # Score first text char '#'
    first_score = float(logp_next[idx[s_text[0]]].item())
    print(f"Score for '{s_text[0]}' after prompt: {first_score:.6f}")
    beam_total = first_score

    # Feed first char '#'
    x0 = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
    _, h = model(x0, h)

    # Get logits for second char 'i'
    x_prev = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)  # '#'
    logits, h = model(x_prev, h)

    # IMPORTANT: Simulate the detach that beam search does
    h = (h[0].detach(), h[1].detach())

    logp = F.log_softmax(logits[0, -1, :], dim=-1)
    second_score = float(logp[idx[s_text[1]]].item())
    print(f"Score for '{s_text[1]}' after '{s_text[0]}': {second_score:.6f}")
    beam_total += second_score

    print(f"Total beam score: {beam_total:.6f}")
    beam_bpc = -beam_total / (len(s_text) * torch.log(torch.tensor(2.0)).item())
    print(f"Beam BPC: {beam_bpc:.6f}")

    # === STEP 2: conditional_bpc scoring ===
    print(f"\n=== CONDITIONAL_BPC CALCULATION ===")
    fresh_bpc = conditional_bpc(model, alphabet, prompt, text, p0_log, str(device))
    print(f"Fresh BPC: {fresh_bpc:.6f}")

    print(f"\n=== COMPARISON ===")
    print(f"Beam total: {beam_total:.6f}")
    print(f"Beam BPC: {beam_bpc:.6f}")
    print(f"Fresh BPC: {fresh_bpc:.6f}")
    print(f"Difference: {abs(beam_bpc - fresh_bpc):.6f}")

    # === STEP 3: Manual conditional_bpc to see what it does ===
    print(f"\n=== MANUAL CONDITIONAL_BPC TRACE ===")

    # Reset and manually do what conditional_bpc does
    h = None
    total_nll = 0.0

    # Process prompt
    if len(s_prompt) >= 1:
        for t in range(1, len(s_prompt)):
            x = torch.tensor([[idx[s_prompt[t-1]]]], dtype=torch.long, device=device)
            _, h = model(x, h)
        lastp = torch.tensor([[idx[s_prompt[-1]]]], dtype=torch.long, device=device)
        logits, h = model(lastp, h)
        logp = F.log_softmax(logits[0,-1,:], dim=-1)
        first_nll = float(-logp[idx[s_text[0]]].item())
        print(f"First char NLL: {first_nll:.6f} (logp: {-first_nll:.6f})")
        total_nll += first_nll

    # Feed first char and continue
    x0 = torch.tensor([[idx[s_text[0]]]], dtype=torch.long, device=device)
    _, h = model(x0, h)

    for t in range(1, len(s_text)):
        x = torch.tensor([[idx[s_text[t-1]]]], dtype=torch.long, device=device)
        logits, h = model(x, h)
        logp = F.log_softmax(logits[0,-1,:], dim=-1)
        char_nll = float(-logp[idx[s_text[t]]].item())
        print(f"Char '{s_text[t]}' NLL: {char_nll:.6f} (logp: {-char_nll:.6f})")
        total_nll += char_nll

    manual_bpc = (total_nll / len(s_text)) / torch.log(torch.tensor(2.0)).item()
    print(f"Manual total NLL: {total_nll:.6f}")
    print(f"Manual BPC: {manual_bpc:.6f}")

if __name__ == "__main__":
    trace_beam_accumulation()