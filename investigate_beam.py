#!/usr/bin/env python3
"""
Investigate the beam search puzzle: why does pt bpc sometimes beat bpc
even though the true plaintext should be on the beam?
"""
import torch
from beam_subst17 import beam_decode, load_model, clean_text, prior_uniform, reconstruct

def investigate_beam_puzzle():
    model_path = "charlarge_pile80_last.pt"
    cipher_file = "test-data/test.ct.goodnews.txt"
    prompt = "#isaythat"
    true_pt = "#thegoodnewsyoucangetmostofthebenefitwithoutanyknownplaintext#"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet, char_priors = load_model(model_path, device=str(device))

    if char_priors is not None:
        p0_log = torch.log(char_priors.clamp_min(1e-30)).to(device)
    else:
        p0_log = None

    with open(cipher_file, 'r') as f:
        ct = f.read().strip()

    print("=== Investigating Beam Search Puzzle ===")
    print(f"True plaintext: {true_pt}")
    print(f"Prompt: {prompt}")
    print()

    # Run beam search with detailed inspection
    best_pt, c2p_full, g_raw, pack, pt_falloff = beam_decode(
        model, ct, alphabet,
        beam_size=500, micro=8192, topk_expand=26,
        prior_w=0.0, alpha=0.0, gumbel=0.05,
        prefix="", prompt=prompt,
        p0_log=p0_log,
        char_priors=char_priors, pt_debug=true_pt, homophonic_limit=0
    )

    states, nodes = pack
    print(f"\n=== Final Analysis ===")
    print(f"pt_falloff reported: {pt_falloff}")
    print(f"Final beam size: {len(states)}")
    print(f"Best result: {best_pt}")

    # Check if true plaintext path is actually on final beam
    true_pt_clean = clean_text(true_pt, alphabet)
    print(f"True plaintext (cleaned): {true_pt_clean}")
    print(f"Best result matches true PT: {best_pt == true_pt_clean}")

    # Examine all final beam states
    print(f"\n=== Final Beam States ===")
    for i, st in enumerate(states[:10]):  # Show top 10
        pt = reconstruct(nodes, st.node, alphabet)
        print(f"Beam {i}: score={st.g:.4f} len={st.length} text={pt}")
        if pt == true_pt_clean:
            print(f"  ^^^ TRUE PLAINTEXT FOUND ON BEAM AT POSITION {i}")

if __name__ == "__main__":
    investigate_beam_puzzle()