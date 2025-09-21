#!/usr/bin/env python3

import torch
import math
from array import array

# Load the same model and setup as beam_subst23.py
def load_model(model_path):
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    print("Checkpoint keys:", list(checkpoint.keys()))

    # Model architecture (same as beam_subst23.py)
    class AWDCharLSTM(torch.nn.Module):
        def __init__(self, vocab_size=27, emb=512, hidden=512, layers=3,
                     p_in=0.0, p_h=0.0, p_out=0.0, tie_weights=False):
            super().__init__()
            self.encoder = torch.nn.Embedding(vocab_size, emb)
            self.drop_in  = torch.nn.Dropout(p_in)
            self.lstm = torch.nn.LSTM(emb, hidden, layers, batch_first=True, dropout=p_h)
            self.drop_out = torch.nn.Dropout(p_out)
            self.decoder = torch.nn.Linear(hidden, vocab_size, bias=False)
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

    # Get model config
    model_config = checkpoint['model_config']
    print("Model config:", model_config)
    print("Model state dict keys:", list(checkpoint['model'].keys()))

    # Get alphabet to determine vocab size
    alphabet = checkpoint['alphabet']
    vocab_size = len(alphabet)

    # Create model with checkpoint parameters
    model = AWDCharLSTM(
        vocab_size=vocab_size,
        emb=model_config['emb'],
        hidden=model_config['hidden'],
        layers=model_config['layers'],
        tie_weights=model_config['tie_weights']
    )

    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint

def get_lstm_state_after_sequence(model, sequence, alphabet, device):
    """Get LSTM hidden state after processing a sequence of characters."""
    with torch.no_grad():
        # Start with zero hidden state
        h_current = None

        for char in sequence:
            if char in alphabet:
                char_idx = alphabet.index(char)
                input_tensor = torch.tensor([[char_idx]], device=device)
                _, h_current = model(input_tensor, h_current)
            else:
                raise ValueError(f"Character '{char}' not in alphabet")

        return h_current

def get_all_continuations(model, prefix, alphabet, device):
    """Get scores for all possible 2-character continuations after prefix."""
    print(f"Getting LSTM state after '{prefix}'...")
    h_state = get_lstm_state_after_sequence(model, prefix, alphabet, device)

    results = []

    print("Computing all 676 possible continuations...")
    for first_char in alphabet:
        if first_char == '#':
            continue  # Skip hash for normal text

        first_idx = alphabet.index(first_char)

        # Get distribution for first continuation character
        dummy_input = torch.tensor([[0]], device=device)
        logits1, _ = model(dummy_input, h_state)
        probs1 = torch.softmax(logits1[0, -1], dim=0)
        first_prob = probs1[first_idx].item()
        first_logprob = math.log(max(first_prob, 1e-10))

        # Update state with first character
        actual_input1 = torch.tensor([[first_idx]], device=device)
        _, h_after_first = model(actual_input1, h_state)

        for second_char in alphabet:
            if second_char == '#':
                continue  # Skip hash for normal text

            second_idx = alphabet.index(second_char)

            # Get distribution for second continuation character
            dummy_input2 = torch.tensor([[0]], device=device)
            logits2, _ = model(dummy_input2, h_after_first)
            probs2 = torch.softmax(logits2[0, -1], dim=0)
            second_prob = probs2[second_idx].item()
            second_logprob = math.log(max(second_prob, 1e-10))

            # Total score is sum of log probabilities
            total_score = first_logprob + second_logprob
            continuation = first_char + second_char
            full_sequence = prefix + continuation

            results.append((total_score, full_sequence, continuation, first_prob, second_prob))

    # Sort by total score (descending, since higher log prob is better)
    results.sort(key=lambda x: x[0], reverse=True)
    return results

def main():
    model_path = "char916.pt"
    prefix = "re"
    alphabet = "#abcdefghijklmnopqrstuvwxyz"
    device = torch.device('cpu')

    model, checkpoint = load_model(model_path)
    model.to(device)

    print(f"Model vocabulary size: {len(checkpoint['alphabet'])}")
    print(f"Alphabet: {alphabet}")
    print()

    results = get_all_continuations(model, prefix, alphabet, device)

    print(f"All 676 possible 2-character continuations after '{prefix}' (sorted by LSTM score):")
    print(f"{'Rank':<4} {'Sequence':<8} {'Continuation':<12} {'Total Score':<12} {'1st Prob':<10} {'2nd Prob':<10}")
    print("-" * 70)

    for i, (total_score, full_sequence, continuation, first_prob, second_prob) in enumerate(results):
        print(f"{i+1:<4} {full_sequence:<8} {continuation:<12} {total_score:<12.6f} {first_prob:<10.6f} {second_prob:<10.6f}")

    print(f"\nShowing all {len(results)} results.")

if __name__ == "__main__":
    main()