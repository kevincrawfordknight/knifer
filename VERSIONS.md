
# Version notes for programs.

## LSTM (LM building)

### lstm v1 — Initial AWD char-LSTM trainer (TBPTT), prints train/test BPC+PPL, saves state_dict.

### lstm v2 — Disabled weight tying by default (emb≠hidden OK); sturdier DataLoader/drop_last behavior.

### lstm v3 — Fixed IterableDataset length, cleaner TBPTT hidden handling, param count print.

### lstm v4 — 27-char alphabet support (#abcdefghijklmnopqrstuvwxyz) with strict cleaning.

### lstm v5 — Resume training: load checkpoints (model/optimizer), periodic full ckpts with metadata.

## Beam Subst (deciphering)

### beam_subst v1 — Basic LM-scored beam search for simple substitution (top-k expand, key constraints).

### beam_subst v2 — Minor stability/bug fixes (key/mapping edge cases).

### beam_subst v3 — No more “?” placeholders: always emit a plaintext char; stricter mapping consistency.

### beam_subst v7 — Added --lookahead, --gumbel, length norm --alpha, heuristic --prior_w, --prefix, and --refine.

### beam_subst v8 — Major speed/memory win: micro-batched LSTM steps, hidden-state reuse, compact key arrays; removed --lookahead.

### beam_subst v9 — N-best output (sorted) and --pt with “fall-off index” reporting.

### beam_subst v11 — --prompt warms the LM; global score includes prompt+decoding under the same boundary policy.

### beam_subst v12 — 27-char # support with fixed #-># mapping; unified P₀/prompt boundary handling.

### beam_subst v14 — Inference dtype switch (--dtype fp32/fp16/bf16) + micro-batch knob; stable N-best rescoring.

### beam_subst v15 — Fixed --prefix hard-enforcement; consistent conditional-BPC rescoring with P0/prompt.

## Score (scoring plaintexts)

### score v1 — Simple 26-char next-char BPC/PPL scorer (no prompt).

### score v2 — 27-char # support and safe torch.load (suppressed warning).

### score v3 — --prompt warm-start and (optional) --skip_first; score excludes prompt tokens.

### score v4 — Boundary-compatible scoring: first char via P0 (or model-conditional with --prompt); --prior_text + smoothing.

