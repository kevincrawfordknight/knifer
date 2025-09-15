
# Knifer: Deciphering with a neural language model

This code uses a neural language model, along with beam search, to decipher text.

## Programs and scripts

### run.endtoend

Creates small train/test data, builds a small LM, and deciphers a text.  It takes about 10 minutes to run.

### makedata-small

Creates a small LSTM LM.

### makedata-large

Creates a large LSTM LM. Needs access to data directory ../knifer-data.

### lstm.py 

Trains an LSTM LM

Usage (fresh train):

```
python lstm6.py \
  --train train-data/train.small.char.txt --test test-data/test.small.char.txt \
  --epochs 5 \
  --emb 512 --hidden 512 --layers 3 --block 512 --bsz 64 \
  --lr 2e-3 --wd 0.01 --dropin 0.2 --droph 0.2 --dropout 0.2 \
  --save charsmall.pt # --save_dir ckpts.small --save_every 2 --full_ckpt
```

Usage (resume from checkpoint):

```
python lstm6.py --train train.char.txt --test test.char.txt \
  --epochs 10 \
  --resume ckpts/char_lstm_epoch8.pt --save charlarge.pt 
```

Usage (resume from weights only):

```
python lstm6.py --train train.char.txt --test test.char.txt \
  --resume char27_best.pt --resume_strict \
  --save char27_best_resumed.pt --save_dir ckpts2 --save_every 1 --full_ckpt
```

### beam_subst.py 

Deciphers a ciphertext string by searching for a compatible plaintext with the best score (lowest bits-per-characters according to the LM).

Usage:

```
python beam_subst16.py charsmall.pt test-data/test.ct.goodnews.txt \
  --beam 20000 --nbest 20 \
  --topk 26 --alpha 0.00 --gumbel 0.05 --prior_w 0.0 --refine 1 \
  --dtype bf16
```

Usage (with crib prefix):

```
--prefix "string"     Crib plaintext for initial part, to help decipherment.
```

Usage (with prompt):

```
--prompt "string"     Warms up language model before scoring plaintext candidates.
```

Usage (with plaintext for beam):

```
--pt "string"         Reports position where true plaintext falls off the beam.
```

### score_lstm.py 

Scores any plaintext (candidate), in bits-per-character.

Usage:

```
python score5_lstm.py <LM-file> <TXT-file>
```

Usage (with prompt):

```
python score5_lstm.py <LM-file> <TXT-file> --prompt "#mynameis"
```

Note that score_lstm.py may return a different score on the same test set given to the lstm.py trainer. This is due to a difference in how the first character on each line is scored.

