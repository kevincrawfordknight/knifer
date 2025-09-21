
# Knifer: Deciphering with a neural language model

This code uses a neural language model, along with beam search, to decipher text.

## Programs and scripts

### run.endtoend

Creates small train/test data, builds a small LM, and deciphers a text.  It takes about 10 minutes to run.

### makedata-small and makedata-large

Creates train/test data for an LSTM LM. Needs access to data directory ../knifer-data.

### lstm.py 

Trains an LSTM LM

Usage (fresh train):

```
python lstm.py \
  --train train-data/train.small.char.txt --test test-data/test.small.char.txt \
  --epochs 5 \
  --save charsmall.pt --ckpts ckpts.small --ckpt_every 2 \
  --emb 512 --hidden 512 --layers 3 --block 512 --bsz 64 \
  --lr 2e-3 --wd 0.01 --dropin 0.2 --droph 0.2 --dropout 0.2 
```

Usage (resume from checkpoint):

```
python lstm.py \
  --train train.char.txt --test test.char.txt \
  --epochs 10 \
  --save charsmall2.pt --resume ckpts/char_lstm_epoch8.pt
```

In the latter case, --epochs should refer to the total epochs aimed for, which will be greater than the epochs recorded in the --resume checkpoint.

### beam_subst.py 

Deciphers a ciphertext string by searching for a compatible plaintext with the best score (lowest bits-per-characters according to the LM).

Usage:

```
python beam_subst.py charsmall.pt test-data/test.ct.goodnews.txt \
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

A filename can be used in place of a string for these switches.

### score_lstm.py 

Scores any plaintext (candidate), in bits-per-character.

Usage:

```
python score_lstm.py <LM-file> <TXT-file>
```

Usage (with prompt):

```
python score_lstm.py <LM-file> <TXT-file> --prompt "#mynameis"
```

Note that score_lstm.py may return a different score on the same test set given to the lstm.py trainer. This is due to a difference in how the first character on each line is scored.

