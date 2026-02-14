
# Knifer: Deciphering with a neural language model

This code uses a neural language model, along with beam search, to decipher text.

## Programs and scripts

### run.decipher

Demo script deciphering some stuff with a large, trained LM.

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

### beam_subst.py 

Deciphers a ciphertext string by searching for a compatible plaintext with the best score (lowest bits-per-characters according to the LM).

Usage:

```
python beam_subst.py charsmall.pt test-data/test.ct.goodnews.txt \
  --beam 20000 --nbest 20 \
  --incremental \
  --dtype bf16 \
  --topk 26 --alpha 0.00 --gumbel 0.05 --prior_w 0.0 --refine 1 
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

### EXAMPLE of --prompt idea

```
% python beam_subst24.py char916.pt test-data/test.ct.z13.txt --beam 2000 --nbest 10 --prefix "er" --dtype bf16

Top 10 results:
 1. bpc= 2.850 ertwasadapted#
 2. bpc= 2.861 eriwasanalien#
 3. bpc= 2.918 erthanadapted#
 4. bpc= 2.958 ericasanalien#
 5. bpc= 2.988 erihasanalien#
 6. bpc= 3.024 ertoanadapted#
 7. bpc= 3.024 erihadanalien#
 8. bpc= 3.046 ersmilitiaset#
 9. bpc= 3.059 ertialadapted#
10. bpc= 3.070 ertinandnoted#
KEY c->p: #xmateknz #adeprstw
```

```
% python beam_subst24.py char916.pt test-data/test.ct.z13.txt --beam 2000 --nbest 10 --prefix "er" --prompt "#mynameis" --dtype bf16

Top 10 results:
 1. bpc= 2.933 ericasanalien#
 2. bpc= 3.064 eriwasanalien#
 3. bpc= 3.082 erihasanalien#
 4. bpc= 3.131 ericamanalien#
 5. bpc= 3.195 ertoanadapted#
 6. bpc= 3.199 erihadanalien#
 7. bpc= 3.232 ericatanalien#
 8. bpc= 3.239 erikasanalien#
 9. bpc= 3.260 erisamanalien#
10. bpc= 3.309 erikamanalien#
KEY c->p: #xzantmek #aceilnrs
```

### Output of run.decipher with large LM

```
A=24
B=2000
DATADIR=../knifer-data
LM=char919.pt

python beam_subst$A.py $LM test-data/test.ct.goodnews.txt 
  --beam $B --nbest 20 --incremental --dtype bf16 

DECIPHERING: simple substitution
gurtbbqarjflbhpnatrgzbfgbsgurorarsvgjvgubhgnalxabjacynvagrkg#

bpc=3.093 e s.g=-2.144 s.h=0.000
bpc=2.330 th s.g=-3.231 s.h=0.000
bpc=1.728 the s.g=-3.596 s.h=0.000
bpc=1.936 ofth s.g=-5.368 s.h=0.000
bpc=1.586 ofthe s.g=-5.496 s.h=0.000
bpc=1.834 hasbee s.g=-7.629 s.h=0.000
bpc=1.572 hasbeen s.g=-7.629 s.h=0.000
bpc=1.700 nomatter s.g=-9.429 s.h=0.000
bpc=1.652 interrupt s.g=-10.316 s.h=0.000
bpc=1.744 theofficer s.g=-12.089 s.h=0.000
bpc=1.687 theofficers s.g=-12.874 s.h=0.000
bpc=1.728 thegoodnewsi s.g=-14.378 s.h=0.000
bpc=1.831 thegoodnewsfo s.g=-16.504 s.h=0.000
bpc=1.702 thegoodnewsfor s.g=-16.523 s.h=0.000
bpc=1.816 thegoodnewsfory s.g=-18.882 s.h=0.000
bpc=1.794 thegoodnewsyouca s.g=-19.896 s.h=0.000
bpc=1.695 thegoodnewsyoucan s.g=-19.969 s.h=0.000
bpc=1.846 thegoodnewsyoucang s.g=-23.031 s.h=0.000
bpc=1.782 thegoodnewsyoucange s.g=-23.463 s.h=0.000
bpc=1.693 thegoodnewsyoucanget s.g=-23.470 s.h=0.000
bpc=1.751 thegoodnewsyoucangeti s.g=-25.486 s.h=0.000
bpc=1.829 thegoodnewsyoucangetfo s.g=-27.884 s.h=0.000
bpc=1.919 thegoodnewsyoucangetmos s.g=-30.587 s.h=0.000
bpc=1.840 thegoodnewsyoucangetmost s.g=-30.596 s.h=0.000
bpc=1.811 thegoodnewsyoucangetmosto s.g=-31.377 s.h=0.000
bpc=1.742 thegoodnewsyoucangetmostof s.g=-31.394 s.h=0.000
bpc=1.727 thegoodnewsyoucangetmostoft s.g=-32.316 s.h=0.000
bpc=1.667 thegoodnewsyoucangetmostofth s.g=-32.345 s.h=0.000
bpc=1.623 thegoodnewsyoucangetmostofthe s.g=-32.621 s.h=0.000
bpc=1.690 thegoodnewsyoucangetmostofthep s.g=-35.136 s.h=0.000
bpc=1.690 thegoodnewsyoucangetmostofthere s.g=-36.314 s.h=0.000
bpc=1.820 thegoodnewsyoucangetmostoftheben s.g=-40.386 s.h=0.000
bpc=1.776 thegoodnewsyoucangetmostofthebene s.g=-40.626 s.h=0.000
bpc=1.724 thegoodnewsyoucangetmostofthebenef s.g=-40.638 s.h=0.000
bpc=1.675 thegoodnewsyoucangetmostofthebenefi s.g=-40.641 s.h=0.000
bpc=1.630 thegoodnewsyoucangetmostofthebenefit s.g=-40.688 s.h=0.000
bpc=1.770 thegoodnewsyoucangetmostofthebenefitw s.g=-45.407 s.h=0.000
bpc=1.758 thegoodnewsyoucangetmostofthebenefitwi s.g=-46.325 s.h=0.000
bpc=1.739 thegoodnewsyoucangetmostofthebenefitwit s.g=-47.017 s.h=0.000
bpc=1.695 thegoodnewsyoucangetmostofthebenefitwith s.g=-47.018 s.h=0.000
bpc=1.727 thegoodnewsyoucangetmostofthebenefitwitho s.g=-49.080 s.h=0.000
bpc=1.690 thegoodnewsyoucangetmostofthebenefitwithou s.g=-49.193 s.h=0.000
bpc=1.652 thegoodnewsyoucangetmostofthebenefitwithout s.g=-49.231 s.h=0.000
bpc=1.678 thegoodnewsyoucangetmostofthebenefitwithouta s.g=-51.176 s.h=0.000
bpc=1.678 thegoodnewsyoucangetmostofthebenefitwithoutan s.g=-52.332 s.h=0.000
bpc=1.650 thegoodnewsyoucangetmostofthebenefitwithoutany s.g=-52.573 s.h=0.000
bpc=1.695 thegoodnewsyoucangetmostofthebenefitwithoutanyp s.g=-55.214 s.h=0.000
bpc=1.758 thegoodnewsyoucangetmostofthebenefitwithoutanykn s.g=-58.480 s.h=0.000
bpc=1.725 thegoodnewsyoucangetmostofthebenefitwithoutanykno s.g=-58.581 s.h=0.000
bpc=1.691 thegoodnewsyoucangetmostofthebenefitwithoutanyknow s.g=-58.594 s.h=0.000
bpc=1.731 thegoodnewsyoucangetmostofthebenefitwithoutanyknown s.g=-61.188 s.h=0.000
bpc=1.762 thegoodnewsyoucangetmostofthebenefitwithoutanyknownp s.g=-63.516 s.h=0.000
bpc=1.757 thegoodnewsyoucangetmostofthebenefitwithoutanyknownpr s.g=-64.539 s.h=0.000
bpc=1.781 thegoodnewsyoucangetmostofthebenefitwithoutanyknownpla s.g=-66.647 s.h=0.000
bpc=1.826 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplai s.g=-69.600 s.h=0.000
bpc=1.794 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplain s.g=-69.628 s.h=0.000
bpc=1.774 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplaint s.g=-70.097 s.h=0.000
bpc=1.840 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplainte s.g=-73.956 s.h=0.000
bpc=1.817 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplaintex s.g=-74.320 s.h=0.000
bpc=1.787 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplaintext s.g=-74.340 s.h=0.000
bpc=1.809 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplaintext# s.g=-76.465 s.h=0.000

Top 20 results:
 1. bpc= 1.809 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplaintext#
 2. bpc= 2.050 thegoodnewsyoucangetmostofthebenefitwithoutanyknownpraintext#
 3. bpc= 2.051 thegoodnewsyoucangetmostoftheqenefitwithoutanyknownplaintext#
 4. bpc= 2.055 thegoodnewsyoucangetmostoftherenefitwithoutanyknownplaintext#
 5. bpc= 2.068 thegoodnewsyoucangetzostofthebenefitwithoutanyknownplaintext#
 6. bpc= 2.069 thegoodnewsyoucangetvostofthebenefitwithoutanyknownplaintext#
 7. bpc= 2.072 thegoodnewsyoucangetjostofthebenefitwithoutanyknownplaintext#
 8. bpc= 2.079 thegoodnewsyoucangetmostofthevenefitwithoutanyknownplaintext#
 9. bpc= 2.083 thegoodnewsyoucangetrostofthebenefitwithoutanyknownplaintext#
10. bpc= 2.092 thegoodnewsyoucangetmostofthejenefitwithoutanyknownplaintext#
11. bpc= 2.099 thegoodnewsyoucangetmostofthebenefitwithoutanyrnownplaintext#
12. bpc= 2.100 thegoodnewsyourangetmostofthebenefitwithoutanyknownplaintext#
13. bpc= 2.104 thegoodnewsyoucangetmostofthebenefitwithoutanyknownrvaintext#
14. bpc= 2.116 thegoodnewsyoucangetmostofthebenefitwithoutanyknownplaintezt#
15. bpc= 2.121 thegoodnewsyoucangetmostofthebenefitwithoutanyknownrpaintext#
16. bpc= 2.130 thegoodnewsyoucangetmostofthebenefitwithoutanyknownqlaintext#
17. bpc= 2.132 thegoodnewsyoucangetmostofthezenefitwithoutanyknownplaintext#
18. bpc= 2.133 thegoodnewsyoujangetmostofthebenefitwithoutanyknownplaintext#
19. bpc= 2.135 thegoodnewsyoucangetmostofthebenefitwithoutanyknownlraintext#
20. bpc= 2.136 thegoodnewsyoucangetmostofthebenefitwithoutanyknownjpaintext#
KEY c->p:
#nopqrstuvxyzabcfghjkl
#abcdefghiklmnopstuwxy
```
