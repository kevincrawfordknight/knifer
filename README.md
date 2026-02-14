
# Knifer: Deciphering with a neural language model

This code uses a neural language model, along with beam search, to decipher text.
(Note that the 75M language models are too large to be included the github site, but are available.)

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

### EXAMPLE of --prompt idea (warm up the LM with context string)

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

#### Simple substitution

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

#### Homophonic substitution

```
python beam_subst17.py charlarge2.pt test-data/test.ct.z408.txt   --beam 50000 --nbest 1 --topk 26 --alpha 0.00 --gumbel 0.05 --prior_w 0.0 --refine 1   --dtype bf16 --homophonic 7 --prefix "ilike" --pt "ilikekillingpeoplebecauseitissomuchfunitismorefunthankillingwildgameintheforrestbecausemanisthemost"
Using alphabet from checkpoint
Using character priors from model checkpoint
Homophonic mode: cipher vocab size = 55, plaintext vocab size = 27, max 7 cipher symbols per plaintext
i
il
ili
ilik
ilike
ilikek
ilikeki
ilikekin
ilikekill
ilikekilli
ilikekillin
ilikekilling
ilikekillingt
ilikekillingth
ilikekillingthe
ilikekillingthat
ilikekillingpeopl
ilikekillingpeople
ilikekillingpeoplew
ilikekillingpeoplewh
ilikekillingpeoplewho
ilikekillingpeoplewith
ilikekillingpeopleinthe
ilikekillingpeoplewhoare
ilikekillingpeoplewhohave
ilikekillingpeoplewiththei
ilikekillingpeoplewiththeir
ilikekillingpeoplewithpoliti
ilikekillingpeoplewithpolitic
ilikekillingpeoplewithpolitica
ilikekillingpeoplewithpolitical
ilikekillingpeoplewhoparticipate
ilikekillingpeopleinthephilippine
ilikekillingpeoplewhoarelivingthro
ilikekillingpeoplewhoarelivingthrou
ilikekillingpeoplewhoarelivingthroug
ilikekillingpeoplewiththeirintentthat
ilikekillingpeoplewithindividualitywit
ilikekillingpeoplethetraditionalrecordi
ilikekillingpeoplethetraditionalrecordin
ilikekillingpeopleinthebritishthetelevisi
ilikekillingpeopleinthebritishthetelevisio
ilikekillingpeopleinthephilippinetelevision
ilikekillingpeopleinthephilippinetelevisions
ilikekillingpeopleinthephilippinetelevisionse
ilikekillingpeopleinthephilippinetelevisionser
ilikekillingpeopleinthephilippinetelevisionseri
*** True plaintext fell off beam at position 46 ***
ilikekillingpeopleinthephilippinetelevisionserie
ilikekillingpeopleinthephilippinetelevisionseries
ilikekillingpeopleinthephilippinetelevisionseriest
ilikekillingpeopleinthephilippinetelevisionseriesth
ilikekillingpeopleinthephilippinetelevisionseriesand

==== beam 500k ===

(base) % python beam_subst19.py charlarge_pile80_last.pt test-data/test.ct.z408.txt   --beam 500000 --nbest 1   --topk 26 --alpha 0.00 --gumbel 0.05 --prior_w 0.0 --dtype bf16 --refine 0 --prefix "ilike" --homophonic 7 --incremental --pt "ilikekillingpeoplebecauseitissomuchfunitismorefunthankillingwildgameintheforrestbecauseman" 
Using alphabet from checkpoint
Using character priors from model checkpoint
bpc=3.769 (pt bpc=3.769) i
bpc=4.466 (pt bpc=4.466) il
bpc=4.157 (pt bpc=4.157) ili
bpc=3.808 (pt bpc=3.808) ilik
bpc=3.070 (pt bpc=3.070) ilike
bpc=3.708 (pt bpc=3.708) ilikek
bpc=3.415 (pt bpc=3.415) ilikeki
bpc=3.163 (pt bpc=3.267) ilikekno
bpc=2.933 (pt bpc=2.933) ilikekill
bpc=2.729 (pt bpc=2.729) ilikekilli
bpc=2.482 (pt bpc=2.482) ilikekillin
bpc=2.276 (pt bpc=2.276) ilikekilling
bpc=2.298 (pt bpc=2.524) ilikekillingt
bpc=2.150 (pt bpc=2.478) ilikekillingth
bpc=2.024 (pt bpc=2.337) ilikekillingthe
bpc=2.155 (pt bpc=2.191) ilikekillingthat
bpc=2.062 (pt bpc=2.062) ilikekillingpeopl
bpc=1.948 (pt bpc=1.948) ilikekillingpeople
bpc=1.967 (pt bpc=2.098) ilikekillingpeoplew
bpc=1.914 (pt bpc=2.067) ilikekillingpeoplewh
bpc=1.829 (pt bpc=2.059) ilikekillingpeoplewho
bpc=1.787 (pt bpc=1.975) ilikekillingpeoplewith
bpc=1.807 (pt bpc=1.890) ilikekillingpeopleinthe
bpc=1.738 (pt bpc=1.812) ilikekillingpeoplewhoare
bpc=1.686 (pt bpc=1.739) ilikekillingpeoplewhohave
bpc=1.752 (pt bpc=1.752) ilikekillingpeoplebecausei
bpc=1.695 (pt bpc=1.737) ilikekillingpeoplewiththeir
bpc=1.751 (pt bpc=1.792) ilikekillingpeoplewithpoliti
bpc=1.691 (pt bpc=1.734) ilikekillingpeoplewithpolitic
bpc=1.643 (pt bpc=1.814) ilikekillingpeoplewithpolitica
bpc=1.590 (pt bpc=1.804) ilikekillingpeoplewithpolitical
bpc=1.556 (pt bpc=1.798) ilikekillingpeoplewhoparticipate
bpc=1.622 (pt bpc=1.847) ilikekillingpeopleinthephilippine
bpc=1.716 (pt bpc=1.793) ilikekillingpeoplewhoarelivingthro
bpc=1.667 (pt bpc=1.742) ilikekillingpeoplewhoarelivingthrou
bpc=1.620 (pt bpc=1.814) ilikekillingpeoplewhoarelivingthroug
bpc=1.675 (pt bpc=1.823) ilikekillingpeoplesaidtheirinvestigat
bpc=1.635 (pt bpc=1.791) ilikekillingpeopleperhapsitisclearthat
bpc=1.674 (pt bpc=1.844) ilikekillingpeopleperhapsitisclearthati
bpc=1.659 (pt bpc=1.856) ilikekillingpeoplefromreligiousproportio
bpc=1.673 (pt bpc=1.881) ilikekillingpeopleperhapsitisclearthatiti
bpc=1.634 (pt bpc=1.838) ilikekillingpeopleperhapsitisclearthatitis
bpc=1.639 (pt bpc=1.909) ilikekillingpeopleinthephilistinetelevision
bpc=1.639 (pt bpc=1.908) ilikekillingpeopleinthephilistinetelevisions
bpc=1.642 (pt bpc=1.875) ilikekillingpeopleinthephilistinetelevisionse
bpc=1.621 (pt bpc=1.835) ilikekillingpeopleinthephilistinetelevisionser
bpc=1.590 (pt bpc=1.900) ilikekillingpeopleinthephilistinetelevisionseri
bpc=1.557 (pt bpc=1.919) ilikekillingpeopleinthephilistinetelevisionserie
bpc=1.525 (pt bpc=1.886) ilikekillingpeopleinthephilistinetelevisionseries
*** True plaintext fell off beam at position 49 ***
bpc=1.551 (pt bpc=1.901) ilikekillingpeopleinthephilistinetelevisionseriesa
```
