# Interpretability for Language Learners Using Example-Based Grammatical Error Correction

Code for the paper: "[Interpretability for Language Learners Using Example-Based Grammatical Error Correction](https://arxiv.org/abs/2203.07085)" (In ACL 2022).
If you use any part of this work, make sure you include the following citation:
```
@inproceedings{Kaneko:ACL:2022,
    title={Interpretability for Language Learners Using Example-Based Grammatical Error Correction},
    author={Masahiro Kaneko, Sho Takase, Ayana Niwa, Naoaki Okazaki},
    booktitle={Proc. of the 60th Annual Meeting of the Association for Computational Linguistics (ACL)},
    year={2022}
}
```

## Setup

We use Python version == 3.7.10.

All requirements can be found in `requirements.txt`. You can install all required packages with following:
```
pip install -r requirements.txt
```
You can also install fairseq with following:
```
cd knnmt
pip install --editable ./
```
You need to place `train.src, train.trg, dev.src, dev.trg, test.src` in `data` directory.

## To train EB-GEC model

You can train EB-GEC model using `train.sh` in `scripts` directory.

```
cd scripts
./train.sh $seed
```

## To generate corrected texts and examples with EB-GEC model

You can generate corrected texts and examples with EB-GEC model using `generate.sh` in `scripts` directory.

```
cd scripts
./generate.sh $seed
```

In the output nbest file, `SRC_EXAMPLE` and `TGT_EXAMPLE` for `EDIT` (`.` &rarr; `:`) for source `S` and detokenized output `D` are displayed as follows:
```
S-527 There are three reasons as follows .
D-527 -0.24829396605491638  There are three reasons as follows :
EDIT-527  :
SRC_EXEMPLE-527 there are two big reasons as follows .
TGT_EXAMPLE-527 There are two big reasons as follows :
```

