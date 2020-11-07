#!/bin/sh
set -x

python -u train.py --data=cornell --model=Seq2Seq
python -u train.py --data=cornell --model=HRED
python -u train.py --data=cornell --model=VHRED
python -u train.py --data=cornell --model=VHCR
