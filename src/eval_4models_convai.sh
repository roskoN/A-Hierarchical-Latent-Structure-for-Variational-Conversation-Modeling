#!/bin/sh
set -x

python -u eval_convai.py --model=Seq2Seq --data=cornell --checkpoint="conversation/cornell/Seq2Seq/2019-10-10_12:33:01/30.pkl"
python -u eval_convai.py --model=HRED    --data=cornell --checkpoint="conversation/cornell/HRED/2019-10-10_13:52:56/30.pkl"
python -u eval_convai.py --model=VHRED   --data=cornell --checkpoint="conversation/cornell/VHRED/2019-10-10_15:34:42/30.pkl"
python -u eval_convai.py --model=VHCR    --data=cornell --checkpoint="conversation/cornell/VHCR/2019-10-10_17:30:08/30.pkl"
