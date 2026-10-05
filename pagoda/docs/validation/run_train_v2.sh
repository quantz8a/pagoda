#!/bin/bash
set -e
PAGODA_HOME=${PAGODA_HOME:-$HOME/pagoda}
PY=$PAGODA_HOME/bench-ref/venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $PAGODA_HOME/distill
nohup $PY train_laya.py --seeds seeds/abstracts-real-v2.jsonl \
  --out out/laya-screening-v2 --fn-weight 8 --epochs 6 > train-v2.log 2>&1 &
echo "train pid: $!"
sleep 60
tail -8 train-v2.log
