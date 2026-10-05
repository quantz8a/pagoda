#!/bin/bash
# stop-train-restart.sh — free the GPU, run Laya screening fine-tune, restart the stack.
set -u
PAGODA_HOME=${PAGODA_HOME:-$HOME/pagoda}
cd $PAGODA_HOME/distill

echo "=== [1/4] stopping SGLang student on :31102 ==="
SPID=$(ss -tlnp 2>/dev/null | grep ':31102' | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "$SPID" ]; then kill "$SPID"; fi
sleep 6
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "=== [2/4] training (GPU) ==="
$PAGODA_HOME/bench-ref/venv/bin/python train_laya.py --out $PAGODA_HOME/distill/out/laya-screening > train-screening.log 2>&1
echo "train exit: $?"
tail -12 train-screening.log

echo "=== [3/4] restarting SGLang student on :31102 ==="
export PATH=$HOME/cuda12/hostbin:$HOME/cuda12/root/usr/local/cuda-12.8/bin:$PATH
cd $PAGODA_HOME/distill
nohup $HOME/sglang-diff-3050/.venv/bin/python -m sglang.launch_server \
  --model-path $PAGODA_HOME/distill/out/student-merged \
  --port 31102 --host 127.0.0.1 \
  --attention-backend torch_native --sampling-backend pytorch \
  --mem-fraction-static 0.55 --served-model-name student \
  > student-31102.log 2>&1 &
echo "student restarting, pid $!"

echo "=== [4/4] done (student warmup happens in background) ==="
