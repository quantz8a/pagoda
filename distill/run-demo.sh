#!/bin/bash
# run-demo.sh — after training completes: tuned Laya (:31181) + student (:31102)
# + screening demo. Base Laya (:31180) and gateway (:31100) stay untouched.
set -u
PAGODA_HOME=${PAGODA_HOME:-$HOME/pagoda}
cd $PAGODA_HOME/distill
BIN=$PAGODA_HOME/pagoda-hf/target/release/laya_server
OUT=$PAGODA_HOME/distill/out/laya-screening

echo "=== [1/4] tuned laya_server on :31181 (CPU, mirrors base :31180) ==="
pids=$(ss -tlnp 2>/dev/null | grep ':31181' | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "$pids" ] && kill $pids
nohup $BIN --port 31181 --model-dir $OUT > laya-31181.log 2>&1 &
echo "laya tuned pid $!"

echo "=== [2/4] student SGLang on :31102 ==="
export PATH=$HOME/cuda12/hostbin:$HOME/cuda12/root/usr/local/cuda-12.8/bin:$PATH
pids=$(ss -tlnp 2>/dev/null | grep ':31102' | grep -oP 'pid=\K[0-9]+' | head -1)
[ -n "$pids" ] && kill $pids
nohup $HOME/sglang-diff-3050/.venv/bin/python -m sglang.launch_server \
  --model-path $PAGODA_HOME/distill/out/student-merged \
  --port 31102 --host 127.0.0.1 \
  --attention-backend torch_native --sampling-backend pytorch \
  --mem-fraction-static 0.55 --served-model-name student \
  > student-31102.log 2>&1 &
echo "student pid $!"

echo "=== [3/4] wait for health ==="
for i in $(seq 1 60); do
  ok=0
  curl -sf http://127.0.0.1:31181/health >/dev/null 2>&1 && ok=$((ok+1))
  curl -sf http://127.0.0.1:31102/health >/dev/null 2>&1 && ok=$((ok+1))
  [ $ok -eq 2 ] && break
  sleep 5
done
curl -sf http://127.0.0.1:31181/health >/dev/null && echo "laya-31181 OK" || { echo "LAYA FAIL"; tail -5 laya-31181.log; }
curl -sf http://127.0.0.1:31102/health >/dev/null && echo "student-31102 OK" || { echo "STUDENT FAIL"; tail -5 student-31102.log; }

echo "=== [4/4] screening demo ==="
$PAGODA_HOME/bench-ref/venv/bin/python demo_screening.py
