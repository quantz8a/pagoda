#!/usr/bin/env bash
# Pagoda × SGLang 一键共部署（Linux/macOS）
#
# 架构：pagoda 当控制面网关（:30000），SGLang 当 GPU worker（:30001）。
#   客户端 → pagoda  /health /stats /checkpoint* 由 pagoda 本地服务
#                     /generate /v1/chat/completions 透明转发给 SGLang
#
# 用法：在 pagoda 目录下执行
#   bash scripts/co-deploy.sh [模型repo]           # 默认 TinyLlama/TinyLlama-1.1B-Chat-v1.0
#   INSTALL_SGLANG=1 bash scripts/co-deploy.sh     # 首次：自动建 venv 并 pip install sglang
set -euo pipefail
cd "$(dirname "$0")/.."   # 仓库 pagoda/ 根目录

MODEL="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PAGODA_PORT="${PAGODA_PORT:-30000}"
SGLANG_PORT="${SGLANG_PORT:-30001}"

wait_health() {  # $1=url $2=name $3=timeout_sec
  local deadline=$(( $(date +%s) + $3 ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 2 "$1/health" >/dev/null 2>&1; then
      echo "    $2 健康检查通过: $1"; return 0
    fi
    sleep 2
  done
  echo "!! $2 在 $3s 内未就绪（$1）" >&2; exit 1
}

echo "==> [1/4] 构建 pagoda（release）"
cargo build --release

echo "==> [2/4] 准备 SGLang（python）"
PY="python3"
if [ "${INSTALL_SGLANG:-0}" = "1" ]; then
  [ -d .venv-sglang ] || python3 -m venv .venv-sglang
  PY=".venv-sglang/bin/python"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install "sglang[all]"
fi
"$PY" -c "import sglang" 2>/dev/null || {
  echo "!! 未检测到 sglang。用 INSTALL_SGLANG=1 bash scripts/co-deploy.sh 安装" >&2; exit 1; }

echo "==> [3/4] 启动 SGLang worker（模型 $MODEL，端口 $SGLANG_PORT）"
"$PY" -m sglang.launch_server --model-path "$MODEL" --port "$SGLANG_PORT" \
  > sglang-worker.log 2>&1 &
SGLANG_PID=$!
echo "    SGLang PID=$SGLANG_PID，日志 sglang-worker.log"
wait_health "http://127.0.0.1:$SGLANG_PORT" "SGLang" 600   # 首次要下载模型+编译 kernel

echo "==> [4/4] 启动 pagoda 网关（端口 $PAGODA_PORT → 上游 $SGLANG_PORT）"
./target/release/pagoda serve --port "$PAGODA_PORT" \
  --upstream "http://127.0.0.1:$SGLANG_PORT" > pagoda-gateway.log 2>&1 &
PAGODA_PID=$!
echo "    pagoda PID=$PAGODA_PID，日志 pagoda-gateway.log"
wait_health "http://127.0.0.1:$PAGODA_PORT" "pagoda" 30

cat <<EOF

共部署完成！
  对外入口（客户端只认这个）:  http://127.0.0.1:$PAGODA_PORT
    POST /generate               → 转发 SGLang
    POST /v1/chat/completions    → 转发 SGLang
    GET  /health  /stats         → pagoda 本地（含 upstream / proxied_requests）
    POST /checkpoint*            → pagoda 本地（agent 原语）
  冒烟测试:
    curl -X POST http://127.0.0.1:$PAGODA_PORT/generate -H 'Content-Type: application/json' \
      -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16}}'
  停止:
    kill $PAGODA_PID $SGLANG_PID
EOF