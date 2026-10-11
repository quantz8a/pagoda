#!/usr/bin/env bash
# Pagoda × mlx-serve 共部署（macOS / Apple Silicon）
#
# Qwen3.8 等 Gated-DeltaNet 模型不能走 pagoda-hf/Candle（只覆盖 Llama/Qwen2 系）。
# 本机用 mlx-serve 做引擎，pagoda 当网关：/health /stats /checkpoint* 本地，
# /v1/chat/completions /generate 透传到 mlx-serve。
#
# 用法（在仓库 pagoda/ 子目录执行）：
#   bash scripts/co-deploy-mlx.sh
#   MLX_PORT=11236 PAGODA_PORT=30000 bash scripts/co-deploy-mlx.sh
#
# 前提：mlx-serve 已在跑且模型已 ready，例如：
#   mlx-serve serve --host 127.0.0.1 --port 11236 \
#     --model ~/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit
set -euo pipefail
cd "$(dirname "$0")/.."

PAGODA_PORT="${PAGODA_PORT:-30000}"
# Auto-detect a healthy mlx-serve if MLX_PORT is unset / stale.
MODEL="${MODEL:-mlx-community/Qwen3.8-27B-8bit}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

detect_mlx_port() {
  if [ -n "${MLX_PORT:-}" ]; then
    echo "$MLX_PORT"
    return
  fi
  for port in 11236 11246 11241 11234 11239; do
    if curl -fsS --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "$port"
      return
    fi
  done
  # fall back to any listening mlx-serve
  local p
  p=$(lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk '/mlx-serve/ {print $9}' | sed -n 's/.*://p' | head -1)
  if [ -n "$p" ]; then
    echo "$p"
    return
  fi
  echo "!! 未找到正在监听的 mlx-serve；先启动：
  mlx-serve serve --host 127.0.0.1 --port 11236 \\
    --model ~/.mlx-serve/models/mlx-community/Qwen3.8-27B-8bit" >&2
  exit 1
}

wait_url() {
  local url="$1" name="$2" timeout="$3"
  local deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if curl -fsS --max-time 2 "$url" >/dev/null 2>&1; then
      echo "    $name OK: $url"
      return 0
    fi
    sleep 2
  done
  echo "!! $name 在 ${timeout}s 内未就绪: $url" >&2
  exit 1
}

MLX_PORT="$(detect_mlx_port)"
MLX_URL="http://127.0.0.1:${MLX_PORT}"

echo "==> [1/3] 检查 mlx-serve ($MLX_URL)"
wait_url "$MLX_URL/health" "mlx-serve /health" 30
python3 - <<PY
import json, sys, time, urllib.request
deadline = time.time() + 300
while True:
    raw = urllib.request.urlopen("$MLX_URL/v1/models", timeout=5).read()
    data = json.loads(raw)["data"]
    m = next((x for x in data if x["id"] == "$MODEL"), None)
    if not m:
        ids = ", ".join(x["id"] for x in data)
        sys.exit(f"模型未注册: $MODEL（现有: {ids}）")
    if m.get("loaded"):
        print(f"    model ready: {m['id']} ({m.get('bytes_resident', 0)} bytes resident)")
        break
    if time.time() > deadline:
        sys.exit(f"模型 300s 内未就绪: $MODEL state={m.get('state')}")
    print(f"    waiting for load: state={m.get('state')}")
    time.sleep(5)
PY

echo "==> [2/3] 构建 pagoda（release）"
CARGO_TARGET_DIR="$(pwd)/target" cargo build --release --bin pagoda

echo "==> [3/3] 启动 pagoda 网关 :$PAGODA_PORT → $MLX_URL"
if lsof -nP -iTCP:"$PAGODA_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "    端口 $PAGODA_PORT 已被占用，先停旧进程"
  lsof -nP -iTCP:"$PAGODA_PORT" -sTCP:LISTEN -t | xargs kill || true
  sleep 1
fi
nohup ./target/release/pagoda serve --port "$PAGODA_PORT" --upstream "$MLX_URL" \
  > "$LOG_DIR/pagoda-gateway.log" 2>&1 &
PAGODA_PID=$!
echo "    pagoda PID=$PAGODA_PID，日志 $LOG_DIR/pagoda-gateway.log"
wait_url "http://127.0.0.1:$PAGODA_PORT/health" "pagoda /health" 30

cat <<EOF

共部署完成（Qwen3.8 via mlx-serve）
  入口:  http://127.0.0.1:$PAGODA_PORT
  上游:  $MLX_URL  ($MODEL)

冒烟:
  curl -sS http://127.0.0.1:$PAGODA_PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"model":"$MODEL","messages":[{"role":"user","content":"你好"}],"max_tokens":32}'

停止网关:
  kill $PAGODA_PID
EOF
