#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${TEACHER_BASE_URL:-https://api.openai.com/v1}"
API_KEY="${TEACHER_API_KEY:-}"
MODEL="${TEACHER_MODEL:-gpt-4o-mini}"
PORT="${PORT:-8642}"
cd "$(dirname "$0")"
export TEACHER_BASE_URL="$BASE_URL"
export TEACHER_API_KEY="$API_KEY"
export TEACHER_MODEL="$MODEL"

echo "[1/3] labeling with teacher (paid/private LLM) ..."
python teacher.py
echo "[2/3] distilling a tiny non-autoregressive head ..."
python train.py
echo "[3/3] starting local service on http://127.0.0.1:$PORT ..."
python serve.py --port "$PORT" &
SERVER_PID=$!
sleep 2

echo "--- /health ---"
curl -s "http://127.0.0.1:$PORT/health"; echo
echo "--- /decide ---"
curl -s -X POST "http://127.0.0.1:$PORT/decide" \
  -H 'Content-Type: application/json' \
  -d '{"text":"I was charged twice, please refund me and cancel my account."}'; echo
echo "Serving in background (PID $SERVER_PID)."