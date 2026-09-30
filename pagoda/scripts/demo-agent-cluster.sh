#!/usr/bin/env bash
# Pagoda agent 集群演示（Linux/macOS）
# 启动服务 → 创建 checkpoint（共享树干）→ 3 个 agent 分支 → 看省了多少算力
# 用法: bash scripts/demo-agent-cluster.sh [PORT]
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8080}"
BASE="http://127.0.0.1:$PORT"

echo "==> 构建并启动 pagoda 服务于 $BASE"
cargo build --offline >/dev/null
./target/debug/pagoda serve --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

for i in $(seq 1 40); do
    if curl -sf "$BASE/health" >/dev/null 2>&1; then break; fi
    sleep 0.25
    if [ "$i" -eq 40 ]; then echo "服务未能启动" >&2; exit 1; fi
done
echo "    服务已就绪"

TRUNK="你是客服团队的自主 agent。共享规则：礼貌、简洁、先查证再回答。工具说明：……"
echo ""
echo "==> [1/3] 创建 checkpoint（共享树干）"
CP=$(curl -sf -X POST "$BASE/checkpoint" -H 'Content-Type: application/json' \
    -d "{\"text\":\"$TRUNK\"}")
echo "    $CP"
CPID=$(echo "$CP" | grep -o '"checkpoint_id":[0-9]*' | grep -o '[0-9]*')

echo ""
echo "==> [2/3] 启动 3 个 agent 分支（共享同一树干）"
for TURN in "用户：我要退货" "用户：物流到哪了" "用户：发票怎么开"; do
    R=$(curl -sf -X POST "$BASE/checkpoint/generate" -H 'Content-Type: application/json' \
        -d "{\"checkpoint_id\":$CPID,\"text\":\" $TURN\",\"sampling_params\":{\"max_tokens\":24}}")
    echo "    分支 [$TURN] => $R"
done

echo ""
echo "==> [3/3] 算力账本（GET /stats）"
curl -sf "$BASE/stats" | grep -o '"compute_saved_tokens":[0-9.]*\|"prefill_skip_ratio":[0-9.]*\|"active_checkpoints":[0-9]*'
echo "    每个分支的树干部分零重算——agent 越多，省得越多。"

curl -sf -X POST "$BASE/checkpoint/delete" -H 'Content-Type: application/json' \
    -d "{\"checkpoint_id\":$CPID}" >/dev/null
echo ""
echo "演示结束（服务将自动停止）。"