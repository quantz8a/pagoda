#!/usr/bin/env bash
# 一键蒸馏流水线：付费/私有教师 → 数据集 → LoRA 蒸馏 → 评估 → SGLang 部署学生
#
# 两种教师模式：
#   A. 付费/私有 API（推荐生产）： export TEACHER_BASE_URL=https://api.openai.com/v1
#                                   export TEACHER_API_KEY=sk-...
#                                   export TEACHER_MODEL=gpt-4o-mini
#      bash run-distill.sh
#   B. 本地教师（零成本演示）：     bash run-distill.sh --local-teacher
#      （用 SGLang 在本地起 Qwen2.5-3B-Instruct-AWQ 当教师，同一 OpenAI 兼容接口）
#
# 流程：  [1/6] 环境自检 → [2/6] 教师造数据 → [3/6] LoRA 蒸馏 →
#         [4/6] 合并权重 → [5/6] 逐字段评估 → [6/6] SGLang 部署学生 + 冒烟
# 幂等：中间产物存在即跳过，--force 强制重跑。
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
STUDENT_BASE=${STUDENT_BASE:-Qwen/Qwen2.5-0.5B-Instruct}
VARIATIONS=${VARIATIONS:-3}
EPOCHS=${EPOCHS:-3}
LOCAL_TEACHER=0
FORCE=0
TEACHER_PORT=${TEACHER_PORT:-30001}
STUDENT_PORT=${STUDENT_PORT:-30002}
for arg in "$@"; do
    case "$arg" in
        --local-teacher) LOCAL_TEACHER=1 ;;
        --force) FORCE=1 ;;
        *) echo "未知参数 $arg"; exit 2 ;;
    esac
done

if [ "$LOCAL_TEACHER" = "1" ]; then
    export TEACHER_BASE_URL=${TEACHER_BASE_URL:-http://127.0.0.1:$TEACHER_PORT/v1}
    export TEACHER_API_KEY=${TEACHER_API_KEY:-EMPTY}
    export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}
fi
: "${TEACHER_MODEL:?set TEACHER_MODEL (付费模型名或本地 repo id)}"
export TEACHER_BASE_URL=${TEACHER_BASE_URL:-http://127.0.0.1:$TEACHER_PORT/v1}
export TEACHER_API_KEY=${TEACHER_API_KEY:-EMPTY}

echo "==> [1/6] 环境自检（python: $PYTHON）"
$PYTHON - <<'EOF'
import importlib, sys
missing = [m for m in ("openai", "torch", "transformers", "peft")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("缺少依赖: %s —— pip install -r requirements.txt" % ", ".join(missing))
import torch
print("    torch", torch.__version__, "cuda:", torch.cuda.is_available())
EOF

TEACHER_PID=""
cleanup() { [ -n "$TEACHER_PID" ] && kill "$TEACHER_PID" 2>/dev/null || true; }
trap cleanup EXIT

if [ "$LOCAL_TEACHER" = "1" ]; then
    echo "==> [2/6] 启动本地教师（SGLang: $TEACHER_MODEL :$TEACHER_PORT）"
    nohup $PYTHON -m sglang.launch_server --model-path "$TEACHER_MODEL" \
        --port "$TEACHER_PORT" > teacher-server.log 2>&1 &
    TEACHER_PID=$!
    for i in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$TEACHER_PORT/health" >/dev/null 2>&1 && break
        kill -0 "$TEACHER_PID" 2>/dev/null || { echo "教师启动失败:"; tail -20 teacher-server.log; exit 1; }
        sleep 5
    done
    echo "    教师就绪"
else
    echo "==> [2/6] 使用外部教师: $TEACHER_BASE_URL (model=$TEACHER_MODEL)"
fi

if [ ! -s dataset.jsonl ] || [ "$FORCE" = "1" ]; then
    echo "    教师造数据（每条种子 $VARIATIONS 个变体）..."
    $PYTHON gen_data.py --variations "$VARIATIONS"
else
    echo "    dataset.jsonl 已存在，跳过（--force 重造）"
fi

if [ ! -d out/student-merged ] || [ "$FORCE" = "1" ]; then
    echo "==> [3/6] LoRA 蒸馏（$STUDENT_BASE, $EPOCHS epochs）"
    $PYTHON train_lora.py --base "$STUDENT_BASE" --epochs "$EPOCHS"
    echo "==> [4/6] 合并权重 -> out/student-merged（train_lora.py 已含）"
else
    echo "==> [3/6]+[4/6] out/student-merged 已存在，跳过（--force 重训）"
fi

if [ "$LOCAL_TEACHER" = "1" ]; then
    echo "    关闭本地教师，释放显存"
    cleanup; TEACHER_PID=""
    sleep 5
fi

if [ ! -s out/eval_report.json ] || [ "$FORCE" = "1" ]; then
    echo "==> [5/6] 逐字段评估（base vs distilled vs 教师答案）"
    $PYTHON eval.py --base "$STUDENT_BASE" --merged out/student-merged \
        --report out/eval_report.json
else
    echo "==> [5/6] out/eval_report.json 已存在，跳过"
fi

echo "==> [6/6] SGLang 部署学生（:$STUDENT_PORT）+ 冒烟"
nohup $PYTHON -m sglang.launch_server --model-path out/student-merged \
    --port "$STUDENT_PORT" > student-server.log 2>&1 &
STUDENT_PID=$!
for i in $(seq 1 90); do
    curl -sf "http://127.0.0.1:$STUDENT_PORT/health" >/dev/null 2>&1 && break
    kill -0 "$STUDENT_PID" 2>/dev/null || { echo "学生部署失败:"; tail -20 student-server.log; exit 1; }
    sleep 5
done
echo "    学生服务就绪: http://127.0.0.1:$STUDENT_PORT/v1 (OpenAI 兼容)"
SMOKE=$(curl -sf -X POST "http://127.0.0.1:$STUDENT_PORT/v1/chat/completions" \
    -H "Content-Type: application/json" -d '{
      "model": "out/student-merged",
      "messages": [{"role":"user","content":"你是电商客服工单结构化助手。阅读客户消息，提取信息并只输出 JSON：{\"department\": \"billing|shipping|technical|product|other\", \"urgency\": 0|1|2, \"sentiment\": \"angry|neutral|positive\", \"order_id\": \"订单号或null\", \"refund_amount\": \"数字或null\", \"needs_human\": true|false, \"reply\": \"不超过80字的中文回复草稿\"}\n客户消息：\n我3月被重复扣了两次会员费，订单ORD-99821，多扣了59元，请马上退款，否则我找银行拒付！"}],
      "temperature": 0, "max_tokens": 320}')
echo "$SMOKE" | $PYTHON -c "import json,sys; print('    冒烟输出:', json.load(sys.stdin)['choices'][0]['message']['content'][:200])"
echo
echo "蒸馏完成。学生服务 PID=$STUDENT_PID，停止： kill $STUDENT_PID"
echo "报告: out/eval_report.json ；数据: dataset.jsonl ；模型: out/student-merged/"