#!/usr/bin/env bash
# Laya 决策服务一键部署（Linux/macOS）
#
# 单二进制 Rust 版 convaiinnovations/laya：无需 Python / PyTorch / CUDA 工具链，
# CPU 即可运行（有 NVIDIA 显卡时 PAGODA_DEVICE=cuda 自动加速，需先以 --features cuda 构建）。
#
# 用法：在 pagoda-hf 目录下执行
#   bash scripts/serve-laya.sh [--port 8081] [--smoke]
#
# 可选环境变量 PAGODA_CUBLAS_LIBDIR：指向与驱动同代的 cuBLAS 所在目录。
# candle 运行时按 libcublas.so.12 动态加载，若系统默认路径里是更新代的库
# （如 12.8）而驱动较旧（如 535 / CUDA 12.2），GEMM 会报
# CUBLAS_STATUS_INVALID_VALUE；用该变量把配套版本前置即可。
set -euo pipefail
cd "$(dirname "$0")/.."   # 仓库 pagoda-hf/ 根目录

if [ -n "${PAGODA_CUBLAS_LIBDIR:-}" ]; then
    export LD_LIBRARY_PATH="$PAGODA_CUBLAS_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi   # 仓库 pagoda-hf/ 根目录

PORT=8081
REPO="convaiinnovations/laya"
SMOKE=0
FEATURES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --cuda) FEATURES="--features cuda"; shift ;;
        --smoke) SMOKE=1; shift ;;
        *) echo "未知参数 $1"; exit 2 ;;
    esac
done

echo "==> [1/3] 构建 laya_server（release $FEATURES）"
cargo build --release $FEATURES --bin laya_server

echo
echo "==> [2/3] 启动服务（端口 $PORT，模型 $REPO）"
echo "    首次运行会从 Hugging Face 下载约 842MB 权重，之后走本地缓存秒开。"
./target/release/laya_server --port "$PORT" --repo "$REPO" > laya-server.log 2> laya-server.err.log &
SERVER_PID=$!
echo "    laya_server PID=$SERVER_PID，日志 laya-server.log / laya-server.err.log"

echo
echo "==> [3/3] 等待健康检查"
for i in $(seq 1 200); do   # 最多 10 分钟（首次含模型下载）
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then
        echo "    健康检查通过: http://127.0.0.1:$PORT"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "服务进程退出，日志：" >&2; cat laya-server.err.log >&2; exit 1
    fi
    sleep 3
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "服务未就绪" >&2; exit 1; }

if [ "$SMOKE" = "1" ]; then
    echo
    echo "==> 冒烟：README 账单场景（应路由到 billing）"
    RESP=$(curl -sf -X POST "http://127.0.0.1:$PORT/decide" -H "Content-Type: application/json" -d '{"state":"Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.","questions":{"department":{"type":"choice","instructions":"Which department should handle this?","criteria":{"billing":"invoices, payments, refunds","technical":"bugs, outages, system errors","other":"everything else"}},"churn_risk":{"type":"noul","instructions":"Does the user threaten to cancel or leave?"}}}')
    echo "$RESP"
    echo "$RESP" | grep -q '"choice":"billing"' || { echo "冒烟失败：未路由到 billing" >&2; exit 1; }
    echo "    冒烟通过：department=billing"
fi

cat <<EOF

部署完成。Jev 兼容调用：
  POST http://127.0.0.1:$PORT/decide
  {"state": "<文本>", "questions": {"id": {"type": "choice|score|noul", ...}}}
停止服务： kill $SERVER_PID
EOF