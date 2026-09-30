#!/usr/bin/env bash
# Pagoda P1 一键联网验证（Linux/macOS）
# 真实 HF tokenizer + Candle 权重驱动 pagoda 引擎，全自动：拉依赖 → 编译 → 跑断言
# 用法:
#   bash scripts/verify-p1.sh           # 直连 HuggingFace
#   bash scripts/verify-p1.sh --mirror  # 国内走 hf-mirror.com
set -o pipefail
cd "$(dirname "$0")/.."   # pagoda-hf/

if [ "${1:-}" = "--mirror" ]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "==> 使用镜像 HF_ENDPOINT=$HF_ENDPOINT"
fi

LOG=verify-p1.log
echo "=== P1 verification $(date -Iseconds) ===" | tee $LOG

echo "==> [1/3] cargo fetch（拉取依赖）"
cargo fetch 2>&1 | tee -a $LOG || exit 1

echo "==> [2/3] cargo build --release --example e2e_tiny_llama"
cargo build --release --example e2e_tiny_llama 2>&1 | tee -a $LOG || exit 1

echo "==> [3/3] 运行端到端验证（下载 tiny-random-Llama 权重）"
cargo run --release --example e2e_tiny_llama 2>&1 | tee -a $LOG || exit 1

echo ""
echo "P1 VERIFICATION OK — 结果已记录到 $LOG"