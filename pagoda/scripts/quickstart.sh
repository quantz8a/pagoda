#!/usr/bin/env bash
# Pagoda 一键体验脚本（Linux/macOS）
# 用法: bash scripts/quickstart.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # 仓库 pagoda/ 根目录

echo "==> [1/4] 构建（离线，零依赖）"
cargo build --offline

echo ""
echo "==> [2/4] 运行测试套件（93 项）"
cargo test --offline

echo ""
echo "==> [3/4] 生成演示：同一 prompt 重复 3 次，观察前缀缓存省算力"
cargo run --offline --bin pagoda -- sample -p "Pagoda turns SGLang ideas into zero-dependency Rust" --max-tokens 32 --repeat 3

echo ""
echo "==> [4/4] DSL 演示：gen / select / fork"
cargo run --offline --bin pagoda -- program

cat <<'EOF'

全部完成！接下来可以：
  - 启动 HTTP 服务:   cargo run --offline --bin pagoda -- serve --port 8080
  - agent 集群演示:   bash scripts/demo-agent-cluster.sh
  - 小白文档:         docs/guide/README.md
EOF
