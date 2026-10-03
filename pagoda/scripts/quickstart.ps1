# Pagoda 一键体验脚本（Windows PowerShell）
# 用法: 在 pagoda 目录下执行  powershell -ExecutionPolicy Bypass -File scripts\quickstart.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # 仓库 pagoda/ 根目录

Write-Host "==> [1/4] 构建（离线，零依赖）" -ForegroundColor Cyan
cargo build --offline

Write-Host "`n==> [2/4] 运行测试套件（93 项）" -ForegroundColor Cyan
cargo test --offline

Write-Host "`n==> [3/4] 生成演示：同一 prompt 重复 3 次，观察前缀缓存省算力" -ForegroundColor Cyan
cargo run --offline --bin pagoda -- sample -p "Pagoda turns SGLang ideas into zero-dependency Rust" --max-tokens 32 --repeat 3

Write-Host "`n==> [4/4] DSL 演示：gen / select / fork" -ForegroundColor Cyan
cargo run --offline --bin pagoda -- program

Write-Host @"

全部完成！接下来可以：
  - 启动 HTTP 服务:   cargo run --offline --bin pagoda -- serve --port 8080
  - agent 集群演示:   scripts\demo-agent-cluster.ps1
  - 小白文档:         docs\guide\README.md
"@ -ForegroundColor Green
