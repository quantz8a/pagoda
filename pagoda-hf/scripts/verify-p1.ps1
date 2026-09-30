# Pagoda P1 一键联网验证（Windows PowerShell）
# 真实 HF tokenizer + Candle 权重驱动 pagoda 引擎，全自动：拉依赖 → 编译 → 跑断言
# 用法:
#   powershell -ExecutionPolicy Bypass -File scripts\verify-p1.ps1           # 直连 HuggingFace
#   powershell -ExecutionPolicy Bypass -File scripts\verify-p1.ps1 -Mirror   # 国内走 hf-mirror.com
param([switch]$Mirror)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # pagoda-hf/

if ($Mirror) {
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    Write-Host "==> 使用镜像 HF_ENDPOINT=$env:HF_ENDPOINT" -ForegroundColor Yellow
}

$log = "verify-p1.log"
"=== P1 verification $(Get-Date -Format o) ===" | Tee-Object $log

# cargo 把正常进度写到 stderr；PowerShell 的 2>&1 会把它误报成 NativeCommandError。
# 经 cmd /c 合并为纯 stdout 再进管道，既保留日志又不误杀。
function Invoke-Step([string]$name, [string]$cmd) {
    Write-Host "==> $name" -ForegroundColor Cyan
    cmd /c "$cmd 2>&1" | Tee-Object -Append $log
    if ($LASTEXITCODE -ne 0) { Write-Host "`n失败：$cmd（详见 $log）" -ForegroundColor Red; exit 1 }
}

Invoke-Step "[1/3] cargo fetch（拉取依赖）" "cargo fetch"
Invoke-Step "[2/3] cargo build --release --example e2e_tiny_llama（编译，首次较久）" "cargo build --release --example e2e_tiny_llama"
Invoke-Step "[3/3] 运行端到端验证（下载 tiny 模型并跑断言）" "cargo run --release --example e2e_tiny_llama"

Write-Host "`nP1 VERIFICATION OK — 结果已记录到 $log" -ForegroundColor Green