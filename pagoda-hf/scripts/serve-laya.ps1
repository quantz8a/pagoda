# Laya 决策服务一键部署（Windows PowerShell）
#
# 单二进制 Rust 版 convaiinnovations/laya：无需 Python / PyTorch / CUDA 工具链，
# CPU 即可运行（有 NVIDIA 显卡时设 $env:PAGODA_DEVICE="cuda" 自动加速）。
#
# 用法：在 pagoda-hf 目录下执行
#   powershell -ExecutionPolicy Bypass -File scripts\serve-laya.ps1 [-Port 8081] [-Smoke]
param(
    [int]$Port = 8081,
    [string]$Repo = "convaiinnovations/laya",
    [switch]$Smoke   # 起服务后自动打一发账单场景冒烟（断言路由到 billing）
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # 仓库 pagoda-hf/ 根目录

Write-Host "==> [1/3] 构建 laya_server（release）" -ForegroundColor Cyan
cargo build --release --bin laya_server

Write-Host "`n==> [2/3] 启动服务（端口 $Port，模型 $Repo）" -ForegroundColor Cyan
Write-Host "    首次运行会从 Hugging Face 下载约 842MB 权重，之后走本地缓存秒开。"
$server = Start-Process -PassThru -WindowStyle Hidden -FilePath ".\target\release\laya_server.exe" `
    -ArgumentList "--port","$Port","--repo","$Repo" `
    -RedirectStandardOutput "laya-server.log" -RedirectStandardError "laya-server.err.log"
Write-Host "    laya_server PID=$($server.Id)，日志 laya-server.log / laya-server.err.log"

Write-Host "`n==> [3/3] 等待健康检查" -ForegroundColor Cyan
$deadline = (Get-Date).AddMinutes(10)  # 首次含模型下载
$ok = $false
while ((Get-Date) -lt $deadline) {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        $ok = $true; break
    } catch { Start-Sleep -Seconds 3 }
}
if (-not $ok) { throw "服务 10 分钟内未就绪，日志见 laya-server.err.log" }
Write-Host "    健康检查通过: http://127.0.0.1:$Port" -ForegroundColor Green

if ($Smoke) {
    Write-Host "`n==> 冒烟：README 账单场景（应路由到 billing）" -ForegroundColor Cyan
    $body = '{"state":"Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.","questions":{"department":{"type":"choice","instructions":"Which department should handle this?","criteria":{"billing":"invoices, payments, refunds","technical":"bugs, outages, system errors","other":"everything else"}},"churn_risk":{"type":"noul","instructions":"Does the user threaten to cancel or leave?"}}}'
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/decide" -Method Post -ContentType "application/json" -Body $body
    $dept = $resp.answers.department.choice
    $churn = $resp.answers.churn_risk.noul
    Write-Host ($resp | ConvertTo-Json -Depth 8)
    if ($dept -ne "billing") { throw "冒烟失败：department=$dept，预期 billing" }
    Write-Host "    冒烟通过：department=billing, churn_risk=$churn" -ForegroundColor Green
}

Write-Host @"

部署完成。Jev 兼容调用：
  POST http://127.0.0.1:$Port/decide
  {"state": "<文本>", "questions": {"id": {"type": "choice|score|noul", ...}}}
停止服务： Stop-Process -Id $($server.Id)
"@ -ForegroundColor Cyan