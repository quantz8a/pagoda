# Pagoda × SGLang 一键共部署（Windows PowerShell）
#
# 架构：pagoda 当控制面网关（:30000），SGLang 当 GPU worker（:30001）。
#   客户端 → pagoda  /health /stats /checkpoint* 由 pagoda 本地服务
#                     /generate /v1/chat/completions 透明转发给 SGLang
#
# 用法：在 pagoda 目录下执行
#   powershell -ExecutionPolicy Bypass -File scripts\co-deploy.ps1 [-Model <hf-repo>] [-InstallSglang]
param(
    [string]$Model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    [int]$PagodaPort = 30000,
    [int]$SglangPort = 30001,
    [switch]$InstallSglang   # 首次运行加上：自动建 venv 并 pip install sglang
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # 仓库 pagoda/ 根目录

function Wait-Health($url, $name, $timeoutSec) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $null = Invoke-RestMethod -Uri "$url/health" -TimeoutSec 2
            Write-Host "    $name 健康检查通过: $url" -ForegroundColor Green
            return
        } catch { Start-Sleep -Seconds 2 }
    }
    throw "$name 在 ${timeoutSec}s 内未就绪（$url），日志见上"
}

Write-Host "==> [1/4] 构建 pagoda（release）" -ForegroundColor Cyan
cargo build --release

Write-Host "`n==> [2/4] 准备 SGLang（python）" -ForegroundColor Cyan
$py = "python"
if ($InstallSglang) {
    if (-not (Test-Path ".venv-sglang")) { python -m venv .venv-sglang }
    $py = ".venv-sglang\Scripts\python.exe"
    & $py -m pip install --upgrade pip
    & $py -m pip install "sglang[all]"
}
& $py -c "import sglang" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "未检测到 sglang。重跑时加 -InstallSglang，或自行 pip install `"sglang[all]`""
}

Write-Host "`n==> [3/4] 启动 SGLang worker（模型 $Model，端口 $SglangPort）" -ForegroundColor Cyan
$sglang = Start-Process -PassThru -WindowStyle Hidden -FilePath $py `
    -ArgumentList "-m sglang.launch_server --model-path $Model --port $SglangPort" `
    -RedirectStandardOutput "sglang-worker.log" -RedirectStandardError "sglang-worker.err.log"
Write-Host "    SGLang PID=$($sglang.Id)，日志 sglang-worker.log"
Wait-Health "http://127.0.0.1:$SglangPort" "SGLang" 600   # 首次要下载模型+编译 kernel，给足时间

Write-Host "`n==> [4/4] 启动 pagoda 网关（端口 $PagodaPort → 上游 $SglangPort）" -ForegroundColor Cyan
$pagoda = Start-Process -PassThru -WindowStyle Hidden -FilePath ".\target\release\pagoda.exe" `
    -ArgumentList "serve --port $PagodaPort --upstream http://127.0.0.1:$SglangPort" `
    -RedirectStandardOutput "pagoda-gateway.log" -RedirectStandardError "pagoda-gateway.err.log"
Write-Host "    pagoda PID=$($pagoda.Id)，日志 pagoda-gateway.log"
Wait-Health "http://127.0.0.1:$PagodaPort" "pagoda" 30

Write-Host @"

共部署完成！
  对外入口（客户端只认这个）:  http://127.0.0.1:$PagodaPort
    POST /generate               → 转发 SGLang
    POST /v1/chat/completions    → 转发 SGLang
    GET  /health  /stats         → pagoda 本地（含 upstream / proxied_requests）
    POST /checkpoint*            → pagoda 本地（agent 原语）
  冒烟测试:
    curl -X POST http://127.0.0.1:$PagodaPort/generate -H "Content-Type: application/json" -d '{\"text\":\"The capital of France is\",\"sampling_params\":{\"max_new_tokens\":16}}'
  停止:
    Stop-Process -Id $($pagoda.Id),$($sglang.Id)
"@ -ForegroundColor Green