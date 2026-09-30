# Pagoda agent 集群演示（Windows PowerShell）
# 启动服务 → 创建 checkpoint（共享树干）→ 3 个 agent 分支 → 看省了多少算力
# 用法: powershell -ExecutionPolicy Bypass -File scripts\demo-agent-cluster.ps1 [-Port 8080]
param([int]$Port = 8080)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "==> 构建并启动 pagoda 服务于 127.0.0.1:$Port" -ForegroundColor Cyan
cargo build --offline | Out-Null
$proc = Start-Process -FilePath ".\target\debug\pagoda.exe" `
    -ArgumentList "serve --port $Port" -WindowStyle Hidden -PassThru
try {
    # 等待服务就绪
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        try {
            Invoke-RestMethod "http://127.0.0.1:$Port/health" | Out-Null
            $ready = $true; break
        } catch { Start-Sleep -Milliseconds 250 }
    }
    if (-not $ready) { throw "服务未能启动" }
    Write-Host "    服务已就绪" -ForegroundColor Green

    # 1. 创建 checkpoint：所有 agent 共享的系统树干
    $trunk = "你是客服团队的自主 agent。共享规则：礼貌、简洁、先查证再回答。工具说明：……"
    Write-Host "`n==> [1/3] 创建 checkpoint（共享树干：$($trunk.Length) 字符）" -ForegroundColor Cyan
    $cp = Invoke-RestMethod -Method Post "http://127.0.0.1:$Port/checkpoint" `
        -ContentType "application/json" -Body (@{ text = $trunk } | ConvertTo-Json -Compress)
    $cpid = $cp.checkpoint_id
    Write-Host "    checkpoint_id = $cpid"

    # 2. 三个 agent 分支，共用同一个树干
    Write-Host "`n==> [2/3] 启动 3 个 agent 分支（共享同一树干）" -ForegroundColor Cyan
    $turns = @("用户：我要退货", "用户：物流到哪了", "用户：发票怎么开")
    foreach ($t in $turns) {
        $r = Invoke-RestMethod -Method Post "http://127.0.0.1:$Port/checkpoint/generate" `
            -ContentType "application/json" `
            -Body (@{ checkpoint_id = $cpid; text = " $t"; sampling_params = @{ max_tokens = 24 } } | ConvertTo-Json -Compress)
        Write-Host ("    分支 [{0}]  prefix_hit={1}/{2}  forward={3}  => {4}" -f `
            $t, $r.prefix_hit_tokens, $r.prompt_tokens, $r.forward_count, $r.text.Trim())
    }

    # 3. 总账：省了多少算力
    Write-Host "`n==> [3/3] 算力账本（GET /stats）" -ForegroundColor Cyan
    $s = Invoke-RestMethod "http://127.0.0.1:$Port/stats"
    Write-Host ("    compute_saved_tokens = {0}   prefill_skip_ratio = {1:P0}" -f `
        $s.compute_saved_tokens, $s.prefill_skip_ratio)
    Write-Host "    每个分支的树干部分零重算——agent 越多，省得越多。" -ForegroundColor Green

    Invoke-RestMethod -Method Post "http://127.0.0.1:$Port/checkpoint/delete" `
        -ContentType "application/json" -Body (@{ checkpoint_id = $cpid } | ConvertTo-Json -Compress) | Out-Null
} finally {
    if ($proc -and -not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}
Write-Host "`n演示结束（服务已停止）。" -ForegroundColor Green