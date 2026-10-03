param(
  [string]$BaseUrl = "https://api.openai.com/v1",
  [string]$ApiKey = "",
  [string]$Model = "gpt-4o-mini",
  [int]$Port = 8642
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:TEACHER_BASE_URL = $BaseUrl
$env:TEACHER_API_KEY   = $ApiKey
$env:TEACHER_MODEL     = $Model

Write-Host "[1/3] labeling with teacher (paid/private LLM) ..."
python teacher.py
Write-Host "[2/3] distilling a tiny non-autoregressive head ..."
python train.py
Write-Host "[3/3] starting local service on http://127.0.0.1:$Port (Ctrl+C to stop) ..."
python serve.py --port $Port