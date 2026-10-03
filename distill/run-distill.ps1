# 一键蒸馏流水线（Windows PowerShell）
#
# 教师模式：
#   A. 付费/私有 API： $env:TEACHER_BASE_URL="https://api.openai.com/v1"
#                       $env:TEACHER_API_KEY="sk-..."
#                       $env:TEACHER_MODEL="gpt-4o-mini"
#      powershell -File run-distill.ps1
#   B. 本地教师（零成本演示）：powershell -File run-distill.ps1 -LocalTeacher
#      （SGLang 起 Qwen2.5-3B-Instruct-AWQ 当教师，同一 OpenAI 兼容接口）
param(
    [switch]$LocalTeacher,
    [switch]$Force,
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$StudentBase = if ($env:STUDENT_BASE) { $env:STUDENT_BASE } else { "Qwen/Qwen2.5-0.5B-Instruct" }
$TeacherPort = 30001; $StudentPort = 30002
if ($LocalTeacher) {
    if (-not $env:TEACHER_BASE_URL) { $env:TEACHER_BASE_URL = "http://127.0.0.1:$TeacherPort/v1" }
    if (-not $env:TEACHER_API_KEY)  { $env:TEACHER_API_KEY = "EMPTY" }
    if (-not $env:TEACHER_MODEL)    { $env:TEACHER_MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ" }
}
if (-not $env:TEACHER_MODEL) { throw "设置 TEACHER_MODEL 环境变量（付费模型名或本地 repo id）" }
if (-not $env:TEACHER_BASE_URL) { $env:TEACHER_BASE_URL = "http://127.0.0.1:$TeacherPort/v1" }
if (-not $env:TEACHER_API_KEY) { $env:TEACHER_API_KEY = "EMPTY" }

function Wait-Health($url, $name, $proc, $timeoutMin) {
    $deadline = (Get-Date).AddMinutes($timeoutMin)
    while ((Get-Date) -lt $deadline) {
        try { $null = Invoke-RestMethod "$url/health" -TimeoutSec 3; return } catch {}
        if ($proc.HasExited) { throw "$name 启动失败，见日志" }
        Start-Sleep 5
    }
    throw "$name ${timeoutMin} 分钟未就绪"
}

Write-Host "==> [1/6] 环境自检" -ForegroundColor Cyan
& $Python -c "import openai, torch, transformers, peft; print('    torch', torch.__version__, 'cuda:', torch.cuda.is_available())"

$teacherProc = $null
if ($LocalTeacher) {
    Write-Host "==> [2/6] 启动本地教师（SGLang: $env:TEACHER_MODEL）" -ForegroundColor Cyan
    $teacherProc = Start-Process -PassThru -WindowStyle Hidden -FilePath $Python `
        -ArgumentList "-m sglang.launch_server --model-path $env:TEACHER_MODEL --port $TeacherPort" `
        -RedirectStandardOutput teacher-server.log -RedirectStandardError teacher-server.err.log
    Wait-Health "http://127.0.0.1:$TeacherPort" "教师" $teacherProc 15
} else {
    Write-Host "==> [2/6] 使用外部教师: $env:TEACHER_BASE_URL ($env:TEACHER_MODEL)" -ForegroundColor Cyan
}

if ((-not (Test-Path dataset.jsonl)) -or $Force) {
    Write-Host "    教师造数据..." -ForegroundColor Cyan
    & $Python gen_data.py
} else { Write-Host "    dataset.jsonl 已存在，跳过（-Force 重造）" }

if ($teacherProc) { $teacherProc.Kill(); Start-Sleep 5 }

if ((-not (Test-Path out\student-merged)) -or $Force) {
    Write-Host "==> [3/6]+[4/6] LoRA 蒸馏 + 合并（$StudentBase）" -ForegroundColor Cyan
    & $Python train_lora.py --base $StudentBase
} else { Write-Host "==> [3/6]+[4/6] 已存在，跳过" }

if ((-not (Test-Path out\eval_report.json)) -or $Force) {
    Write-Host "==> [5/6] 逐字段评估" -ForegroundColor Cyan
    & $Python eval.py --base $StudentBase --merged out\student-merged --report out\eval_report.json
} else { Write-Host "==> [5/6] 已存在，跳过" }

Write-Host "==> [6/6] SGLang 部署学生（:$StudentPort）" -ForegroundColor Cyan
$studentProc = Start-Process -PassThru -WindowStyle Hidden -FilePath $Python `
    -ArgumentList "-m sglang.launch_server --model-path out\student-merged --port $StudentPort" `
    -RedirectStandardOutput student-server.log -RedirectStandardError student-server.err.log
Wait-Health "http://127.0.0.1:$StudentPort" "学生" $studentProc 10
Write-Host "    学生服务就绪: http://127.0.0.1:$StudentPort/v1 (OpenAI 兼容)，PID=$($studentProc.Id)" -ForegroundColor Green
Write-Host "    停止命令： Stop-Process -Id $($studentProc.Id)"
