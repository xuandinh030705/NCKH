# run_ui.ps1 - Khoi dong Frontend Gia su Toan Tieu hoc AI
param(
    [switch]$Mock,
    [switch]$Share,
    [int]$Port = 7860,
    [string]$Config
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

# Tim Python trong .venv cua thu muc cha hoac thu muc hien tai
$pyCandidates = @(
    (Join-Path $ScriptDir "..\.venv\Scripts\python.exe"),
    (Join-Path $ScriptDir ".venv\Scripts\python.exe")
)

$py = $null
foreach ($c in $pyCandidates) {
    if (Test-Path $c) {
        $py = (Resolve-Path $c).Path
        break
    }
}

if (-not $py) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $py = "python"
    } else {
        Write-Host "[LOI] Khong tim thay Python! Vui long kiem tra moi truong." -ForegroundColor Red
        exit 1
    }
}

$pyArgs = @("app.py")
if ($Mock)   { $pyArgs += "--mock" }
if ($Share)  { $pyArgs += "--share" }
if ($Port)   { $pyArgs += "--port"; $pyArgs += $Port }
if ($Config) { $pyArgs += "--config"; $pyArgs += $Config }

Write-Host "===========================================================" -ForegroundColor Green
Write-Host "  GIA SU TOAN TIEU HOC (AI SOCRATIC) - FRONTEND" -ForegroundColor Green
Write-Host "  Giao dien : Gradio Sharp Minimal + Lich su Chat" -ForegroundColor Cyan
Write-Host "  URL       : http://localhost:$Port" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Green
Write-Host ""

& $py @pyArgs

