# math-tutor-cli - one-shot setup for Windows (PowerShell 5.1+)
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Cpu
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1 -ModelRepo unsloth/Qwen3-1.7B-GGUF -ModelFile Qwen3-1.7B-Instruct-Q4_K_M.gguf
#
# ASCII-only on purpose: Windows PowerShell 5.1 misreads UTF-8 without BOM.

[CmdletBinding()]
param(
    [switch]$Cpu,
    [switch]$SkipModel,
    [switch]$SkipTests,
    [string]$ModelRepo = "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    [string]$ModelFile = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"

function Info([string]$m) { Write-Host "[setup] $m" -ForegroundColor Cyan }
function Warn([string]$m) { Write-Host "[canh bao] $m" -ForegroundColor Yellow }
function Fail([string]$m) { Write-Host "[loi] $m" -ForegroundColor Red; exit 1 }

Write-Host "=== math-tutor-cli setup ===" -ForegroundColor Green
Info "Thu muc: $Root"

# ---------------------------------------------------------------- 1. Python
$pyCmd = $null
$pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.12", "3.11", "3.10")) {
        & py "-$v" -c "import sys" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $pyCmd = "py"; $pyArgs = @("-$v"); break }
    }
}
if (-not $pyCmd) {
    if (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3,10) else 1)" 2>$null
        if ($LASTEXITCODE -eq 0) { $pyCmd = "python"; $pyArgs = @() }
    }
}
if (-not $pyCmd) {
    Fail "Khong tim thay Python 3.10+. Cai tai https://www.python.org/downloads/ (nho tick 'Add python.exe to PATH')."
}

# ---------------------------------------------------------------- 2. venv
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Info "Tao moi truong ao .venv ..."
    & $pyCmd @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail "Tao venv that bai." }
} else {
    Info "Da co .venv, dung lai."
}
if (-not (Test-Path $venvPy)) { Fail "Khong thay $venvPy" }

Info "Nang pip ..."
& $venvPy -m pip install --upgrade pip --quiet
if ($LASTEXITCODE -ne 0) { Warn "Nang pip that bai, tiep tuc voi pip hien co." }

# ---------------------------------------------------------------- 3. llama-cpp
$cudaWheel = "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36-cu124/llama_cpp_python-0.3.36-py3-none-win_amd64.whl"
if (-not $Cpu) {
    Info "Cai llama-cpp-python ban CUDA 12.4 (Windows x64) ..."
    & $venvPy -m pip install $cudaWheel
    if ($LASTEXITCODE -ne 0) {
        Warn "Khong cai duoc wheel CUDA (co the may khong co GPU NVIDIA). Chuyen sang ban CPU."
        $Cpu = $true
    } else {
        Info "Cai CUDA runtime + cuBLAS ..."
        & $venvPy -m pip install "nvidia-cuda-runtime-cu12==12.4.127" "nvidia-cublas-cu12==12.4.5.8"
        if ($LASTEXITCODE -ne 0) { Warn "Cai CUDA runtime loi; neu chay GPU bi 'DLL load failed' thi cai lai 2 goi nay." }
    }
}
if ($Cpu) {
    Info "Cai llama-cpp-python ban CPU ..."
    & $venvPy -m pip install "llama-cpp-python==0.3.36"
    if ($LASTEXITCODE -ne 0) { Fail "Cai llama-cpp-python that bai. Xem loi phia tren." }
}

Info "Cai cac goi con lai ..."
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "Cai requirements.txt that bai." }

# ---------------------------------------------------------------- 4. model
if (-not $SkipModel) {
    Info "Tai model $ModelRepo / $ModelFile (co the mat vai phut) ..."
    & $venvPy "scripts\download_model.py" --repo $ModelRepo --file $ModelFile
    if ($LASTEXITCODE -ne 0) {
        Warn "Tai model that bai. Thu lai sau, hoac tai thu cong theo models/README.md."
        Warn "Sau khi co file, chay: .\run.ps1 -Model models\$ModelFile"
    }
} else {
    Info "Bo qua tai model (-SkipModel)."
}

# ---------------------------------------------------------------- 5. tests
if (-not $SkipTests) {
    Info "Cai pytest va chay kiem thu ..."
    & $venvPy -m pip install pytest --quiet
    & $venvPy -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        Warn "Mot so test khong dat. Chatbot van co the chay, nhung nen gui loi nay cho nguoi lam project."
    }
}

# ---------------------------------------------------------------- 6. done
Write-Host ""
Write-Host "=== XONG ===" -ForegroundColor Green
if ($ModelFile -eq "Qwen3-4B-Instruct-2507-Q4_K_M.gguf") {
    Write-Host "Chay chatbot:  .\run.ps1" -ForegroundColor Green
    Write-Host "Chay 1 cau :  .\run.ps1 -Once ""Tim x, biet: x + 18 = 45"" -Mode 3" -ForegroundColor Green
} else {
    Write-Host "Chay chatbot:  .\run.ps1 -Model models\$ModelFile" -ForegroundColor Green
}
Write-Host "Khong can GPU de thu:  .\run.ps1 -Mock"
