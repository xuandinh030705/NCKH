# math-tutor-cli - Script khoi dong nhanh chatbot tren Windows
param(
    [string]$Mode,
    [string]$Model,
    [string]$Once,
    [switch]$Mock,
    [switch]$Stream,
    [switch]$VerboseLog,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# UTF-8 cho console Windows
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "[CANH BAO] Chua tim thay .venv\Scripts\python.exe." -ForegroundColor Yellow
    Write-Host "Dang thu dung python tu he thong..." -ForegroundColor Yellow
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $venvPy = "python"
    } else {
        Write-Host "[LOI] Khong tim thay Python. Vui long chay setup.ps1 truoc!" -ForegroundColor Red
        exit 1
    }
}

# Kiem tra file model neu khong chay --mock
if (-not $Mock) {
    $defaultModel = Join-Path $Root "models\Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    $hasModel = $false
    if ($Model -and (Test-Path $Model)) {
        $hasModel = $true
    } elseif (Test-Path $defaultModel) {
        $hasModel = $true
    } else {
        $ggufFiles = Get-ChildItem -Path (Join-Path $Root "models") -Filter "*.gguf" -ErrorAction SilentlyContinue
        if ($ggufFiles -and $ggufFiles.Count -gt 0) {
            $hasModel = $true
        }
    }

    if (-not $hasModel) {
        Write-Host "==========================================================" -ForegroundColor Yellow
        Write-Host "[THONG BAO] Chua tim thay file model GGUF trong thu muc models/!" -ForegroundColor Yellow
        Write-Host "De chay duoc chatbot, ban can phai tai model truoc." -ForegroundColor Yellow
        Write-Host "Model mac dinh: Qwen3-4B-Instruct-2507-Q4_K_M.gguf (~2.3 GB)" -ForegroundColor Cyan
        Write-Host "==========================================================" -ForegroundColor Yellow
        $choice = Read-Host "Ban co muon tu dong tai model ngay bay gio khong? (Y/n)"
        if ($choice -eq "" -or $choice -match "^[yY]") {
            Write-Host "Dang tai model..." -ForegroundColor Green
            & $venvPy "scripts\download_model.py"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "[LOI] Tai model that bai. Vui long kiem tra mang hoac tai thu cong theo models/README.md" -ForegroundColor Red
                exit 1
            }
        } else {
            Write-Host "Huy bo. Ban co the tu tai sau bang lenh: python scripts\download_model.py" -ForegroundColor Yellow
            Write-Host "Hoac chay thu nghiem gia lap (khong can model): .\run.ps1 -Mock" -ForegroundColor Yellow
            exit 0
        }
    }
}

$cliArgs = @("-m", "tutor")
if ($Mode) { $cliArgs += @("--mode", $Mode) }
if ($Model) { $cliArgs += @("--model", $Model) }
if ($Once) { $cliArgs += @("--once", $Once) }
if ($Mock) { $cliArgs += "--mock" }
if ($Stream) { $cliArgs += "--stream" }
if ($VerboseLog) { $cliArgs += "--verbose" }
if ($RemainingArgs) { $cliArgs += $RemainingArgs }

& $venvPy @cliArgs
