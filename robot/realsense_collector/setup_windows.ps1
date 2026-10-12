$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host "[1/3] Checking Python..."
$BasePython = Get-Command python -ErrorAction SilentlyContinue
if (-not $BasePython) {
    throw "Python was not found. Install 64-bit Python 3.11, enable 'Add Python to PATH', then rerun this script."
}

$Version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($Version -notin @("3.11", "3.12")) {
    throw "Python 3.11 or 3.12 (64-bit) is required; current version is $Version."
}

$Architecture = & python -c "import struct; print(struct.calcsize('P') * 8)"
if ($Architecture -ne "64") {
    throw "64-bit Python is required; current Python is $Architecture-bit."
}

Write-Host "[2/3] Creating isolated environment..."
if (-not (Test-Path $Python)) {
    & python -m venv $Venv
}

Write-Host "[3/3] Installing RealSense dependencies..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")

Write-Host ""
Write-Host "Installation complete. Checking connected cameras..."
& $Python (Join-Path $ProjectRoot "collector.py") --list-devices
Write-Host ""
Write-Host "Start collection with:"
Write-Host '.\run_collector.ps1 -Instruction "把红色方块放入盒子" -Output ".\dataset"'
