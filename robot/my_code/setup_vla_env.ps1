param(
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvDir = Join-Path $ProjectRoot ".venv_vla"
$ReqFile = Join-Path $ScriptDir "requirements_vla_realsense.txt"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "Project root: $ProjectRoot"
Write-Host "Venv: $VenvDir"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating virtual environment..."
    try {
        & $PythonLauncher "-$PythonVersion" -m venv $VenvDir
    }
    catch {
        Write-Host "Fallback: creating venv with default python..."
        python -m venv $VenvDir
    }
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

Write-Host "Installing xArm SDK from local project..."
& $VenvPython -m pip install $ProjectRoot

Write-Host "Installing RealSense/VLA dependencies..."
& $VenvPython -m pip install -r $ReqFile

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Run integrated collection with:"
Write-Host "& `"$VenvPython`" `"$ProjectRoot\my_code\run_scan_collect_vla_realsense.py`""
