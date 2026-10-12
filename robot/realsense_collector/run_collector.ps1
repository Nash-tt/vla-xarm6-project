param(
    [string]$Instruction = "",
    [string]$Output = ".\dataset",
    [double]$Duration = 0
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

# Keep compatibility with the development workspace used to build this package.
if (-not (Test-Path $Python)) {
    $Python = Join-Path $ProjectRoot "..\..\work\.venv\Scripts\python.exe"
}

if (-not (Test-Path $Python)) {
    throw "Python environment not found. Run .\setup_windows.ps1 first."
}

& $Python (Join-Path $ProjectRoot "collector.py") `
    --output $Output `
    --instruction $Instruction `
    --duration $Duration
