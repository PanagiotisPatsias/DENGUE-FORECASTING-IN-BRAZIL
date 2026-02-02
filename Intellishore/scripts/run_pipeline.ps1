# dengue forecasting pipeline execution script (PowerShell)
# runs the complete forecasting pipeline from data loading to visualization

$ErrorActionPreference = "Stop"

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Dengue Forecasting Pipeline" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# check if python is available
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "Using: $pythonCmd"
Write-Host ""

# check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & "venv\Scripts\Activate.ps1"
} elseif (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..."
    & ".venv\Scripts\Activate.ps1"
} else {
    Write-Host "No virtual environment found (optional)"
}

Write-Host ""

# check if required packages are installed
Write-Host "Checking dependencies..."
$checkDeps = & $pythonCmd -c "import pandas, numpy, sklearn, matplotlib" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    & $pythonCmd -m pip install -r requirements.txt
} else {
    Write-Host "All dependencies satisfied " -NoNewline
    Write-Host "✓" -ForegroundColor Green
}

Write-Host ""
Write-Host "Starting pipeline execution..." -ForegroundColor Yellow
Write-Host ""

# run the pipeline
& $pythonCmd -m src.main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Green
    Write-Host "Pipeline execution completed ✓" -ForegroundColor Green
    Write-Host "==================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "==================================" -ForegroundColor Red
    Write-Host "Pipeline execution failed ✗" -ForegroundColor Red
    Write-Host "==================================" -ForegroundColor Red
    exit 1
}
