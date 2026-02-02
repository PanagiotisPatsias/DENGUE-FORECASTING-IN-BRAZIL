# Complete training and monitoring startup script
# Runs training, then starts drift scheduler (every 8 hours by default)

param(
    [string]$SchedulerInterval = "8h",
    [switch]$StartScheduler = $true
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DENGUE FORECASTING - COMPLETE PIPELINE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Train models
Write-Host "`nSTEP 1: Training models..." -ForegroundColor Yellow
python -m src.main

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n[OK] Training complete!" -ForegroundColor Green

# Step 2: Start drift scheduler (if requested)
if ($StartScheduler) {
    Write-Host "`nSTEP 2: Starting drift scheduler..." -ForegroundColor Yellow
    Write-Host "Interval: $SchedulerInterval" -ForegroundColor Gray
    Write-Host "Press Ctrl+C to stop scheduler`n" -ForegroundColor Gray
    
    python scripts\drift_scheduler.py --interval $SchedulerInterval
} else {
    Write-Host "`nSTEP 2: Drift scheduler NOT started" -ForegroundColor Yellow
    Write-Host "To start scheduler, run:" -ForegroundColor Gray
    Write-Host "  python scripts\drift_scheduler.py --interval $SchedulerInterval" -ForegroundColor White
    Write-Host "`nOr use this script with -StartScheduler flag:" -ForegroundColor Gray
    Write-Host "  .\scripts\run_training_with_monitoring.ps1 -StartScheduler -SchedulerInterval 'daily'" -ForegroundColor White
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "PIPELINE COMPLETE" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
