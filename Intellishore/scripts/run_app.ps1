# launch mlflow ui and streamlit app together

Write-Host "`nStarting Dengue Forecasting System..." -ForegroundColor Green
Write-Host ""

# check if model exists
if (-not (Test-Path "models\baseline_model.pkl")) {
    Write-Host "ERROR: No baseline model found!" -ForegroundColor Red
    Write-Host "   Please run training first: python -m src.main"
    exit 1
}

Write-Host "Baseline model found" -ForegroundColor Green
Write-Host ""

# start mlflow in background with explicit backend store
Write-Host "Starting MLflow UI on http://localhost:5000..." -ForegroundColor Cyan
$mlrunsPath = Join-Path $PWD "mlruns"
$mlflowJob = Start-Job -ScriptBlock { 
    Set-Location $using:PWD
    python -m mlflow ui --backend-store-uri "file:///$using:mlrunsPath" --host 0.0.0.0 --port 5000 
}

# wait for mlflow to start
Start-Sleep -Seconds 3

# start streamlit
Write-Host "Starting Streamlit App on http://localhost:8501..." -ForegroundColor Cyan
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "                    SYSTEM READY                            " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  MLflow UI:      http://localhost:5000                     " -ForegroundColor Yellow
Write-Host "  Streamlit App:  http://localhost:8501                     " -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop both services" -ForegroundColor Gray
Write-Host ""

try {
    # run streamlit (foreground) using python -m to ensure it's found
    python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
} finally {
    # cleanup: stop mlflow job
    Write-Host "`nStopping services..." -ForegroundColor Yellow
    Stop-Job -Job $mlflowJob
    Remove-Job -Job $mlflowJob
    Write-Host "Services stopped" -ForegroundColor Green
}
