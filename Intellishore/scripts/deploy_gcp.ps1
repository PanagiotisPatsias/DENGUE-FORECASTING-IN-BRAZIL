# Deploy Dengue Forecasting System to Google Cloud Run
# PowerShell version for Windows users
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Docker Desktop running
#   - GCP project created with billing enabled
#   - Cloud Run API enabled

param(
    [string]$ProjectId = $env:GCP_PROJECT_ID,
    [string]$Region = "us-central1",
    [string]$ServiceName = "dengue-forecasting"
)

$ErrorActionPreference = "Stop"

if (-not $ProjectId) {
    Write-Host "[ERROR] GCP_PROJECT_ID not set. Use: -ProjectId your-project-id" -ForegroundColor Red
    exit 1
}

$ImageName = "gcr.io/$ProjectId/$ServiceName"

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Dengue Forecasting - GCP Deployment" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Project ID: $ProjectId"
Write-Host "Region: $Region"
Write-Host "Service: $ServiceName"
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if gcloud is installed
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] gcloud CLI not found" -ForegroundColor Red
    Write-Host "Install from: https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

# Set project
Write-Host "[1/6] Setting GCP project..." -ForegroundColor Yellow
gcloud config set project $ProjectId

# Enable required APIs
Write-Host "[2/6] Enabling required APIs..." -ForegroundColor Yellow
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# Build Docker image
Write-Host "[3/6] Building Docker image..." -ForegroundColor Yellow
docker build -t "${ImageName}:latest" .

# Push to GCR
Write-Host "[4/6] Pushing image to Google Container Registry..." -ForegroundColor Yellow
docker push "${ImageName}:latest"

# Deploy to Cloud Run
Write-Host "[5/6] Deploying Streamlit app to Cloud Run..." -ForegroundColor Yellow
gcloud run deploy $ServiceName `
    --image "${ImageName}:latest" `
    --platform managed `
    --region $Region `
    --allow-unauthenticated `
    --port 8501 `
    --memory 2Gi `
    --cpu 2 `
    --timeout 300 `
    --set-env-vars="MLFLOW_TRACKING_URI=./mlruns" `
    --max-instances 10

# Get service URL
Write-Host "[6/6] Deployment complete!" -ForegroundColor Green
Write-Host ""
$ServiceUrl = gcloud run services describe $ServiceName --region $Region --format "value(status.url)"
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Streamlit App URL: $ServiceUrl" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Notes:" -ForegroundColor Yellow
Write-Host "  - MLflow UI requires separate deployment"
Write-Host "  - Consider Cloud Storage for mlruns in production"
Write-Host "  - Check Cloud Logging for application logs"
Write-Host ""
