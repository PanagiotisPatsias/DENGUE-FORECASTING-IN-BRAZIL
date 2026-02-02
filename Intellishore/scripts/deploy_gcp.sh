#!/bin/bash
# Deploy Dengue Forecasting System to Google Cloud Run
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Docker installed
#   - GCP project created with billing enabled
#   - Cloud Run API enabled

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-your-project-id}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="dengue-forecasting"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "====================================="
echo "Dengue Forecasting - GCP Deployment"
echo "====================================="
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo "====================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "[ERROR] gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo "[1/6] Setting GCP project..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "[2/6] Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com

# Build and push Docker image
echo "[3/6] Building Docker image..."
docker build -t $IMAGE_NAME:latest .

echo "[4/6] Pushing image to Google Container Registry..."
docker push $IMAGE_NAME:latest

# Deploy to Cloud Run (Streamlit app)
echo "[5/6] Deploying Streamlit app to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_NAME:latest \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --port 8501 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --set-env-vars="MLFLOW_TRACKING_URI=./mlruns" \
    --max-instances 10

# Get service URL
echo "[6/6] Deployment complete!"
echo ""
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
echo "====================================="
echo "Streamlit App URL: $SERVICE_URL"
echo "====================================="
echo ""
echo "Notes:"
echo "  - MLflow UI not included in Cloud Run deployment (use separate VM or Cloud Run service)"
echo "  - For production, consider using Cloud Storage for mlruns and models"
echo "  - Enable Cloud Logging for monitoring"
echo ""
