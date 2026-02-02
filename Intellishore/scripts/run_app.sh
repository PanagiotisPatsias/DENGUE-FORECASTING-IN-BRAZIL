#!/bin/bash
# launch mlflow ui and streamlit app together

echo " Starting Dengue Forecasting System...."
echo ""

# check if model exists
if [ ! -f "models/baseline_model.pkl" ]; then
    echo "No baseline model found!"
    echo "   Please run training first: python -m src.main"
    exit 1
fi

echo "✓ Baseline model found"
echo ""

# start mlflow in background with explicit backend store
echo "📊 Starting MLflow UI on http://localhost:5000..."
mlflow ui --backend-store-uri "file://$(pwd)/mlruns" --host 0.0.0.0 --port 5000 > /dev/null 2>&1 &
MLFLOW_PID=$!

# wait a bit for mlflow to start
sleep 2

# start streamlit
echo "Starting Streamlit App on http://localhost:8501..."
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    SYSTEM READY                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  📊 MLflow UI:      http://localhost:5000                  ║"
echo "║  🎨 Streamlit App:  http://localhost:8501                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# trap to kill mlflow when script exits
trap "kill $MLFLOW_PID 2>/dev/null" EXIT

# run streamlit (foreground)
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
