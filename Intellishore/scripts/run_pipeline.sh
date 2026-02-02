#!/bin/bash
# dengue forecasting pipeline execution script
# runs the complete forecasting pipeline from data loading to visualization

set -e  # exit on error

echo "=================================="
echo "Dengue Forecasting Pipeline"
echo "=================================="
echo ""

# check if python is available
if ! command -v python &> /dev/null; then
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python is not installed or not in PATH"
        exit 1
    else
        PYTHON_CMD="python3"
    fi
else
    PYTHON_CMD="python"
fi

echo "Using: $PYTHON_CMD"
echo ""

# check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "No virtual environment found (optional)"
fi

echo ""

# check if required packages are installed
echo "Checking dependencies..."
$PYTHON_CMD -c "import pandas, numpy, sklearn, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    $PYTHON_CMD -m pip install -r requirements.txt
else
    echo "All dependencies satisfied ✓"
fi

echo ""
echo "Starting pipeline execution..."
echo ""

# run the pipeline
$PYTHON_CMD -m src.main

echo ""
echo "=================================="
echo "Pipeline execution completed ✓"
echo "=================================="
