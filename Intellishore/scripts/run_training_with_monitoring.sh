#!/bin/bash
# Complete training and monitoring startup script
# Runs training, then optionally starts drift scheduler

SCHEDULER_INTERVAL="daily"
START_SCHEDULER=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --interval)
            SCHEDULER_INTERVAL="$2"
            shift 2
            ;;
        --start-scheduler)
            START_SCHEDULER=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--interval daily|hourly|6h] [--start-scheduler]"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================"
echo "DENGUE FORECASTING - COMPLETE PIPELINE"
echo "========================================"

# Step 1: Train models
echo ""
echo "STEP 1: Training models..."
python -m src.main

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Training failed!"
    exit 1
fi

echo ""
echo "[OK] Training complete!"

# Step 2: Start drift scheduler (if requested)
if [ "$START_SCHEDULER" = true ]; then
    echo ""
    echo "STEP 2: Starting drift scheduler..."
    echo "Interval: $SCHEDULER_INTERVAL"
    echo "Press Ctrl+C to stop scheduler"
    echo ""
    
    python scripts/drift_scheduler.py --interval "$SCHEDULER_INTERVAL"
else
    echo ""
    echo "STEP 2: Drift scheduler NOT started"
    echo "To start scheduler, run:"
    echo "  python scripts/drift_scheduler.py --interval $SCHEDULER_INTERVAL"
    echo ""
    echo "Or use this script with --start-scheduler flag:"
    echo "  ./scripts/run_training_with_monitoring.sh --start-scheduler --interval daily"
fi

echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "========================================"
