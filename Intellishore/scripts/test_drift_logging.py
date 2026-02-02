"""
test script to demonstrate drift detection logging to mlflow.
this will create a drift check run that shows up in mlflow ui.
"""

import sys
from pathlib import Path

# add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.data_loader import DataLoader
from src.core.feature_engineer import FeatureEngineer
from src.monitoring.model_monitor import ModelMonitor
from src.utils.model_manager import ModelManager
from src.utils.config import Config
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def test_drift_detection():
    """test drift detection and logging to mlflow"""
    
    print("\n" + "=" * 80)
    print("TESTING DRIFT DETECTION LOGGING TO MLFLOW")
    print("=" * 80)
    
    # load baseline model
    print("\n1. Loading baseline model...")
    model_manager = ModelManager()
    model, metadata = model_manager.load_baseline_model()
    
    print(f"    Loaded {metadata['model_name']}")
    print(f"    Baseline R²: {metadata['metrics']['r2']:.4f}")
    print(f"    Test Year: {metadata['test_year']}")
    
    # initialize monitor
    print("\n2. Initializing ModelMonitor...")
    monitor = ModelMonitor(
        experiment_name="dengue_forecasting",
        tracking_uri="./mlruns",
        drift_threshold=0.15
    )
    
    # set baseline from saved model
    monitor.set_baseline(
        metrics=metadata['metrics'],
        model_name=metadata['model_name'],
        test_year=metadata['test_year']
    )
    
    # load data for 2025
    print("\n3. Loading 2025 data for drift testing...")
    loader = DataLoader(Config.DENGUE_DATA_PATH)
    data = loader.load_and_prepare_data()
    
    engineer = FeatureEngineer()
    processed_data, feature_names = engineer.create_features(data)
    
    # prepare 2025 test data
    test_year = 2025
    test_data = processed_data[processed_data['year'] == test_year].copy()
    
    if len(test_data) == 0:
        print(f"    No data for year {test_year}")
        return
    
    X_test = test_data[metadata['features']]
    y_test = test_data['casos_est']
    
    print(f"    Loaded {len(test_data)} samples from {test_year}")
    
    # make predictions
    print("\n4. Making predictions...")
    y_pred = model.predict(X_test)
    
    # calculate metrics
    current_metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    print(f"   Current R²: {current_metrics['r2']:.4f}")
    print(f"   Current MAE: {current_metrics['mae']:,.0f}")
    print(f"   Current RMSE: {current_metrics['rmse']:,.0f}")
    
    # detect drift (this will automatically log to mlflow)
    print("\n5. Checking for drift (logging to MLflow)...")
    drift_detected, drift_reasons = monitor.detect_performance_drift(
        current_metrics=current_metrics,
        model_name=metadata['model_name'],
        test_year=test_year
    )
    
    # display results
    print("\n" + "=" * 80)
    if drift_detected:
        print("[ALERT] DRIFT DETECTED")
        print("=" * 80)
        print("\nReasons:")
        for reason in drift_reasons:
            print(f"  • {reason}")
        print("\n Recommendation: Retrain model with recent data")
    else:
        print(" NO DRIFT DETECTED")
        print("=" * 80)
        print("Model is performing within acceptable thresholds")
    
    print("\n" + "=" * 80)
    print("MLFLOW LOGGING COMPLETE")
    print("=" * 80)
    print("\nTo view in MLflow UI:")
    print("  1. Run: .\\scripts\\run_app.ps1")
    print("  2. Open: http://localhost:5000")
    print("  3. Look for run: 'drift_check_2025'")
    print(f"  4. Tags: drift_detected={drift_detected}")
    print("\nWhat you'll see in MLflow:")
    print("  • Run tagged as 'drift_check'")
    print("  • Drift status in tags")
    print("  • Baseline vs current metrics")
    print("  • R² degradation metric")
    print("  • Drift report JSON artifact")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_drift_detection()
