"""
standalone script to monitor model performance and detect drift.
can be run independently to check for drift on new data.
"""

import argparse
import sys
from pathlib import Path

# add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.data_loader import DataLoader
from src.core.feature_engineer import FeatureEngineer
from src.core.model_trainer import ModelTrainer
from src.monitoring.model_monitor import ModelMonitor
from src.utils.model_manager import ModelManager


def monitor_model_performance(
    test_year: int,
    train_start_year: int = 2010,
    exclude_years: list = None,
    drift_threshold: float = 0.15,
    performance_threshold: float = 0.3
):
    """
    monitor model performance on new data and detect drift.
    
    args:
        test_year: year to test for drift
        train_start_year: starting year for training
        exclude_years: list of years to exclude from training
        drift_threshold: threshold for detecting performance drift
        performance_threshold: minimum acceptable r2 score
    """
    print("\n" + "=" * 80)
    print("DENGUE FORECASTING - MODEL DRIFT MONITORING")
    print("=" * 80)
    
    # initialize components
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    model_manager = ModelManager()
    monitor = ModelMonitor(
        drift_threshold=drift_threshold,
        performance_threshold=performance_threshold
    )
    
    print("\n[METRICS] Loading and preparing data...")
    df = data_loader.load_and_prepare_data()
    df, feature_cols = feature_engineer.create_features(df)
    print(f" Data loaded: {len(df)} quarters, {len(feature_cols)} features")
    
    # load baseline model instead of retraining
    print("\n[METRICS] Loading baseline model...")
    baseline_result = model_manager.load_baseline_model()
    
    if baseline_result is None:
        print("\n[WARNING]  ERROR: No baseline model found!")
        print("   Please run the main pipeline first to train and save the baseline model.")
        print("   Run: python -m src.main")
        return 1
    
    baseline_model, baseline_metadata = baseline_result
    
    # set baseline metrics from saved model
    monitor.set_baseline(
        metrics=baseline_metadata['metrics'],
        model_name=baseline_metadata['model_name'],
        test_year=baseline_metadata['test_year']
    )
    
    print(f" Baseline established from saved model:")
    print(f"  → Model: {baseline_metadata['model_name']}")
    print(f"  → Test Year: {baseline_metadata['test_year']}")
    print(f"  → R²: {baseline_metadata['metrics']['r2']:.4f}")
    
    # prepare test data using same features as baseline
    print(f"\n[METRICS] Testing on year {test_year}...")
    baseline_features = baseline_metadata['features']
    
    # filter to test year and prepare features
    test_df = df[df['year'] == test_year].copy()
    if test_df.empty:
        print(f"\n[WARNING]  ERROR: No data found for year {test_year}")
        return 1
    
    # ensure all baseline features exist
    missing_features = [f for f in baseline_features if f not in test_df.columns]
    if missing_features:
        print(f"\n[WARNING]  WARNING: Missing features in test data: {missing_features[:5]}...")
        # use only available features
        baseline_features = [f for f in baseline_features if f in test_df.columns]
    
    X_test = test_df[baseline_features]
    y_test = test_df['casos_est']
    
    # make predictions using baseline model
    predictions = baseline_model.predict(X_test)
    
    # calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    current_metrics = {
        'r2': r2,
        'mae': mae,
        'rmse': rmse
    }
    
    # print test results
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'R²':<20} {r2:>15.4f}")
    print(f"{'MAE':<20} {mae:>15,.0f}")
    print(f"{'RMSE':<20} {rmse:>15,.0f}")
    
    # detect drift
    print("\n[METRICS] Detecting drift...")
    drift_detected, drift_reasons = monitor.detect_performance_drift(
        current_metrics=current_metrics,
        model_name=baseline_metadata['model_name'],
        test_year=test_year
    )
    
    # check data drift
    baseline_year = baseline_metadata['test_year']
    baseline_df = df[df['year'] == baseline_year]
    test_df_full = df[df['year'] == test_year]
    data_drift, data_drift_reasons = monitor.check_data_drift(
        test_df_full, baseline_df, baseline_features
    )
    
    if data_drift:
        print("\n[WARNING]  DATA DRIFT DETECTED:")
        for reason in data_drift_reasons:
            print(f"  • {reason}")
    
    # generate report
    report = monitor.generate_drift_report(
        current_metrics={
            'r2': best_test_res['r2'],
            'mae': best_test_res['mae'],
            'rmse': best_test_res['rmse']
        },
        model_name=best_test_name,
        test_year=test_year,
        save_path=f"drift_report_{test_year}.txt"
    )
    print(report)
    
    # log alert
    monitor.log_drift_alert(
        drift_detected=drift_detected,
        drift_reasons=drift_reasons,
        model_name=best_test_name,
        test_year=test_year
    )
    
    # export summary
    monitor.export_monitoring_summary(f"monitoring_summary_{test_year}.json")
    
    # return exit code based on drift
    return 1 if drift_detected else 0


def main():
    """main entry point for cli usage."""
    parser = argparse.ArgumentParser(
        description='Monitor dengue forecasting model for drift'
    )
    parser.add_argument(
        '--test-year',
        type=int,
        required=True,
        help='Year to test for drift'
    )
    parser.add_argument(
        '--train-start',
        type=int,
        default=2010,
        help='Starting year for training (default: 2010)'
    )
    parser.add_argument(
        '--exclude-years',
        type=int,
        nargs='+',
        default=None,
        help='Years to exclude from training'
    )
    parser.add_argument(
        '--drift-threshold',
        type=float,
        default=0.15,
        help='Threshold for drift detection (default: 0.15)'
    )
    parser.add_argument(
        '--performance-threshold',
        type=float,
        default=0.3,
        help='Minimum acceptable R2 score (default: 0.3)'
    )
    
    args = parser.parse_args()
    
    exit_code = monitor_model_performance(
        test_year=args.test_year,
        train_start_year=args.train_start,
        exclude_years=args.exclude_years,
        drift_threshold=args.drift_threshold,
        performance_threshold=args.performance_threshold
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
