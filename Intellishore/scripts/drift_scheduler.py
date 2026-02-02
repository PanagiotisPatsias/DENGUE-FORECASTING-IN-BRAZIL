"""
Automated Drift Monitoring Scheduler
Run this script to start continuous drift monitoring.

Usage:
    python drift_scheduler.py --interval daily
    python drift_scheduler.py --interval hourly
    python drift_scheduler.py --interval 6h
    python drift_scheduler.py --run-once
"""

import schedule
import time
import argparse
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_drift_check():
    """Execute drift detection and log to MLflow."""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running scheduled drift check...")
    print('='*80)
    
    try:
        from src.core.data_loader import DataLoader
        from src.core.feature_engineer import FeatureEngineer
        from src.monitoring.model_monitor import ModelMonitor
        from src.utils.model_manager import ModelManager
        from src.utils.config import Config
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import numpy as np
        
        # Load baseline model
        model_manager = ModelManager()
        model, metadata = model_manager.load_baseline_model()
        
        print(f"   [OK] Loaded {metadata['model_name']} (Baseline R²: {metadata['metrics']['r2']:.4f})")
        
        # Load and process latest data
        loader = DataLoader(Config.DENGUE_DATA_PATH)
        data = loader.load_and_prepare_data()
        
        engineer = FeatureEngineer()
        processed_data, _ = engineer.create_features(data)
        
        # Get latest available data
        latest_year = processed_data['year'].max()
        test_data = processed_data[processed_data['year'] == latest_year]
        
        if len(test_data) == 0:
            print(f"   [WARNING] No data available for year {latest_year}")
            return
        
        print(f"   [OK] Testing on year {latest_year} ({len(test_data)} samples)")
        
        X_test = test_data[metadata['features']]
        y_test = test_data['casos_est']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        current_metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        print(f"   [METRICS] R²: {current_metrics['r2']:.4f} | MAE: {current_metrics['mae']:,.0f} | RMSE: {current_metrics['rmse']:,.0f}")
        
        # Initialize monitor and set baseline
        monitor = ModelMonitor(
            experiment_name="dengue_forecasting",
            tracking_uri="./mlruns"
        )
        monitor.set_baseline(
            metrics=metadata['metrics'],
            model_name=metadata['model_name'],
            test_year=metadata['test_year']
        )
        
        # Run drift detection (automatically logs to MLflow)
        drift_detected, reasons = monitor.detect_performance_drift(
            current_metrics=current_metrics,
            model_name=metadata['model_name'],
            test_year=latest_year
        )
        
        # Send alert if drift detected
        if drift_detected:
            print(f"   [ALERT] DRIFT DETECTED - {len(reasons)} issue(s)")
            for reason in reasons:
                print(f"      - {reason}")
            send_drift_alert(reasons, current_metrics, latest_year)
        else:
            print("   [OK] No drift detected")
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Drift check complete")
        print('='*80)
        
    except Exception as e:
        print(f"\n[ERROR] Drift check failed: {str(e)}")
        import traceback
        traceback.print_exc()

        
    except Exception as e:
        print(f"\n[ERROR] Drift check failed: {e}")
        import traceback
        traceback.print_exc()


def send_drift_alert(reasons: list, metrics: dict, year: int):
    """
    Send alert when drift is detected.
    Customize this to send emails, Slack messages, etc.
    """
    print("\n" + "!"*80)
    print("[ALERT] DRIFT DETECTED - SENDING NOTIFICATION")
    print("!"*80)
    
    # Write to alert log file
    alert_file = Path("drift_alerts.log")
    with open(alert_file, "a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] DRIFT DETECTED\n")
        f.write(f"  Year: {year}\n")
        f.write(f"  R²: {metrics['r2']:.4f}\n")
        f.write(f"  MAE: {metrics['mae']:.2f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"  Reasons:\n")
        for reason in reasons:
            f.write(f"    - {reason}\n")
    
    print(f"[OK] Alert logged to {alert_file}")
    
    # TODO: Add your notification method here
    # Examples:
    # - send_email(subject="Model Drift Alert", body=...)
    # - send_slack_message(channel="#ml-alerts", message=...)
    # - trigger_pagerduty_incident(...)


def main():
    parser = argparse.ArgumentParser(
        description="Automated Drift Monitoring Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drift_scheduler.py --interval daily       # Daily at 08:00
  python drift_scheduler.py --interval hourly      # Every hour
  python drift_scheduler.py --interval 6h          # Every 6 hours
  python drift_scheduler.py --interval 30m         # Every 30 minutes
  python drift_scheduler.py --run-once             # Run once and exit
        """
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="daily",
        help="Check interval: 'hourly', 'daily', 'weekly', or custom like '6h', '30m'"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run once and exit (useful for cron jobs)"
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("AUTOMATED DRIFT MONITORING SCHEDULER")
    print("=" * 80)
    
    # Run once mode
    if args.run_once:
        print("\n[MODE] Run once and exit")
        run_drift_check()
        return
    
    # Schedule based on interval
    interval = args.interval.lower()
    
    if interval == "hourly":
        schedule.every().hour.do(run_drift_check)
        print("\n[SCHEDULE] Every hour")
    elif interval == "daily":
        schedule.every().day.at("08:00").do(run_drift_check)
        print("\n[SCHEDULE] Daily at 08:00")
    elif interval == "weekly":
        schedule.every().monday.at("08:00").do(run_drift_check)
        print("\n[SCHEDULE] Every Monday at 08:00")
    elif interval.endswith("h"):
        hours = int(interval[:-1])
        schedule.every(hours).hours.do(run_drift_check)
        print(f"\n[SCHEDULE] Every {hours} hours")
    elif interval.endswith("m"):
        minutes = int(interval[:-1])
        schedule.every(minutes).minutes.do(run_drift_check)
        print(f"\n[SCHEDULE] Every {minutes} minutes")
    else:
        print(f"\n[ERROR] Unknown interval: {interval}")
        print("Use: hourly, daily, weekly, or custom like '6h', '30m'")
        return
    
    # Run immediately on start
    print("\n[INFO] Running initial drift check...")
    run_drift_check()
    
    # Keep running
    print(f"\n[INFO] Scheduler active. Press Ctrl+C to stop.")
    print(f"[INFO] Next run: {schedule.next_run()}\n")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        print("\n\n[INFO] Scheduler stopped by user")
        print("=" * 80)


if __name__ == "__main__":
    main()