"""
INTEGRATION CODE - Add this to your model_monitor.py
=====================================================

This shows exactly what to add/modify in your existing ModelMonitor class
to generate the interactive HTML dashboard in MLflow.
"""

# =============================================================================
# STEP 1: Add this import at the top of model_monitor.py
# =============================================================================

from src.monitoring.drift_dashboard_generator import DriftDashboardGenerator


# =============================================================================
# STEP 2: Add this to your __init__ method
# =============================================================================

def __init__(self, ...):
    # ... your existing code ...
    
    # ADD THIS LINE:
    self.dashboard_generator = DriftDashboardGenerator()


# =============================================================================
# STEP 3: Replace your _log_drift_to_mlflow method with this enhanced version
# =============================================================================

def _log_drift_to_mlflow(
    self,
    current_metrics: Dict[str, float],
    model_name: str,
    test_year: int,
    drift_detected: bool,
    drift_reasons: List[str],
    baseline_metrics: Dict[str, float]
) -> None:
    """
    Enhanced drift logging with interactive HTML dashboard.
    """
    # Calculate severity and changes
    severity = self._get_drift_severity(current_metrics)
    r2_drop = baseline_metrics['r2'] - current_metrics['r2']
    mae_increase_pct = ((current_metrics['mae'] - baseline_metrics['mae']) / baseline_metrics['mae']) * 100
    rmse_increase_pct = ((current_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse']) * 100
    
    with mlflow.start_run(run_name=f"drift_check_{test_year}") as run:
        # =====================================================
        # CLEAR STATUS TAGS (visible in MLflow run list)
        # =====================================================
        status_emoji = '[ALERT]' if drift_detected else '[OK]'
        mlflow.set_tag("DRIFT_STATUS", f"{status_emoji} {'RETRAIN NEEDED' if drift_detected else 'MODEL OK'}")
        mlflow.set_tag("severity", severity)
        mlflow.set_tag("run_type", "drift_check")
        mlflow.set_tag("model_tested", model_name)
        mlflow.set_tag("test_year", str(test_year))
        
        # Color-coded metric status
        mlflow.set_tag("r2_status", "[FAIL] FAIL" if current_metrics['r2'] < self.performance_threshold else "[OK] OK")
        mlflow.set_tag("mae_status", "[FAIL] +{:.0f}%".format(mae_increase_pct) if mae_increase_pct > 50 else "[OK] OK")
        
        # =====================================================
        # KEY METRICS
        # =====================================================
        mlflow.log_metric("drift_detected", 1.0 if drift_detected else 0.0)
        mlflow.log_metric("drift_severity_score", {'CRITICAL': 3, 'WARNING': 2, 'OK': 1}.get(severity, 0))
        
        # Baseline metrics
        mlflow.log_metric("baseline_r2", baseline_metrics['r2'])
        mlflow.log_metric("baseline_mae", baseline_metrics['mae'])
        mlflow.log_metric("baseline_rmse", baseline_metrics['rmse'])
        
        # Current metrics
        mlflow.log_metric("current_r2", current_metrics['r2'])
        mlflow.log_metric("current_mae", current_metrics['mae'])
        mlflow.log_metric("current_rmse", current_metrics['rmse'])
        
        # Change metrics
        mlflow.log_metric("r2_degradation", r2_drop)
        mlflow.log_metric("mae_increase_pct", mae_increase_pct)
        mlflow.log_metric("rmse_increase_pct", rmse_increase_pct)
        
        # Thresholds
        mlflow.log_param("drift_threshold", self.drift_threshold)
        mlflow.log_param("performance_threshold", self.performance_threshold)
        
        # =====================================================
        # GENERATE AND LOG HTML DASHBOARD
        # =====================================================
        
        # Prepare metrics history for time series chart
        history_data = []
        for entry in self.metrics_history:
            status = 'ok'
            if entry['drift_detected']:
                status = 'critical'
            elif entry['metrics']['r2'] < self.performance_threshold:
                status = 'warning'
            
            history_data.append({
                'period': str(entry['test_year']),
                'r2': entry['metrics']['r2'],
                'mae': entry['metrics']['mae'],
                'status': status
            })
        
        # Generate HTML dashboard
        dashboard_path = self.dashboard_generator.generate_dashboard(
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            test_year=test_year,
            baseline_year=self.baseline_metrics.get('test_year', 2022),
            drift_detected=drift_detected,
            drift_reasons=drift_reasons,
            drift_threshold=self.drift_threshold,
            performance_threshold=self.performance_threshold,
            metrics_history=history_data if history_data else None,
            save_path=f"drift_dashboard_{test_year}.html"
        )
        
        # Log dashboard as artifact
        mlflow.log_artifact(dashboard_path)
        Path(dashboard_path).unlink()  # Clean up local file
        
        # =====================================================
        # ALSO LOG JSON REPORT FOR PROGRAMMATIC ACCESS
        # =====================================================
        drift_report = {
            'summary': {
                'drift_detected': drift_detected,
                'severity': severity,
                'recommendation': 'RETRAIN IMMEDIATELY' if severity == 'CRITICAL' else 
                                  'Consider retraining' if severity == 'WARNING' else 'No action needed'
            },
            'test_year': test_year,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics_comparison': {
                'r2': {'baseline': baseline_metrics['r2'], 'current': current_metrics['r2'], 'change': -r2_drop},
                'mae': {'baseline': baseline_metrics['mae'], 'current': current_metrics['mae'], 'change_pct': mae_increase_pct},
                'rmse': {'baseline': baseline_metrics['rmse'], 'current': current_metrics['rmse'], 'change_pct': rmse_increase_pct}
            },
            'drift_reasons': drift_reasons
        }
        
        report_path = f"drift_report_{test_year}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(drift_report, f, indent=2)
        mlflow.log_artifact(report_path)
        Path(report_path).unlink()


def _get_drift_severity(self, current_metrics: Dict[str, float]) -> str:
    """Determine drift severity level."""
    if not self.baseline_metrics:
        return 'UNKNOWN'
    
    baseline_r2 = self.baseline_metrics['metrics']['r2']
    current_r2 = current_metrics['r2']
    r2_drop = baseline_r2 - current_r2
    
    if current_r2 < 0 or r2_drop > 0.3:
        return 'CRITICAL'
    elif current_r2 < self.performance_threshold or r2_drop > self.drift_threshold:
        return 'WARNING'
    return 'OK'


# =============================================================================
# STEP 4: Add this method to print clear console output
#         Call it at the end of detect_performance_drift()
# =============================================================================

def _print_drift_summary(
    self,
    current_metrics: Dict[str, float],
    drift_detected: bool,
    drift_reasons: List[str],
    test_year: int
) -> None:
    """Print clear, formatted drift summary to console."""
    severity = self._get_drift_severity(current_metrics)
    
    baseline = self.baseline_metrics['metrics']
    r2_drop = baseline['r2'] - current_metrics['r2']
    mae_pct = ((current_metrics['mae'] - baseline['mae']) / baseline['mae']) * 100
    
    emoji = {'CRITICAL': '[ALERT]', 'WARNING': '[WARNING]', 'OK': '[OK]'}.get(severity, '[?]')
    
    print("\n" + "=" * 60)
    print(f"{emoji} DRIFT CHECK RESULT: {severity}")
    print("=" * 60)
    print(f"  Test Year:      {test_year}")
    print(f"  Baseline R²:    {baseline['r2']:.4f}")
    print(f"  Current R²:     {current_metrics['r2']:.4f} ({r2_drop:+.4f})")
    print(f"  MAE Change:     {mae_pct:+.1f}%")
    print("-" * 60)
    
    if drift_detected:
        print("  [ALERT] ACTION REQUIRED: RETRAIN MODEL")
        print("\n  Reasons:")
        for reason in drift_reasons:
            print(f"    • {reason}")
    else:
        print("  [OK] No action required")
    
    print("=" * 60)
    print("  [METRICS] View dashboard: MLflow UI → Artifacts → drift_dashboard_{}.html".format(test_year))
    print("=" * 60 + "\n")
