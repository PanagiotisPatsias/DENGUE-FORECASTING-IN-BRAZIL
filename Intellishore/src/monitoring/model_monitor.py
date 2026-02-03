"""
mlflow monitoring module for dengue forecasting.
tracks model performance, detects drift, and triggers retraining alerts.

ENHANCED VERSION with:
- Interactive HTML dashboard
- Clear status tags ([ALERT] RETRAIN NEEDED / [OK] MODEL OK)
- Severity scoring (CRITICAL/WARNING/OK)
- Better visualizations
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import warnings
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# suppress mlflow warnings
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'
import logging
logging.getLogger('mlflow').setLevel(logging.ERROR)

# Import dashboard generator
try:
    from src.monitoring.drift_dashboard_generator import DriftDashboardGenerator
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("[WARNING] DriftDashboardGenerator not found. HTML dashboards disabled.")


class ModelMonitor:
    """
    model monitoring class following single responsibility principle.
    responsible for tracking performance and detecting drift.
    """
    
    def __init__(
        self,
        experiment_name: str = "dengue_forecasting",
        tracking_uri: str = "./mlruns",
        drift_threshold: float = 0.15,
        performance_threshold: float = 0.3
    ):
        """
        initialize model monitor with mlflow configuration.
        
        args:
            experiment_name: name of mlflow experiment
            tracking_uri: path to mlflow tracking directory
            drift_threshold: threshold for detecting performance drift
            performance_threshold: minimum acceptable r2 score
        """
        self.experiment_name = experiment_name
        # Allow env var override for local vs cloud usage
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", tracking_uri)
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        
        # setup mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # baseline metrics storage
        self.baseline_metrics = {}
        self.metrics_history = []
        
        # drift detection state
        self.drift_detected = False
        self.drift_reasons = []
        
        # Initialize dashboard generator
        if DASHBOARD_AVAILABLE:
            self.dashboard_generator = DriftDashboardGenerator()
        else:
            self.dashboard_generator = None
    
    def log_training_run(
        self,
        model: Any,
        model_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        features: List[str],
        train_year_range: Tuple[int, int],
        test_year: int,
        artifacts: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        log a complete training run to mlflow.
        
        args:
            model: trained model instance
            model_name: name of the model
            params: model hyperparameters
            metrics: evaluation metrics (r2, mae, rmse)
            features: list of feature names used
            train_year_range: tuple of (start_year, end_year) for training
            test_year: year used for testing
            artifacts: optional additional artifacts to log
            
        returns:
            run_id of the logged run
        """
        with mlflow.start_run(run_name=f"{model_name}_{test_year}") as run:
            # log parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("train_start_year", train_year_range[0])
            mlflow.log_param("train_end_year", train_year_range[1])
            mlflow.log_param("test_year", test_year)
            mlflow.log_param("num_features", len(features))
            
            # log model hyperparameters
            for param_name, param_value in params.items():
                mlflow.log_param(f"model_{param_name}", param_value)
            
            # log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"{model_name}_dengue_forecaster",
                pip_requirements=None
            )
            
            # log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                Path(importance_path).unlink()
            
            # log features list
            features_path = "features.json"
            with open(features_path, 'w') as f:
                json.dump(features, f)
            mlflow.log_artifact(features_path)
            Path(features_path).unlink()
            
            # log additional artifacts
            if artifacts:
                for artifact_name, artifact_data in artifacts.items():
                    artifact_path = f"{artifact_name}.json"
                    with open(artifact_path, 'w') as f:
                        json.dump(artifact_data, f)
                    mlflow.log_artifact(artifact_path)
                    Path(artifact_path).unlink()
            
            # log timestamp
            mlflow.log_param("logged_at", datetime.now().isoformat())
            
            return run.info.run_id
    
    def set_baseline(
        self,
        metrics: Dict[str, float],
        model_name: str,
        test_year: int
    ) -> None:
        """
        set baseline metrics for drift detection.
        
        args:
            metrics: baseline metrics (r2, mae, rmse)
            model_name: name of the baseline model
            test_year: year used for baseline
        """
        self.baseline_metrics = {
            'metrics': metrics.copy(),
            'model_name': model_name,
            'test_year': test_year,
            'timestamp': datetime.now().isoformat()
        }
        
        print("\n" + "=" * 60)
        print("[METRICS] BASELINE METRICS SET")
        print("=" * 60)
        print(f"  Model:     {model_name}")
        print(f"  Test Year: {test_year}")
        print(f"  R² Score:  {metrics['r2']:.4f}")
        print(f"  MAE:       {metrics['mae']:,.0f}")
        print(f"  RMSE:      {metrics['rmse']:,.0f}")
        print("=" * 60)
    
    def _get_drift_severity(self, current_metrics: Dict[str, float]) -> str:
        """
        Determine drift severity level.
        
        Returns: 'CRITICAL', 'WARNING', or 'OK'
        """
        if not self.baseline_metrics:
            return 'UNKNOWN'
        
        baseline_r2 = self.baseline_metrics['metrics']['r2']
        current_r2 = current_metrics['r2']
        r2_drop = baseline_r2 - current_r2
        
        # Critical: negative R² or drop > 0.3
        if current_r2 < 0 or r2_drop > 0.3:
            return 'CRITICAL'
        # Warning: below threshold or significant drop
        elif current_r2 < self.performance_threshold or r2_drop > self.drift_threshold:
            return 'WARNING'
        else:
            return 'OK'
    
    # def _get_status_emoji(self, severity: str) -> str:
    #     """Get emoji for severity level."""
    #     return {
    #         'CRITICAL': '[ALERT]',
    #         'WARNING': '[WARNING]',
    #         'OK': '[OK]',
    #         'UNKNOWN': '[?]'
    #     }.get(severity, '[?]')
    
    def detect_performance_drift(
        self,
        current_metrics: Dict[str, float],
        model_name: str,
        test_year: int
    ) -> Tuple[bool, List[str]]:
        """
        detect if model performance has drifted from baseline.
        
        args:
            current_metrics: current evaluation metrics
            model_name: name of current model
            test_year: year being tested
            
        returns:
            tuple of (drift_detected, list of drift reasons)
        """
        if not self.baseline_metrics:
            return False, ["No baseline metrics available for comparison"]
        
        drift_reasons = []
        drift_detected = False
        
        baseline_r2 = self.baseline_metrics['metrics']['r2']
        current_r2 = current_metrics['r2']
        
        baseline_mae = self.baseline_metrics['metrics']['mae']
        current_mae = current_metrics['mae']
        
        # check r2 degradation
        r2_change = baseline_r2 - current_r2
        if r2_change > self.drift_threshold:
            drift_detected = True
            drift_reasons.append(
                f"R² degradation: {r2_change:.4f} "
                f"(baseline: {baseline_r2:.4f} → current: {current_r2:.4f})"
            )
        
        # check if r2 is below acceptable threshold
        if current_r2 < self.performance_threshold:
            drift_detected = True
            drift_reasons.append(
                f"R² below threshold: {current_r2:.4f} < {self.performance_threshold}"
            )
        
        # check mae increase (normalized by baseline)
        if baseline_mae > 0:
            mae_increase_ratio = (current_mae - baseline_mae) / baseline_mae
            if mae_increase_ratio > 0.5:  # 50% increase in error
                drift_detected = True
                drift_reasons.append(
                    f"MAE increased by {mae_increase_ratio*100:.1f}% "
                    f"(baseline: {baseline_mae:,.0f} → current: {current_mae:,.0f})"
                )
        
        # store metrics history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'test_year': test_year,
            'metrics': current_metrics.copy(),
            'drift_detected': drift_detected,
            'drift_reasons': drift_reasons.copy()
        })
        
        # log drift detection to mlflow (with dashboard)
        self._log_drift_to_mlflow(
            current_metrics=current_metrics,
            model_name=model_name,
            test_year=test_year,
            drift_detected=drift_detected,
            drift_reasons=drift_reasons,
            baseline_metrics=self.baseline_metrics['metrics']
        )
        
        # Print clear console summary
        self._print_drift_summary(current_metrics, drift_detected, drift_reasons, test_year)
        
        return drift_detected, drift_reasons
    
    def _print_drift_summary(
        self,
        current_metrics: Dict[str, float],
        drift_detected: bool,
        drift_reasons: List[str],
        test_year: int
    ) -> None:
        """Print clear, formatted drift summary to console."""
        severity = self._get_drift_severity(current_metrics)
        # emoji = self._get_status_emoji(severity)
        
        baseline = self.baseline_metrics['metrics']
        r2_drop = baseline['r2'] - current_metrics['r2']
        mae_pct = ((current_metrics['mae'] - baseline['mae']) / baseline['mae']) * 100
        
        print("\n" + "=" * 60)
        print(f" DRIFT CHECK RESULT: {severity}")
        print("=" * 60)
        print(f"  Test Year:      {test_year}")
        print(f"  Baseline R²:    {baseline['r2']:.4f}")
        print(f"  Current R²:     {current_metrics['r2']:.4f} ({-r2_drop:+.4f})")
        print(f"  MAE Change:     {mae_pct:+.1f}%")
        print("-" * 60)
        
        if drift_detected:
            print("  [ALERT] ACTION REQUIRED: RETRAIN MODEL")
            print("\n  Reasons:")
            for reason in drift_reasons:
                print(f"    • {reason}")
        else:
            print("  [OK] No action required - model performing within thresholds")
        
        print("=" * 60)
        print(f"  [METRICS] View dashboard: MLflow UI → Artifacts → drift_dashboard_{test_year}.html")
        print("=" * 60 + "\n")
    
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
            # status_emoji = self._get_status_emoji(severity)
            mlflow.set_tag("DRIFT_STATUS", f" {'RETRAIN NEEDED' if drift_detected else 'MODEL OK'}")
            mlflow.set_tag("severity", severity)
            mlflow.set_tag("run_type", "drift_check")
            mlflow.set_tag("model_tested", model_name)
            mlflow.set_tag("test_year", str(test_year))
            
            # Color-coded metric status
            mlflow.set_tag("r2_status", "[FAIL] FAIL" if current_metrics['r2'] < self.performance_threshold else "[OK] OK")
            mlflow.set_tag("mae_status", f"[FAIL] +{mae_increase_pct:.0f}%" if mae_increase_pct > 50 else "[OK] OK")
            
            # =====================================================
            # KEY METRICS
            # =====================================================
            mlflow.log_metric("drift_detected", 1.0 if drift_detected else 0.0)
            mlflow.log_metric("drift_severity_score", {'CRITICAL': 3, 'WARNING': 2, 'OK': 1, 'UNKNOWN': 0}[severity])
            mlflow.log_metric("num_drift_reasons", len(drift_reasons))
            
            # Baseline metrics
            mlflow.log_metric("baseline_r2", baseline_metrics['r2'])
            mlflow.log_metric("baseline_mae", baseline_metrics['mae'])
            mlflow.log_metric("baseline_rmse", baseline_metrics['rmse'])
            
            # Current metrics
            mlflow.log_metric("current_r2", current_metrics['r2'])
            mlflow.log_metric("current_mae", current_metrics['mae'])
            mlflow.log_metric("current_rmse", current_metrics['rmse'])
            
            # Change metrics (best for visualization)
            mlflow.log_metric("r2_degradation", r2_drop)
            mlflow.log_metric("mae_increase_pct", mae_increase_pct)
            mlflow.log_metric("rmse_increase_pct", rmse_increase_pct)
            
            # Threshold references
            mlflow.log_metric("threshold_r2_min", self.performance_threshold)
            mlflow.log_metric("threshold_drift", self.drift_threshold)
            
            # Parameters
            mlflow.log_param("drift_threshold", self.drift_threshold)
            mlflow.log_param("performance_threshold", self.performance_threshold)
            mlflow.log_param("baseline_year", self.baseline_metrics.get('test_year', 'N/A'))
            
            # =====================================================
            # GENERATE HTML DASHBOARD
            # =====================================================
            if self.dashboard_generator:
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
                
                # Generate dashboard
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
                
                mlflow.log_artifact(dashboard_path)
                Path(dashboard_path).unlink()
            
            # =====================================================
            # JSON REPORT
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
                'drift_reasons': drift_reasons,
                'thresholds': {
                    'drift_threshold': self.drift_threshold,
                    'performance_threshold': self.performance_threshold
                }
            }
            
            report_path = f"drift_report_{test_year}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(drift_report, f, indent=2)
            mlflow.log_artifact(report_path)
            Path(report_path).unlink()
    
    def check_data_drift(
        self,
        current_data: pd.DataFrame,
        baseline_data: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        detect data drift by comparing distributions.
        
        args:
            current_data: current dataset
            baseline_data: baseline dataset
            feature_cols: list of features to check
            
        returns:
            tuple of (drift_detected, list of drift reasons)
        """
        drift_reasons = []
        drift_detected = False
        
        # check for missing features
        missing_features = set(feature_cols) - set(current_data.columns)
        if missing_features:
            drift_detected = True
            drift_reasons.append(
                f"Missing features: {', '.join(missing_features)}"
            )
        
        # check statistical properties for each feature
        for feature in feature_cols:
            if feature not in current_data.columns or feature not in baseline_data.columns:
                continue
            
            baseline_mean = baseline_data[feature].mean()
            current_mean = current_data[feature].mean()
            
            baseline_std = baseline_data[feature].std()
            current_std = current_data[feature].std()
            
            # check if mean has shifted significantly (> 2 std devs)
            if baseline_std > 0:
                mean_shift = abs(current_mean - baseline_mean) / baseline_std
                if mean_shift > 2.0:
                    drift_detected = True
                    drift_reasons.append(
                        f"Feature '{feature}' mean shifted by {mean_shift:.2f} std devs"
                    )
            
            # check if variance has changed significantly
            if baseline_std > 0:
                std_ratio = current_std / baseline_std
                if std_ratio < 0.5 or std_ratio > 2.0:
                    drift_detected = True
                    drift_reasons.append(
                        f"Feature '{feature}' variance changed by {std_ratio:.2f}x"
                    )
        
        return drift_detected, drift_reasons
    
    def generate_drift_report(
        self,
        current_metrics: Dict[str, float],
        model_name: str,
        test_year: int,
        save_path: Optional[str] = None
    ) -> str:
        """
        generate comprehensive drift report.
        
        args:
            current_metrics: current evaluation metrics
            model_name: name of current model
            test_year: year being tested
            save_path: optional path to save report
            
        returns:
            formatted report string
        """
        perf_drift, perf_reasons = self.detect_performance_drift(
            current_metrics, model_name, test_year
        )
        
        severity = self._get_drift_severity(current_metrics)
        # emoji = self._get_status_emoji(severity)
        
        report = []
        report.append("\n" + "=" * 80)
        report.append(f" MODEL DRIFT DETECTION REPORT - {severity}")
        report.append("=" * 80)
        report.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {model_name}")
        report.append(f"Test Year: {test_year}")
        
        report.append("\n" + "-" * 80)
        report.append("BASELINE COMPARISON")
        report.append("-" * 80)
        
        if self.baseline_metrics:
            baseline = self.baseline_metrics['metrics']
            report.append(f"\nBaseline (Year {self.baseline_metrics['test_year']}):")
            report.append(f"  R² Score: {baseline['r2']:>10.4f}")
            report.append(f"  MAE:      {baseline['mae']:>10,.0f}")
            report.append(f"  RMSE:     {baseline['rmse']:>10,.0f}")
            
            report.append(f"\nCurrent (Year {test_year}):")
            report.append(f"  R² Score: {current_metrics['r2']:>10.4f}")
            report.append(f"  MAE:      {current_metrics['mae']:>10,.0f}")
            report.append(f"  RMSE:     {current_metrics['rmse']:>10,.0f}")
            
            # calculate changes
            r2_change = current_metrics['r2'] - baseline['r2']
            mae_change = current_metrics['mae'] - baseline['mae']
            rmse_change = current_metrics['rmse'] - baseline['rmse']
            
            report.append("\nChanges from Baseline:")
            report.append(f"  ΔR²:      {r2_change:>10.4f}")
            report.append(f"  ΔMAE:     {mae_change:>10,.0f}")
            report.append(f"  ΔRMSE:    {rmse_change:>10,.0f}")
        else:
            report.append("\n[WARNING]  No baseline metrics available")
        
        report.append("\n" + "-" * 80)
        report.append("DRIFT DETECTION RESULTS")
        report.append("-" * 80)
        
        if perf_drift:
            report.append(f"\n{emoji} DRIFT DETECTED - {severity}")
            report.append("\nDrift Reasons:")
            for i, reason in enumerate(perf_reasons, 1):
                report.append(f"  {i}. {reason}")
            
            if severity == 'CRITICAL':
                report.append("\n[ALERT] ACTION REQUIRED:")
                report.append("  • Retrain model IMMEDIATELY")
                report.append("  • Investigate root causes")
                report.append("  • Update baseline after retraining")
            else:
                report.append("\n[WARNING] RECOMMENDED:")
                report.append("  • Plan retraining soon")
                report.append("  • Monitor closely")
        else:
            report.append("\n[OK] NO SIGNIFICANT DRIFT DETECTED")
            report.append("Model performance is within acceptable range")
        
        report.append("\n" + "=" * 80)
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
        
        return report_str
    
    def log_drift_alert(
        self,
        drift_detected: bool,
        drift_reasons: List[str],
        model_name: str,
        test_year: int
    ) -> None:
        """
        log drift alert to mlflow and console.
        """
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': drift_detected,
            'drift_reasons': drift_reasons,
            'model_name': model_name,
            'test_year': test_year,
            'baseline_year': self.baseline_metrics.get('test_year', None)
        }
        
        with mlflow.start_run(run_name=f"drift_alert_{test_year}"):
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("test_year", test_year)
            mlflow.log_metric("drift_detected", 1 if drift_detected else 0)
            mlflow.log_metric("num_drift_reasons", len(drift_reasons))
            
            alert_path = "drift_alert.json"
            with open(alert_path, 'w') as f:
                json.dump(alert_data, f, indent=2)
            mlflow.log_artifact(alert_path)
            Path(alert_path).unlink()
        
        if drift_detected:
            print("\n" + "!" * 60)
            print("[ALERT] ALERT: MODEL DRIFT DETECTED - RETRAINING REQUIRED")
            print("!" * 60)
            for reason in drift_reasons:
                print(f"  • {reason}")
            print("!" * 60)
    
    def get_metrics_history(self) -> pd.DataFrame:
        """
        get history of all monitored metrics.
        
        returns:
            dataframe with metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()
        
        history_data = []
        for entry in self.metrics_history:
            row = {
                'timestamp': entry['timestamp'],
                'model_name': entry['model_name'],
                'test_year': entry['test_year'],
                'r2': entry['metrics']['r2'],
                'mae': entry['metrics']['mae'],
                'rmse': entry['metrics']['rmse'],
                'drift_detected': entry['drift_detected'],
                'num_drift_reasons': len(entry['drift_reasons'])
            }
            history_data.append(row)
        
        return pd.DataFrame(history_data)
    
    def export_monitoring_summary(self, output_path: str = "monitoring_summary.json") -> None:
        """
        export complete monitoring summary to file.
        """
        summary = {
            'experiment_name': self.experiment_name,
            'tracking_uri': self.tracking_uri,
            'drift_threshold': self.drift_threshold,
            'performance_threshold': self.performance_threshold,
            'baseline_metrics': self.baseline_metrics,
            'metrics_history': self.metrics_history,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n Monitoring summary exported to: {output_path}")
