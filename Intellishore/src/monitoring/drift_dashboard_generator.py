"""
Drift Dashboard Generator for MLflow
Creates an interactive HTML dashboard as an MLflow artifact.
Add this file to your src/monitoring/ folder.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json


class DriftDashboardGenerator:
    """
    Generates interactive HTML dashboards for drift detection.
    Logs to MLflow as an artifact that can be viewed in the browser.
    """
    
    def __init__(self):
        self.template = self._get_html_template()
    
    def generate_dashboard(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        test_year: int,
        baseline_year: int,
        drift_detected: bool,
        drift_reasons: List[str],
        drift_threshold: float = 0.15,
        performance_threshold: float = 0.3,
        metrics_history: Optional[List[dict]] = None,
        save_path: str = "drift_dashboard.html"
    ) -> str:
        """
        Generate an interactive HTML dashboard.
        
        Args:
            baseline_metrics: dict with r2, mae, rmse
            current_metrics: dict with r2, mae, rmse
            test_year: year being tested
            baseline_year: baseline reference year
            drift_detected: whether drift was detected
            drift_reasons: list of reasons for drift
            drift_threshold: threshold for r2 degradation
            performance_threshold: minimum acceptable r2
            metrics_history: optional list of historical metrics
            save_path: where to save the HTML file
            
        Returns:
            path to saved HTML file
        """
        # Calculate derived values
        r2_drop = baseline_metrics['r2'] - current_metrics['r2']
        mae_increase_pct = ((current_metrics['mae'] - baseline_metrics['mae']) / baseline_metrics['mae']) * 100
        rmse_increase_pct = ((current_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse']) * 100
        
        # Determine severity
        if current_metrics['r2'] < 0 or r2_drop > 0.3:
            severity = 'CRITICAL'
        elif current_metrics['r2'] < performance_threshold or r2_drop > drift_threshold:
            severity = 'WARNING'
        else:
            severity = 'OK'
        
        # Prepare metrics history for chart (use provided or create minimal)
        if metrics_history and len(metrics_history) >= 2:
            history_json = json.dumps(metrics_history)
        else:
            # Create minimal history with baseline and current
            history_json = json.dumps([
                {'period': str(baseline_year), 'r2': baseline_metrics['r2'], 'mae': baseline_metrics['mae'], 'status': 'ok'},
                {'period': str(test_year), 'r2': current_metrics['r2'], 'mae': current_metrics['mae'], 'status': 'critical' if drift_detected else 'ok'}
            ])
        
        # Format drift reasons
        drift_reasons_html = "".join([f"<li>{reason}</li>" for reason in drift_reasons]) if drift_reasons else "<li>None</li>"
        
        # Pre-calculate display values
        drift_reasons_display = 'block' if drift_detected else 'none'
        alert_banner_class = 'critical' if drift_detected else 'ok'
        severity_badge_class = severity.lower()
        drift_multiplier = r2_drop / drift_threshold if drift_threshold > 0 else 0
        
        # Fill template
        html = self.template.format(
            # Header info
            test_year=test_year,
            baseline_year=baseline_year,
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            # Status
            drift_detected=str(drift_detected).lower(),
            severity=severity,
            status_text='DRIFT DETECTED - RETRAIN REQUIRED' if drift_detected else 'Model Healthy',
            status_emoji='[ALERT]' if drift_detected else '[OK]',
            
            # Metrics
            baseline_r2=baseline_metrics['r2'],
            current_r2=current_metrics['r2'],
            baseline_mae=baseline_metrics['mae'],
            current_mae=current_metrics['mae'],
            baseline_rmse=baseline_metrics['rmse'],
            current_rmse=current_metrics['rmse'],
            
            # Calculated changes
            r2_drop=r2_drop,
            mae_increase_pct=mae_increase_pct,
            rmse_increase_pct=rmse_increase_pct,
            
            # Thresholds
            drift_threshold=drift_threshold,
            performance_threshold=performance_threshold,
            
            # For charts
            r2_normalized=int((current_metrics['r2'] / baseline_metrics['r2']) * 100) if baseline_metrics['r2'] != 0 else 0,
            mae_normalized=int((current_metrics['mae'] / baseline_metrics['mae']) * 100),
            rmse_normalized=int((current_metrics['rmse'] / baseline_metrics['rmse']) * 100),
            
            # History data for time series
            metrics_history_json=history_json,
            
            # Drift reasons
            drift_reasons_html=drift_reasons_html,
            drift_reasons_display=drift_reasons_display,
            alert_banner_class=alert_banner_class,
            severity_badge_class=severity_badge_class,
            drift_multiplier=drift_multiplier,
            
            # Card statuses
            r2_status='FAIL' if current_metrics['r2'] < performance_threshold else 'OK',
            r2_card_class='critical' if current_metrics['r2'] < performance_threshold else 'ok',
            mae_status=f"+{mae_increase_pct:.0f}%" if mae_increase_pct > 0 else f"{mae_increase_pct:.0f}%",
            mae_card_class='critical' if mae_increase_pct > 50 else 'warning' if mae_increase_pct > 25 else 'ok',
            rmse_status=f"+{rmse_increase_pct:.0f}%" if rmse_increase_pct > 0 else f"{rmse_increase_pct:.0f}%",
            rmse_card_class='critical' if rmse_increase_pct > 50 else 'warning' if rmse_increase_pct > 25 else 'ok',
            
            # Progress bar widths
            baseline_r2_width=max(0, min(100, baseline_metrics['r2'] * 100)),
            current_r2_width=max(0, min(100, current_metrics['r2'] * 100)) if current_metrics['r2'] > 0 else 0,
            threshold_r2_width=performance_threshold * 100,
        )
        
        # Save file
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return save_path
    
    def _get_html_template(self) -> str:
        """Return the HTML template for the dashboard."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drift Detection Dashboard - {test_year}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e4e4e7;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            font-size: 1.75rem;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .header-info {{
            font-size: 0.875rem;
            color: #9ca3af;
            margin-bottom: 20px;
        }}
        
        /* Alert Banner */
        .alert-banner {{
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        
        .alert-banner.critical {{
            background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
            border: 2px solid #ef4444;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
        }}
        
        .alert-banner.ok {{
            background: linear-gradient(135deg, #14532d 0%, #166534 100%);
            border: 2px solid #22c55e;
            box-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
        }}
        
        .alert-content {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}
        
        .alert-emoji {{
            font-size: 2.5rem;
        }}
        
        .alert-title {{
            font-size: 1.25rem;
            font-weight: bold;
        }}
        
        .alert-subtitle {{
            color: #d1d5db;
            margin-top: 4px;
        }}
        
        .severity-badge {{
            padding: 8px 20px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1rem;
        }}
        
        .severity-badge.critical {{
            background: #dc2626;
        }}
        
        .severity-badge.warning {{
            background: #d97706;
        }}
        
        .severity-badge.ok {{
            background: #16a34a;
        }}
        
        /* Grid Layout */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        @media (max-width: 1024px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Cards */
        .card {{
            background: #1f2937;
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #374151;
        }}
        
        .card-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .card-subtitle {{
            font-size: 0.875rem;
            color: #9ca3af;
            margin-bottom: 16px;
        }}
        
        /* Metric Cards */
        .metric-cards {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
        }}
        
        .metric-card {{
            padding: 16px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid;
        }}
        
        .metric-card.critical {{
            background: rgba(127, 29, 29, 0.5);
            border-color: #ef4444;
        }}
        
        .metric-card.warning {{
            background: rgba(120, 53, 15, 0.5);
            border-color: #f59e0b;
        }}
        
        .metric-card.ok {{
            background: rgba(20, 83, 45, 0.5);
            border-color: #22c55e;
        }}
        
        .metric-label {{
            font-size: 0.75rem;
            color: #9ca3af;
            margin-bottom: 4px;
        }}
        
        .metric-value {{
            font-size: 1.75rem;
            font-weight: bold;
        }}
        
        .metric-card.critical .metric-value {{
            color: #f87171;
        }}
        
        .metric-card.warning .metric-value {{
            color: #fbbf24;
        }}
        
        .metric-card.ok .metric-value {{
            color: #4ade80;
        }}
        
        .metric-baseline {{
            font-size: 0.75rem;
            color: #6b7280;
            margin-top: 4px;
        }}
        
        .metric-status {{
            margin-top: 8px;
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
            display: inline-block;
        }}
        
        .metric-card.critical .metric-status {{
            background: #dc2626;
        }}
        
        .metric-card.warning .metric-status {{
            background: #d97706;
        }}
        
        .metric-card.ok .metric-status {{
            background: #16a34a;
        }}
        
        /* Progress Bars */
        .progress-container {{
            margin-bottom: 16px;
        }}
        
        .progress-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}
        
        .progress-label {{
            width: 100px;
            font-size: 0.875rem;
            color: #9ca3af;
        }}
        
        .progress-bar {{
            flex: 1;
            height: 24px;
            background: #374151;
            border-radius: 12px;
            position: relative;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 12px;
            transition: width 0.5s ease;
        }}
        
        .progress-fill.baseline {{
            background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
        }}
        
        .progress-fill.current {{
            background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%);
        }}
        
        .progress-fill.threshold {{
            background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
        }}
        
        .progress-value {{
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.875rem;
            font-weight: bold;
        }}
        
        .degradation-box {{
            margin-top: 16px;
            padding: 12px 16px;
            background: rgba(127, 29, 29, 0.3);
            border: 1px solid #b91c1c;
            border-radius: 8px;
        }}
        
        .degradation-box span {{
            color: #f87171;
            font-weight: bold;
        }}
        
        .degradation-box .detail {{
            color: #9ca3af;
            margin-left: 8px;
        }}
        
        /* Chart Container */
        .chart-container {{
            height: 250px;
            position: relative;
        }}
        
        /* Recommendations */
        .recommendations {{
            background: #1f2937;
            border-radius: 12px;
            padding: 20px;
            border-left: 4px solid #eab308;
            margin-top: 20px;
        }}
        
        .recommendations h3 {{
            color: #fbbf24;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .recommendations ul {{
            list-style: none;
        }}
        
        .recommendations li {{
            padding: 6px 0;
            color: #d1d5db;
        }}
        
        .recommendations li strong {{
            color: #f3f4f6;
        }}
        
        /* Drift Reasons */
        .drift-reasons {{
            background: rgba(127, 29, 29, 0.2);
            border: 1px solid #7f1d1d;
            border-radius: 8px;
            padding: 16px;
            margin-top: 20px;
        }}
        
        .drift-reasons h4 {{
            color: #f87171;
            margin-bottom: 12px;
        }}
        
        .drift-reasons ul {{
            list-style: disc;
            margin-left: 20px;
        }}
        
        .drift-reasons li {{
            padding: 4px 0;
            color: #fca5a5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>[SEARCH] Model Drift Detection Dashboard</h1>
        <div class="header-info">
            Test Year: {test_year} | Baseline Year: {baseline_year} | Generated: {timestamp}
        </div>
        
        <!-- Alert Banner -->
        <div class="alert-banner {alert_banner_class}">
            <div class="alert-content">
                <span class="alert-emoji">{status_emoji}</span>
                <div>
                    <div class="alert-title">{status_text}</div>
                    <div class="alert-subtitle">
                        R² dropped by {r2_drop:.2f} | MAE increased by {mae_increase_pct:.0f}%
                    </div>
                </div>
            </div>
            <div class="severity-badge {severity_badge_class}">{severity}</div>
        </div>
        
        <div class="dashboard-grid">
            <!-- Performance Over Time -->
            <div class="card">
                <div class="card-title">[CHART] R² Score Over Time</div>
                <div class="card-subtitle">Track performance degradation trend</div>
                <div class="chart-container">
                    <canvas id="timeSeriesChart"></canvas>
                </div>
            </div>
            
            <!-- Threshold Cards -->
            <div class="card">
                <div class="card-title"> Metric Status Cards</div>
                <div class="card-subtitle">Instant pass/fail at a glance</div>
                <div class="metric-cards">
                    <div class="metric-card {r2_card_class}">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">{current_r2:.2f}</div>
                        <div class="metric-baseline">threshold: {performance_threshold}</div>
                        <div class="metric-status">{r2_status}</div>
                    </div>
                    <div class="metric-card {mae_card_class}">
                        <div class="metric-label">MAE</div>
                        <div class="metric-value">{current_mae:,.0f}</div>
                        <div class="metric-baseline">baseline: {baseline_mae:,.0f}</div>
                        <div class="metric-status">{mae_status}</div>
                    </div>
                    <div class="metric-card {rmse_card_class}">
                        <div class="metric-label">RMSE</div>
                        <div class="metric-value">{current_rmse:,.0f}</div>
                        <div class="metric-baseline">baseline: {baseline_rmse:,.0f}</div>
                        <div class="metric-status">{rmse_status}</div>
                    </div>
                </div>
            </div>
            
            <!-- Baseline vs Current Comparison -->
            <div class="card">
                <div class="card-title">[METRICS] Baseline vs Current</div>
                <div class="card-subtitle">Side-by-side metric comparison</div>
                <div class="chart-container">
                    <canvas id="comparisonChart"></canvas>
                </div>
            </div>
            
            <!-- R² Degradation Breakdown -->
            <div class="card">
                <div class="card-title"> R² Degradation Breakdown</div>
                <div class="card-subtitle">Visualize exactly how much performance was lost</div>
                <div class="progress-container">
                    <div class="progress-row">
                        <div class="progress-label">Baseline R²</div>
                        <div class="progress-bar">
                            <div class="progress-fill baseline" style="width: {baseline_r2_width}%"></div>
                            <span class="progress-value">{baseline_r2:.3f}</span>
                        </div>
                    </div>
                    <div class="progress-row">
                        <div class="progress-label">Current R²</div>
                        <div class="progress-bar">
                            <div class="progress-fill current" style="width: {current_r2_width}%"></div>
                            <span class="progress-value" style="color: #f87171; left: 12px; right: auto;">{current_r2:.3f}</span>
                        </div>
                    </div>
                    <div class="progress-row">
                        <div class="progress-label">Min Threshold</div>
                        <div class="progress-bar">
                            <div class="progress-fill threshold" style="width: {threshold_r2_width}%"></div>
                            <span class="progress-value">{performance_threshold}</span>
                        </div>
                    </div>
                </div>
                <div class="degradation-box">
                    <span>Total Degradation: -{r2_drop:.2f}</span>
                    <span class="detail">({drift_multiplier:.1f}x over drift threshold of {drift_threshold})</span>
                </div>
            </div>
        </div>
        
        <!-- Drift Reasons -->
        <div class="drift-reasons" style="display: {drift_reasons_display}">
            <h4>[WARNING] Drift Reasons</h4>
            <ul>
                {drift_reasons_html}
            </ul>
        </div>
        
        <!-- Recommendations -->
        <div class="recommendations">
            <h3>[TIP] Recommended Actions</h3>
            <ul>
                <li>• <strong>Primary:</strong> Time series chart catches drift EARLY - monitor weekly</li>
                <li>• <strong>If CRITICAL:</strong> Retrain model immediately with recent data</li>
                <li>• <strong>If WARNING:</strong> Investigate root cause, plan retraining</li>
                <li>• <strong>After retrain:</strong> Update baseline metrics</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Time Series Chart
        const timeSeriesData = {metrics_history_json};
        
        new Chart(document.getElementById('timeSeriesChart'), {{
            type: 'line',
            data: {{
                labels: timeSeriesData.map(d => d.period),
                datasets: [{{
                    label: 'R² Score',
                    data: timeSeriesData.map(d => d.r2),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 3,
                    pointRadius: 6,
                    pointBackgroundColor: timeSeriesData.map(d => 
                        d.status === 'critical' ? '#ef4444' : 
                        d.status === 'warning' ? '#f59e0b' : '#3b82f6'
                    ),
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e4e4e7' }}
                    }}
                }},
                scales: {{
                    y: {{
                        min: -0.5,
                        max: 1,
                        grid: {{ color: '#374151' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    x: {{
                        grid: {{ color: '#374151' }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }},
                annotation: {{
                    annotations: {{
                        threshold: {{
                            type: 'line',
                            yMin: {performance_threshold},
                            yMax: {performance_threshold},
                            borderColor: '#f59e0b',
                            borderWidth: 2,
                            borderDash: [5, 5]
                        }}
                    }}
                }}
            }}
        }});
        
        // Comparison Bar Chart
        new Chart(document.getElementById('comparisonChart'), {{
            type: 'bar',
            data: {{
                labels: ['R²', 'MAE (÷1000)', 'RMSE (÷1000)'],
                datasets: [
                    {{
                        label: 'Baseline',
                        data: [{baseline_r2}, {baseline_mae}/1000, {baseline_rmse}/1000],
                        backgroundColor: '#22c55e',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Current',
                        data: [{current_r2}, {current_mae}/1000, {current_rmse}/1000],
                        backgroundColor: '#ef4444',
                        borderRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#e4e4e7' }}
                    }}
                }},
                scales: {{
                    y: {{
                        grid: {{ color: '#374151' }},
                        ticks: {{ color: '#9ca3af' }}
                    }},
                    x: {{
                        grid: {{ color: '#374151' }},
                        ticks: {{ color: '#9ca3af' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
'''
