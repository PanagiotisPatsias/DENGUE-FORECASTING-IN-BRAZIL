"""quick check for drift detection runs in mlflow"""
import mlflow

client = mlflow.tracking.MlflowClient('file:///./mlruns')
runs = client.search_runs('829956913424739213', max_results=10)

drift_runs = [r for r in runs if 'drift_check' in r.info.run_name]

print(f'\nFound {len(drift_runs)} drift check run(s):')
for r in drift_runs:
    drift_detected = r.data.tags.get('drift_detected', 'N/A')
    test_year = r.data.tags.get('test_year', 'N/A')
    r2_deg = r.data.metrics.get('r2_degradation', 'N/A')
    print(f'\n  Run: {r.info.run_name}')
    print(f'    - Drift Detected: {drift_detected}')
    print(f'    - Test Year: {test_year}')
    print(f'    - R² Degradation: {r2_deg}')
    print(f'    - Run ID: {r.info.run_id}')
