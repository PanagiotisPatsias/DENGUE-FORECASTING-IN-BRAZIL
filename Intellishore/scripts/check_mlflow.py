"""
check mlflow experiments and runs.
"""

import mlflow
from pathlib import Path

# set tracking uri (use file:// prefix for Windows)
tracking_uri = Path(__file__).parent.parent / "mlruns"
mlflow.set_tracking_uri(f"file:///{tracking_uri}")

print("\n" + "="*60)
print("MLFLOW DATA CHECK")
print("="*60)

# get client
client = mlflow.tracking.MlflowClient()

# list experiments
experiments = client.search_experiments()
print(f"\n Found {len(experiments)} experiment(s):")

for exp in experiments:
    print(f"\n  Experiment: {exp.name}")
    print(f"  ID: {exp.experiment_id}")
    
    # get runs for this experiment
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    print(f"  Runs: {len(runs)}")
    
    if runs:
        print(f"\n  Run Details:")
        for i, run in enumerate(runs[:10], 1):  # show first 10
            print(f"    {i}. {run.info.run_name}")
            print(f"       Status: {run.info.status}")
            print(f"       Metrics: R²={run.data.metrics.get('r2', 'N/A')}")
            print(f"       Started: {run.info.start_time}")

# check registered models
print("\n" + "="*60)
print("REGISTERED MODELS")
print("="*60)

models = client.search_registered_models()
print(f"\n Found {len(models)} registered model(s):")

for model in models:
    print(f"\n  Model: {model.name}")
    latest_versions = client.get_latest_versions(model.name)
    print(f"  Versions: {len(latest_versions)}")
    for version in latest_versions:
        print(f"    - Version {version.version} ({version.current_stage})")

print("\n" + "="*60)
print(f"MLflow Tracking URI: {tracking_uri}")
print(f"MLflow UI: http://localhost:5000")
print("="*60 + "\n")
