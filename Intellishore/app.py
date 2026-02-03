"""
streamlit web application for dengue forecasting.
simplified ui: upload csv, run prediction, download report.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import io
import tempfile
from pathlib import Path
import pickle

# add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.data_loader import DataLoader
from src.core.feature_engineer import FeatureEngineer
from src.core.model_trainer import ModelTrainer
from src.core.forecaster import Forecaster
from src.utils.model_manager import ModelManager
from src.monitoring.model_monitor import ModelMonitor
from src.utils.config import Config

# page configuration
st.set_page_config(
    page_title="Dengue Forecasting - Simple",
    page_icon="[DENGUE]",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# custom css
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        color: #D7263D;
        font-weight: 700;
        text-align: center;
        padding: 0.75rem 0 0.25rem 0;
        letter-spacing: 0.3px;
    }
    .subtext {
        text-align: center;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .panel {
        background: #f7f7fb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_training_data():
    """load the base dataset once for training (default files)."""
    data_loader = DataLoader()
    return data_loader.load_and_prepare_data()


def _read_uploaded_csv(uploaded):
    """read uploaded CSV from UploadedFile, cached bytes, or a file path."""
    if uploaded is None:
        raise ValueError("Missing uploaded file")
    if isinstance(uploaded, str) and os.path.exists(uploaded):
        return pd.read_csv(uploaded)
    if isinstance(uploaded, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(uploaded))
    return pd.read_csv(uploaded)


def _load_uploaded_pickle(uploaded):
    if uploaded is None:
        raise ValueError("Missing uploaded .pkl file")
    try:
        return pickle.load(uploaded)
    except Exception as exc:
        raise ValueError(f"Could not read uploaded .pkl: {exc}") from exc


def prepare_training_data_from_uploads(dengue_file, sst_file):
    """prepare merged quarterly training data from uploaded dengue + SST CSVs."""
    data_loader = DataLoader()

    try:
        dengue_df = _read_uploaded_csv(dengue_file)
        sst_df = _read_uploaded_csv(sst_file)
    except Exception as exc:
        raise ValueError(f"Could not read uploaded files: {exc}") from exc

    required_dengue_cols = {"data_iniSE", "casos_est"}
    if not required_dengue_cols.issubset(dengue_df.columns):
        missing = required_dengue_cols - set(dengue_df.columns)
        raise ValueError(f"Dengue file missing columns: {', '.join(sorted(missing))}")

    required_sst_cols = {"YR", "MON", "NINO1+2", "NINO3", "NINO3.4", "ANOM.3"}
    if not required_sst_cols.issubset(sst_df.columns):
        missing = required_sst_cols - set(sst_df.columns)
        raise ValueError(f"SST file missing columns: {', '.join(sorted(missing))}")

    # Dengue: derive year/quarter
    dengue_df["data_iniSE"] = pd.to_datetime(dengue_df["data_iniSE"])
    dengue_df["year"] = dengue_df["data_iniSE"].dt.year
    dengue_df["quarter"] = dengue_df["data_iniSE"].dt.quarter
    dengue_df["year_quarter"] = dengue_df["year"].astype(str) + "-Q" + dengue_df["quarter"].astype(str)

    # SST: derive year/quarter
    sst_df["date"] = pd.to_datetime(
        sst_df["YR"].astype(str) + "-" + sst_df["MON"].astype(str) + "-01"
    )
    sst_df["quarter"] = sst_df["date"].dt.quarter
    sst_df["year"] = sst_df["YR"]
    sst_df["year_quarter"] = sst_df["year"].astype(str) + "-Q" + sst_df["quarter"].astype(str)

    sst_quarterly = data_loader.aggregate_sst_quarterly(sst_df)
    dengue_quarterly = data_loader.aggregate_dengue_quarterly(dengue_df)

    return data_loader.merge_datasets(dengue_quarterly, sst_quarterly)


def run_full_pipeline(df_base=None):
    """run the same steps as src/main.py: evaluate 2023, evaluate 2025, forecast 2026.  """
    if df_base is None:
        df_base = load_training_data()
    feature_engineer = FeatureEngineer()
    df_feat, feature_cols = feature_engineer.create_features(df_base)

    model_trainer = ModelTrainer()

    # 2023 normal year
    train_years_2023 = list(range(2010, 2023))
    results_2023, y_test_2023, test_df_2023, valid_features_2023, _ = (
        model_trainer.train_and_evaluate(
            df_feat, feature_cols, train_years_2023, 2023
        )
    )
    best_name_2023, best_res_2023 = model_trainer.get_best_model(results_2023)

    # 2025 post-outbreak year (exclude 2024)
    train_years_2025 = [y for y in range(2010, 2025) if y != 2024]
    results_2025, y_test_2025, test_df_2025, valid_features_2025, _ = (
        model_trainer.train_and_evaluate(
            df_feat, feature_cols, train_years_2025, 2025
        )
    )
    best_name_2025, best_res_2025 = model_trainer.get_best_model(results_2025)

    # 2026 forecast using best 2023 model
    forecaster = Forecaster(feature_engineer)
    forecast_2026_df, fitted_model, forecast_features = forecaster.refit_and_forecast(
        df_feat,
        feature_cols,
        best_res_2023["model"],
        forecast_year=2026,
        train_max_year=2025,
        exclude_years=[2024],
    )

    # save baseline model (2023) for reproducibility
    model_manager = ModelManager()
    model_manager.save_baseline_model(
        model=best_res_2023["model"],
        model_name=best_name_2023,
        features=valid_features_2023,
        metrics={
            "r2": best_res_2023["r2"],
            "mae": best_res_2023["mae"],
            "rmse": best_res_2023["rmse"],
        },
        test_year=2023,
        train_years=train_years_2023,
    )

    return {
        "feature_engineer": feature_engineer,
        "results_2023": results_2023,
        "results_2025": results_2025,
        "best_2023": {
            "name": best_name_2023,
            "res": best_res_2023,
            "valid_features": valid_features_2023,
            "train_years": train_years_2023,
            "test_df": test_df_2023,
            "y_test": y_test_2023,
        },
        "best_2025": {
            "name": best_name_2025,
            "res": best_res_2025,
            "valid_features": valid_features_2025,
            "train_years": train_years_2025,
            "test_df": test_df_2025,
            "y_test": y_test_2025,
        },
        "forecast_2026": {
            "values": forecast_2026_df,
            "fitted_model": fitted_model,
            "features": forecast_features,
        },
    }


def build_report_text(metadata, df_in, predictions, metrics=None, pipeline_summary=None):
    """build a simple report as plain text."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = len(df_in)
    cols = len(df_in.columns)

    year_min = df_in["year"].min() if "year" in df_in.columns else "N/A"
    year_max = df_in["year"].max() if "year" in df_in.columns else "N/A"

    report_lines = [
        "=" * 72,
        "DENGUE FORECASTING - PREDICTION REPORT",
        "=" * 72,
        f"Generated: {now}",
        f"Model: {metadata['model_name']}",
        f"Rows: {rows}",
        f"Columns: {cols}",
        f"Year Range: {year_min} to {year_max}",
        "",
        "PREDICTION SUMMARY",
        f"Predictions: {len(predictions)} rows",
    ]

    if metrics is not None:
        report_lines += [
            "",
            "ACCURACY (when actuals provided)",
            f"R2:   {metrics['r2']:.4f}",
            f"MAE:  {metrics['mae']:,.2f}",
            f"RMSE: {metrics['rmse']:,.2f}",
        ]

    if pipeline_summary is not None:
        report_lines += [
            "",
            "TRAINING SUMMARY",
            f"2023 Best Model: {pipeline_summary['best_2023_name']}",
            f"2023 R2: {pipeline_summary['best_2023_r2']:.4f}",
            f"2023 MAE: {pipeline_summary['best_2023_mae']:,.2f}",
            f"2025 Best Model: {pipeline_summary['best_2025_name']}",
            f"2025 R2: {pipeline_summary['best_2025_r2']:.4f}",
            f"2025 MAE: {pipeline_summary['best_2025_mae']:,.2f}",
            "",
            "FORECAST 2026 (QUARTERS)",
        ]
        for i, val in enumerate(pipeline_summary["forecast_2026"], start=1):
            report_lines.append(f"Q{i}: {val:,.0f}")

    report_lines.append("=" * 72)

    return "\n".join(report_lines)


def apply_fast_grid_settings(enable: bool) -> None:
    """Shrink grid + CV splits for faster cloud runs."""
    # always restore defaults before applying any trimming
    Config.ENABLE_GRID_SEARCH = Config.DEFAULT_ENABLE_GRID_SEARCH
    Config.TSCV_SPLITS = Config.DEFAULT_TSCV_SPLITS
    Config.PARAM_GRIDS = {k: v.copy() for k, v in Config.DEFAULT_PARAM_GRIDS.items()}

    if not enable:
        return
    Config.TSCV_SPLITS = 2
    trimmed = {}
    for model_name, grid in Config.PARAM_GRIDS.items():
        trimmed_grid = {}
        for param_name, values in grid.items():
            if isinstance(values, (list, tuple)) and values:
                trimmed_grid[param_name] = [values[0]]
            else:
                trimmed_grid[param_name] = values
        trimmed[model_name] = trimmed_grid
    Config.PARAM_GRIDS = trimmed


def main():
    st.markdown('<div class="main-header">Dengue Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Upload a CSV, run prediction, and download a report.</div>', unsafe_allow_html=True)

    is_cloud = os.getenv("CLOUD_MODE", "").lower() in ("1", "true", "yes")
    is_cloud = is_cloud or os.getenv("STREAMLIT_CLOUD", "").lower() in ("1", "true", "yes")

    if is_cloud:
        st.markdown("<div class='panel'>", unsafe_allow_html=True)
        st.subheader("Cloud Mode: Predict Only")
        st.caption("Upload 2 .pkl files and run prediction. No training is executed in cloud mode.")

        uploaded_model = st.file_uploader("Upload model .pkl", type=["pkl"], key="cloud_model_upload")
        uploaded_meta = st.file_uploader("Upload metadata .pkl (optional)", type=["pkl"], key="cloud_meta_upload")

        if uploaded_model is None:
            st.info("Upload the model .pkl to proceed.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Running prediction..."):
                try:
                    model = _load_uploaded_pickle(uploaded_model)
                    metadata = _load_uploaded_pickle(uploaded_meta) if uploaded_meta else {}

                    df_base = load_training_data()
                    feature_engineer = FeatureEngineer()
                    df_feat, feature_cols = feature_engineer.create_features(df_base)

                    forecaster = Forecaster(feature_engineer)
                    forecast_raw, valid_features = forecaster.forecast_with_fitted_model(
                        df_feat,
                        feature_cols,
                        model,
                        forecast_year=2026,
                        train_max_year=2025,
                        exclude_years=[2024],
                        feature_subset=metadata.get("features"),
                    )

                    st.success("[OK] Forecast completed.")
                    forecast_df = forecast_raw.rename(
                        columns={
                            "year_quarter": "Quarter",
                            "predicted_casos_est": "Forecast",
                        }
                    )
                    st.dataframe(forecast_df, use_container_width=True)

                    report_text = build_report_text(
                        {
                            "model_name": metadata.get("model_name", "uploaded_model"),
                            "features": valid_features,
                            "metrics": metadata.get("metrics", {}),
                            "test_year": metadata.get("test_year", "N/A"),
                            "train_years": metadata.get("train_years", []),
                            "num_features": len(valid_features),
                        },
                        df_base,
                        forecast_df["Forecast"].tolist(),
                        metrics=None,
                        pipeline_summary={
                            "best_2023_name": metadata.get("model_name", "uploaded_model"),
                            "best_2023_r2": metadata.get("metrics", {}).get("r2", 0.0),
                            "best_2023_mae": metadata.get("metrics", {}).get("mae", 0.0),
                            "best_2025_name": "N/A",
                            "best_2025_r2": 0.0,
                            "best_2025_mae": 0.0,
                            "forecast_2026": forecast_df["Forecast"].tolist(),
                        },
                    )

                    st.subheader("Download")
                    st.download_button(
                        "Download forecast_2026.csv",
                        forecast_df.to_csv(index=False),
                        "forecast_2026.csv",
                        "text/csv",
                    )
                    st.download_button(
                        "Download report.txt",
                        report_text,
                        f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain",
                    )
                except Exception as exc:
                    st.error(f"[ERROR] Prediction failed: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("0) Training Plan")
    st.caption("This follows src/main.py: evaluate 2023, evaluate 2025 (exclude 2024), forecast 2026.")
    st.write("Training years: 2010-2022 (test 2023), and 2010-2023 + 2025 (test 2025).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("1) Training Data Source")
    use_bundled = st.checkbox(
        "Use bundled training data (recommended for Streamlit Cloud)",
        value=True,
        help="Loads data from the repo instead of uploads to avoid session resets.",
    )

    if not use_bundled:
        st.caption("Dengue requires: `data_iniSE`, `casos_est`. SST requires: `YR`, `MON`, `NINO1+2`, `NINO3`, `NINO3.4`, `ANOM.3`.")
        dengue_file = st.file_uploader("Upload Dengue CSV", type=["csv"], key="dengue_file")
        sst_file = st.file_uploader("Upload SST CSV", type=["csv"], key="sst_file")

        # persist uploads across reruns (store bytes + temp file for stability)
        tmp_dir = tempfile.gettempdir()
        dengue_tmp = os.path.join(tmp_dir, "dengue_upload.csv")
        sst_tmp = os.path.join(tmp_dir, "sst_upload.csv")

        if dengue_file is not None:
            dengue_bytes = dengue_file.read()
            st.session_state["dengue_file_cached"] = dengue_bytes
            with open(dengue_tmp, "wb") as f:
                f.write(dengue_bytes)
            st.session_state["dengue_file_path"] = dengue_tmp

        if sst_file is not None:
            sst_bytes = sst_file.read()
            st.session_state["sst_file_cached"] = sst_bytes
            with open(sst_tmp, "wb") as f:
                f.write(sst_bytes)
            st.session_state["sst_file_path"] = sst_tmp

        dengue_file = st.session_state.get("dengue_file_cached") or st.session_state.get("dengue_file_path")
        sst_file = st.session_state.get("sst_file_cached") or st.session_state.get("sst_file_path")
    else:
        dengue_file = None
        sst_file = None
        st.caption(f"Using bundled data files: `{Config.DENGUE_DATA_PATH}`, `{Config.SST_DATA_PATH}`.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not use_bundled and (dengue_file is None or sst_file is None):
        st.info("Upload both training files to proceed.")
        st.stop()

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("2) Train + Forecast (Full Pipeline)")

    predict_only = False
    if use_bundled:
        predict_only = st.checkbox(
            "Predict-only mode (use saved baseline model)",
            value=True if is_cloud else False,
            help="Skips training/validation in the cloud and only runs the forecast.",
        )
    if is_cloud:
        predict_only = True

    uploaded_model = None
    uploaded_meta = None
    if predict_only:
        st.caption("Optional: upload baseline model files (baseline_model.pkl + baseline_metadata.json).")
        uploaded_model = st.file_uploader("Upload baseline_model.pkl", type=["pkl"], key="baseline_model_upload")
        uploaded_meta = st.file_uploader("Upload baseline_metadata.json", type=["json"], key="baseline_meta_upload")

    log_to_mlflow = False
    if not is_cloud:
        log_to_mlflow = st.checkbox(
            "Log results to MLflow",
            value=True,
            help="Uses MLFLOW_TRACKING_URI if set; otherwise logs locally to ./mlruns.",
        )
    else:
        st.caption("MLflow logging is disabled in cloud mode.")

    if st.button("Run Full Pipeline + Forecast", type="primary"):
        with st.spinner("Running training pipeline and generating forecast..."):
            try:
                apply_fast_grid_settings(enable=False)
                if use_bundled:
                    df_base = load_training_data()
                else:
                    df_base = prepare_training_data_from_uploads(dengue_file, sst_file)
                if predict_only:
                    # Resolve models directory for cloud (repo root vs Intellishore subdir)
                    temp_models_dir = None
                    if uploaded_model is not None:
                        temp_models_dir = Path(tempfile.mkdtemp(prefix="uploaded_models_"))
                        model_path = temp_models_dir / "baseline_model.pkl"
                        with open(model_path, "wb") as f:
                            f.write(uploaded_model.read())
                        if uploaded_meta is not None:
                            meta_path = temp_models_dir / "baseline_metadata.json"
                            with open(meta_path, "wb") as f:
                                f.write(uploaded_meta.read())

                    candidates = [
                        temp_models_dir,
                        Path.cwd() / "Intellishore" / "models",
                        Path(__file__).parent / "models",
                        Path.cwd() / "models",
                    ]
                    candidates = [p for p in candidates if p is not None]
                    models_dir = next(
                        (p for p in candidates if (p / "baseline_model.pkl").exists()),
                        next((p for p in candidates if p.exists()), candidates[0]),
                    )
                    model_manager = ModelManager(models_dir=str(models_dir))
                    st.caption(f"Using models dir: `{models_dir}`")
                    # Debug listing for cloud visibility
                    if is_cloud:
                        try:
                            if models_dir.exists():
                                files = sorted([f.name for f in models_dir.iterdir() if f.is_file()])
                                st.write(f"[DEBUG] Files in models dir: {', '.join(files) if files else '(empty)'}")
                            else:
                                st.write(f"[DEBUG] Models dir missing: `{models_dir}`")
                        except Exception as exc:
                            st.write(f"[DEBUG] Could not list models dir: {exc}")
                    loaded = model_manager.load_baseline_model()
                    if not loaded:
                        st.warning("[DEBUG] Baseline not found. Checked these paths:")
                        for p in candidates:
                            if p.exists():
                                files = sorted([f.name for f in p.glob("*.pkl")] + [f.name for f in p.glob("*.json")])
                                st.write(f"- `{p}`: {', '.join(files) if files else '(no model files)'}")
                            else:
                                st.write(f"- `{p}`: (missing)")
                        st.error("[ERROR] No baseline model found. Train locally to create baseline_model.pkl.")
                        return
                    baseline_model, baseline_meta = loaded

                    feature_engineer = FeatureEngineer()
                    df_feat, feature_cols = feature_engineer.create_features(df_base)
                    forecaster = Forecaster(feature_engineer)

                    forecast_raw, valid_features = forecaster.forecast_with_fitted_model(
                        df_feat,
                        feature_cols,
                        baseline_model,
                        forecast_year=2026,
                        train_max_year=2025,
                        exclude_years=[2024],
                        feature_subset=baseline_meta.get("features"),
                    )

                    st.success("[OK] Forecast completed (predict-only).")

                    st.subheader("Forecast 2026")
                    forecast_vals = forecast_raw["predicted_casos_est"].tolist()
                    forecast_df = forecast_raw.rename(
                        columns={
                            "year_quarter": "Quarter",
                            "predicted_casos_est": "Forecast",
                        }
                    )
                    st.dataframe(forecast_df, use_container_width=True)

                    metadata = {
                        "model_name": baseline_meta.get("model_name", "baseline_model"),
                        "features": valid_features,
                        "metrics": baseline_meta.get("metrics", {}),
                        "test_year": baseline_meta.get("test_year", "N/A"),
                        "train_years": baseline_meta.get("train_years", []),
                        "num_features": len(valid_features),
                    }

                    report_text = build_report_text(
                        metadata,
                        df_base,
                        forecast_vals,
                        metrics=None,
                        pipeline_summary={
                            "best_2023_name": metadata["model_name"],
                            "best_2023_r2": metadata.get("metrics", {}).get("r2", 0.0),
                            "best_2023_mae": metadata.get("metrics", {}).get("mae", 0.0),
                            "best_2025_name": "N/A",
                            "best_2025_r2": 0.0,
                            "best_2025_mae": 0.0,
                            "forecast_2026": forecast_vals,
                        },
                    )

                    if log_to_mlflow:
                        try:
                            monitor = ModelMonitor(experiment_name="dengue_forecasting")
                            st.caption(f"MLflow tracking URI: `{monitor.tracking_uri}`")
                            monitor.log_forecast_run(
                                model_name=metadata["model_name"],
                                forecast_df=forecast_df,
                                report_text=report_text,
                                params={"predict_only": True},
                                metrics=metadata.get("metrics", {}),
                            )
                            st.success("[OK] Logged forecast run to MLflow.")
                        except Exception as exc:
                            st.warning(f"[WARNING] MLflow logging failed: {exc}")
                else:
                    pipeline = run_full_pipeline(df_base=df_base)

                    results_2023 = pipeline["results_2023"]
                    results_2025 = pipeline["results_2025"]
                    best_2023 = pipeline["best_2023"]
                    best_2025 = pipeline["best_2025"]

                    st.success("[OK] Training + forecast completed.")

                    st.subheader("Training Evaluation (2023)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R2", f"{best_2023['res']['r2']:.4f}")
                    col2.metric("MAE", f"{best_2023['res']['mae']:,.0f}")
                    col3.metric("RMSE", f"{best_2023['res']['rmse']:,.0f}")

                    st.subheader("Training Evaluation (2025)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("R2", f"{best_2025['res']['r2']:.4f}")
                    col2.metric("MAE", f"{best_2025['res']['mae']:,.0f}")
                    col3.metric("RMSE", f"{best_2025['res']['rmse']:,.0f}")

                    st.subheader("Forecast 2026")
                    forecast_raw = pipeline["forecast_2026"]["values"]
                    forecast_vals = forecast_raw["predicted_casos_est"].tolist()
                    forecast_df = forecast_raw.rename(
                        columns={
                            "year_quarter": "Quarter",
                            "predicted_casos_est": "Forecast",
                        }
                    )
                    st.dataframe(forecast_df, use_container_width=True)

                    if not is_cloud:
                        st.subheader("MLflow")
                        mlflow_url = os.getenv(
                            "MLFLOW_UI_URL",
                            "http://localhost:5000",
                        )
                        st.link_button("Open MLflow", mlflow_url)

                    if log_to_mlflow:
                        try:
                            monitor = ModelMonitor(experiment_name="dengue_forecasting")
                            st.caption(f"MLflow tracking URI: `{monitor.tracking_uri}`")

                            def _params_for(model_name: str, model_res: dict):
                                return model_res.get(
                                    "best_params",
                                    Config.RANDOM_FOREST_PARAMS if model_name == "RandomForest"
                                    else Config.GRADIENT_BOOSTING_PARAMS if model_name == "GradientBoosting"
                                    else Config.ADABOOST_PARAMS if model_name == "AdaBoost"
                                    else {},
                                )

                            logged = 0

                            for model_name, model_res in results_2023.items():
                                try:
                                    metrics = {
                                        "r2": model_res["r2"],
                                        "mae": model_res["mae"],
                                        "rmse": model_res["rmse"],
                                    }
                                    if "val_mae" in model_res:
                                        metrics["val_mae"] = model_res["val_mae"]

                                    monitor.log_training_run(
                                        model=model_res["model"],
                                        model_name=model_name,
                                        params=_params_for(model_name, model_res),
                                        metrics=metrics,
                                        features=best_2023["valid_features"],
                                        train_year_range=(2010, 2022),
                                        test_year=2023,
                                    )
                                    logged += 1
                                except Exception as exc:
                                    st.warning(f"[WARNING] MLflow log failed for {model_name} 2023: {exc}")

                            for model_name, model_res in results_2025.items():
                                try:
                                    metrics = {
                                        "r2": model_res["r2"],
                                        "mae": model_res["mae"],
                                        "rmse": model_res["rmse"],
                                    }
                                    if "val_mae" in model_res:
                                        metrics["val_mae"] = model_res["val_mae"]

                                    monitor.log_training_run(
                                        model=model_res["model"],
                                        model_name=model_name,
                                        params=_params_for(model_name, model_res),
                                        metrics=metrics,
                                        features=best_2025["valid_features"],
                                        train_year_range=(2010, 2024),
                                        test_year=2025,
                                        artifacts={"excluded_years": [2024]},
                                    )
                                    logged += 1
                                except Exception as exc:
                                    st.warning(f"[WARNING] MLflow log failed for {model_name} 2025: {exc}")

                            try:
                                monitor.set_baseline(
                                    metrics={
                                        "r2": best_2023["res"]["r2"],
                                        "mae": best_2023["res"]["mae"],
                                        "rmse": best_2023["res"]["rmse"],
                                    },
                                    model_name=best_2023["name"],
                                    test_year=2023,
                                )
                            except Exception as exc:
                                st.warning(f"[WARNING] MLflow baseline set failed: {exc}")

                            st.success(f"[OK] Logged {logged} MLflow run(s).")
                        except Exception as exc:
                            st.warning(f"[WARNING] MLflow logging failed: {exc}")

                    metadata = {
                        "model_name": best_2023["name"],
                        "features": best_2023["valid_features"],
                        "metrics": {
                            "r2": best_2023["res"]["r2"],
                            "mae": best_2023["res"]["mae"],
                            "rmse": best_2023["res"]["rmse"],
                        },
                        "test_year": 2023,
                        "train_years": best_2023["train_years"],
                        "num_features": len(best_2023["valid_features"]),
                    }

                    report_text = build_report_text(
                        metadata,
                        df_base,
                        forecast_vals,
                        metrics=None,
                        pipeline_summary={
                            "best_2023_name": best_2023["name"],
                            "best_2023_r2": best_2023["res"]["r2"],
                            "best_2023_mae": best_2023["res"]["mae"],
                            "best_2025_name": best_2025["name"],
                            "best_2025_r2": best_2025["res"]["r2"],
                            "best_2025_mae": best_2025["res"]["mae"],
                            "forecast_2026": forecast_vals,
                        },
                    )

                st.subheader("Download")
                st.download_button(
                    "Download forecast_2026.csv",
                    forecast_df.to_csv(index=False),
                    "forecast_2026.csv",
                    "text/csv",
                )
                st.download_button(
                    "Download report.txt",
                    report_text,
                    f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                )

            except Exception as exc:
                st.error(f"[ERROR] Training/prediction failed: {exc}")

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
