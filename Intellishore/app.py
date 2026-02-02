"""
streamlit web application for dengue forecasting.
simplified ui: upload csv, run prediction, download report.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.data_loader import DataLoader
from src.core.feature_engineer import FeatureEngineer
from src.core.model_trainer import ModelTrainer
from src.core.forecaster import Forecaster
from src.utils.model_manager import ModelManager

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


def prepare_training_data_from_uploads(dengue_file, sst_file):
    """prepare merged quarterly training data from uploaded dengue + SST CSVs."""
    data_loader = DataLoader()

    try:
        dengue_df = pd.read_csv(dengue_file)
        sst_df = pd.read_csv(sst_file)
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
    """run the same steps as src/main.py: evaluate 2023, evaluate 2025, forecast 2026."""
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


def main():
    st.markdown('<div class="main-header">Dengue Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtext">Upload a CSV, run prediction, and download a report.</div>', unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("0) Training Plan")
    st.caption("This follows src/main.py: evaluate 2023, evaluate 2025 (exclude 2024), forecast 2026.")
    st.write("Training years: 2010-2022 (test 2023), and 2010-2023 + 2025 (test 2025).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("1) Upload Training Files (Dengue + SST)")
    st.caption("Dengue requires: `data_iniSE`, `casos_est`. SST requires: `YR`, `MON`, `NINO1+2`, `NINO3`, `NINO3.4`, `ANOM.3`.")
    dengue_file = st.file_uploader("Upload Dengue CSV", type=["csv"])
    sst_file = st.file_uploader("Upload SST CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if dengue_file is None or sst_file is None:
        st.info("Upload both training files to proceed.")
        st.stop()

    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("2) Train + Forecast (Full Pipeline)")

    if st.button("Run Full Pipeline + Forecast", type="primary"):
        with st.spinner("Running training pipeline and generating forecast..."):
            try:
                df_base = prepare_training_data_from_uploads(dengue_file, sst_file)
                pipeline = run_full_pipeline(df_base=df_base)

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
