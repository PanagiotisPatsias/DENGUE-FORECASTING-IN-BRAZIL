"""
streamlit web application for dengue forecasting.
provides interactive interface for visualization, forecasting, and custom data input.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.data_loader import DataLoader
from src.core.feature_engineer import FeatureEngineer
from src.core.forecaster import Forecaster
from src.utils.model_manager import ModelManager
from src.utils.config import Config
import mlflow
from datetime import datetime, timedelta

# page configuration
st.set_page_config(
    page_title="Dengue Forecasting System",
    page_icon="[DENGUE]",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    """load model, metadata, and historical data."""
    manager = ModelManager()
    result = manager.load_baseline_model()
    
    if result is None:
        return None, None, None, None
    
    model, metadata = result
    
    # load data
    data_loader = DataLoader()
    df = data_loader.load_and_prepare_data()
    
    # create features
    feature_engineer = FeatureEngineer()
    df, _ = feature_engineer.create_features(df)
    
    return model, metadata, df, feature_engineer


def check_for_drift_alerts():
    """
    check mlflow for recent drift detection runs.
    returns tuple of (has_drift, drift_info_dict)
    """
    try:
        # set mlflow tracking uri
        mlflow.set_tracking_uri("./mlruns")
        client = mlflow.tracking.MlflowClient()
        
        # get dengue_forecasting experiment
        experiments = client.search_experiments()
        experiment_id = None
        for exp in experiments:
            if exp.name == "dengue_forecasting":
                experiment_id = exp.experiment_id
                break
        
        if not experiment_id:
            return False, None
        
        # search for recent drift check runs
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="tags.run_type = 'drift_check'",
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            return False, None
        
        latest_run = runs[0]
        
        # check if drift was detected
        drift_detected = latest_run.data.tags.get('drift_detected', 'False') == 'True'
        
        if drift_detected:
            drift_info = {
                'run_id': latest_run.info.run_id,
                'test_year': latest_run.data.tags.get('test_year', 'Unknown'),
                'severity': latest_run.data.tags.get('severity', 'UNKNOWN'),
                'drift_status': latest_run.data.tags.get('DRIFT_STATUS', 'Drift detected'),
                'r2_degradation': latest_run.data.metrics.get('r2_degradation', 0),
                'current_r2': latest_run.data.metrics.get('current_r2', 0),
                'baseline_r2': latest_run.data.metrics.get('baseline_r2', 0),
                'run_name': latest_run.info.run_name,
                'timestamp': datetime.fromtimestamp(latest_run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S')
            }
            return True, drift_info
        
        return False, None
        
    except Exception as e:
        # silently fail if mlflow not available
        return False, None


def show_drift_alert(drift_info):
    """display drift alert banner at top of page."""
    severity = drift_info['severity']
    
    # color coding based on severity
    if severity == 'CRITICAL':
        bg_color = '#f8d7da'
        border_color = '#dc3545'
        icon = '[ALERT]'
    elif severity == 'WARNING':
        bg_color = '#fff3cd'
        border_color = '#ffc107'
        icon = '[WARNING]'
    else:
        bg_color = '#f8d7da'
        border_color = '#dc3545'
        icon = '[FAIL]'
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color};
        border: 2px solid {border_color};
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 2rem;">{icon}</span>
            <div style="flex: 1;">
                <h3 style="margin: 0; color: #721c24; font-size: 1.3rem;">
                    {severity} MODEL DRIFT DETECTED
                </h3>
                <p style="margin: 8px 0 0 0; color: #721c24;">
                    <strong>Test Year:</strong> {drift_info['test_year']} | 
                    <strong>R² Degradation:</strong> {drift_info['r2_degradation']:.4f} | 
                    <strong>Current R²:</strong> {drift_info['current_r2']:.4f} 
                    (was {drift_info['baseline_r2']:.4f})
                </p>
                <p style="margin: 8px 0 0 0; color: #721c24; font-size: 0.9rem;">
                    <em>Detected: {drift_info['timestamp']}</em>
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("[METRICS] View in MLflow", type="primary"):
            st.markdown(f"""
            <script>
                window.open('http://localhost:5000', '_blank');
            </script>
            """, unsafe_allow_html=True)
            st.info("Opening MLflow UI in new tab...")
    
    with col2:
        if st.button("[SEARCH] Check Drift Details"):
            st.session_state['navigate_to'] = 'drift_monitoring'
            st.rerun()
    
    with col3:
        st.markdown(f"""
        <div style="padding: 8px; background-color: #fff; border-radius: 4px; border: 1px solid {border_color};">
            <small><strong>Recommendation:</strong> Review drift details and consider retraining the model with recent data.</small>
        </div>
        """, unsafe_allow_html=True)


def create_time_series_plot(df, actual_col='casos_est', pred_col=None, title="Dengue Cases Over Time"):
    """create interactive time series plot."""
    fig = go.Figure()
    
    # actual values
    fig.add_trace(go.Scatter(
        x=df['year_quarter'],
        y=df[actual_col],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#FF4B4B', width=2),
        marker=dict(size=6)
    ))
    
    # predictions if available
    if pred_col and pred_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df['year_quarter'],
            y=df[pred_col],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#4B4BFF', width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Quarter",
        yaxis_title="Number of Cases",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def create_feature_importance_plot(model, features):
    """create feature importance bar chart."""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 15 Feature Importance',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_forecast_plot(historical_df, forecast_df):
    """create plot showing historical data and forecast."""
    fig = go.Figure()
    
    # historical data
    fig.add_trace(go.Scatter(
        x=historical_df['year_quarter'],
        y=historical_df['casos_est'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#FF4B4B', width=2),
        marker=dict(size=6)
    ))
    
    # forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['year_quarter'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#4B4BFF', width=2, dash='dash'),
        marker=dict(size=8, symbol='star')
    ))
    
    fig.update_layout(
        title='Dengue Cases: Historical & Forecast',
        xaxis_title='Quarter',
        yaxis_title='Number of Cases',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def main():
    """main streamlit app."""
    
    # header
    st.markdown('<div class="main-header">[DENGUE] Dengue Forecasting System</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # sidebar
    st.sidebar.title("Navigation")
    
    # check for drift and show indicator in sidebar
    has_drift, drift_info = check_for_drift_alerts()
    if has_drift:
        severity = drift_info['severity']
        if severity == 'CRITICAL':
            st.sidebar.error("[ALERT] CRITICAL DRIFT DETECTED!")
        elif severity == 'WARNING':
            st.sidebar.warning("[WARNING] Model Drift Warning")
        else:
            st.sidebar.warning("[FAIL] Drift Detected")
        
        st.sidebar.caption(f"Last detected: {drift_info['timestamp']}")
        st.sidebar.caption(f"R² dropped by {drift_info['r2_degradation']:.4f}")
    
    # check if navigation override from drift alert button
    if 'navigate_to' in st.session_state:
        page_map = {
            'drift_monitoring': "[SEARCH] Drift Monitoring"
        }
        default_page = page_map.get(st.session_state['navigate_to'], "[METRICS] Dashboard")
        del st.session_state['navigate_to']
    else:
        default_page = "[METRICS] Dashboard"
    
    page = st.sidebar.radio(
        "Select Page",
        ["[METRICS] Dashboard", "[FORECAST] Make Forecast", "[SEARCH] Drift Monitoring", "[FILE] Custom Data", "[INFO] Model Info"],
        index=["[METRICS] Dashboard", "[FORECAST] Make Forecast", "[SEARCH] Drift Monitoring", "[FILE] Custom Data", "[INFO] Model Info"].index(default_page) if default_page in ["[METRICS] Dashboard", "[FORECAST] Make Forecast", "[SEARCH] Drift Monitoring", "[FILE] Custom Data", "[INFO] Model Info"] else 0
    )
    
    # load model and data
    with st.spinner("Loading model and data..."):
        model, metadata, df, feature_engineer = load_model_and_data()
    
    if model is None:
        st.error("[ERROR] No baseline model found! Please run training first: `python -m src.main`")
        return
    
    # page routing
    if page == "[METRICS] Dashboard":
        show_dashboard(model, metadata, df)
    elif page == "[FORECAST] Make Forecast":
        show_forecast_page(model, metadata, df, feature_engineer)
    elif page == "[SEARCH] Drift Monitoring":
        show_drift_monitoring_page(model, metadata, df, feature_engineer)
    elif page == "[FILE] Custom Data":
        show_custom_data_page(model, metadata, feature_engineer)
    elif page == "[INFO] Model Info":
        show_model_info(model, metadata)


def show_dashboard(model, metadata, df):
    """show main dashboard with visualizations."""
    st.header("[METRICS] Dashboard")
    
    # check for drift alerts
    has_drift, drift_info = check_for_drift_alerts()
    if has_drift:
        show_drift_alert(drift_info)
    
    # model info cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Model Type", metadata['model_name'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("R² Score", f"{metadata['metrics']['r2']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("MAE", f"{metadata['metrics']['mae']:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features", metadata['num_features'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # time series visualization
    st.subheader("Historical Dengue Cases")
    
    # year filter
    years = sorted(df['year'].unique())
    selected_years = st.multiselect(
        "Select Years to Display",
        years,
        default=years[-5:] if len(years) >= 5 else years
    )
    
    filtered_df = df[df['year'].isin(selected_years)]
    
    if not filtered_df.empty:
        fig = create_time_series_plot(filtered_df, title="Dengue Cases by Quarter")
        st.plotly_chart(fig, use_container_width=True)
    
    # statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistics")
        stats_df = df['casos_est'].describe().round(2)
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.subheader("Feature Importance")
        fig = create_feature_importance_plot(model, metadata['features'])
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def show_forecast_page(model, metadata, df, feature_engineer):
    """show forecasting page."""
    st.header("[FORECAST] Make Forecast")
    
    # check for drift alerts
    has_drift, drift_info = check_for_drift_alerts()
    if has_drift:
        show_drift_alert(drift_info)
    
    st.info(" Generate multi-quarter ahead forecasts using the trained model.")
    
    # forecast parameters
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.selectbox(
            "Start Year",
            sorted(df['year'].unique()),
            index=len(df['year'].unique()) - 1
        )
    
    with col2:
        n_quarters = st.slider("Number of Quarters Ahead", 1, 8, 4)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                # prepare data
                forecaster = Forecaster(feature_engineer)
                
                # get last available data
                last_df = df[df['year'] <= start_year].copy()
                
                # refit and forecast
                forecast_results = forecaster.refit_and_forecast(
                    last_df,
                    metadata['features'],
                    model,
                    n_steps=n_quarters
                )
                
                # display results
                st.success("[OK] Forecast generated successfully!")
                
                # forecast table
                st.subheader("Forecast Results")
                forecast_df = pd.DataFrame({
                    'Quarter': [f"{start_year + (q-1)//4}-Q{((q-1)%4)+1}" for q in range(1, n_quarters+1)],
                    'Predicted Cases': [f"{int(val):,}" for val in forecast_results]
                })
                st.dataframe(forecast_df, use_container_width=True)
                
                # visualization
                st.subheader("Forecast Visualization")
                
                # prepare plot data
                historical_df = last_df.tail(12).copy()  # last 3 years
                
                forecast_plot_df = pd.DataFrame({
                    'year_quarter': [f"{start_year + (q-1)//4}-Q{((q-1)%4)+1}" for q in range(1, n_quarters+1)],
                    'forecast': forecast_results
                })
                
                fig = create_forecast_plot(historical_df, forecast_plot_df)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"[ERROR] Error generating forecast: {str(e)}")


def show_drift_monitoring_page(model, metadata, df, feature_engineer):
    """show drift monitoring page."""
    st.header("[SEARCH] Drift Monitoring")
    
    st.info("[METRICS] Monitor model performance degradation and detect when retraining is needed.")
    
    # baseline info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Baseline Model", metadata['model_name'])
    with col2:
        st.metric("Baseline R²", f"{metadata['metrics']['r2']:.4f}")
    with col3:
        st.metric("Baseline Year", metadata['test_year'])
    
    st.markdown("---")
    
    # drift detection section
    st.subheader("Test for Drift")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_year = st.selectbox(
            "Select Year to Test",
            sorted(df['year'].unique(), reverse=True)
        )
    
    with col2:
        drift_threshold = st.slider(
            "Drift Threshold (R² degradation)",
            0.05, 0.30, 0.15, 0.05,
            help="Alert if R² drops by more than this amount"
        )
    
    if st.button("Check for Drift", type="primary"):
        with st.spinner("Analyzing model performance..."):
            try:
                # filter test data
                test_df = df[df['year'] == test_year].copy()
                
                if test_df.empty:
                    st.error(f"No data available for year {test_year}")
                    return
                
                # ensure features match
                available_features = [f for f in metadata['features'] if f in test_df.columns]
                
                if len(available_features) < len(metadata['features']) * 0.5:
                    st.warning("[WARNING] Many features are missing. Results may be unreliable.")
                
                # prepare test data
                X_test = test_df[available_features].fillna(0)
                y_test = test_df['casos_est'].values
                
                # make predictions
                predictions = model.predict(X_test)
                
                # calculate metrics
                from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
                import numpy as np
                
                current_r2 = r2_score(y_test, predictions)
                current_mae = mean_absolute_error(y_test, predictions)
                current_rmse = np.sqrt(mean_squared_error(y_test, predictions))
                
                baseline_r2 = metadata['metrics']['r2']
                baseline_mae = metadata['metrics']['mae']
                baseline_rmse = metadata['metrics']['rmse']
                
                # calculate degradation
                r2_degradation = baseline_r2 - current_r2
                
                # display results
                st.markdown("---")
                st.subheader("Performance Comparison")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "R² Score", 
                        f"{current_r2:.4f}",
                        f"{r2_degradation:.4f}",
                        delta_color="inverse"
                    )
                    st.caption(f"Baseline: {baseline_r2:.4f}")
                
                with col2:
                    mae_change = current_mae - baseline_mae
                    st.metric(
                        "MAE",
                        f"{current_mae:,.0f}",
                        f"{mae_change:,.0f}",
                        delta_color="inverse"
                    )
                    st.caption(f"Baseline: {baseline_mae:,.0f}")
                
                with col3:
                    rmse_change = current_rmse - baseline_rmse
                    st.metric(
                        "RMSE",
                        f"{current_rmse:,.0f}",
                        f"{rmse_change:,.0f}",
                        delta_color="inverse"
                    )
                    st.caption(f"Baseline: {baseline_rmse:,.0f}")
                
                # drift detection
                st.markdown("---")
                st.subheader("Drift Analysis")
                
                drift_detected = r2_degradation > drift_threshold
                
                if drift_detected:
                    st.markdown("""
                    <div style="background-color: #f8d7da; border: 2px solid #dc3545; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                        <h3 style="color: #721c24; margin: 0;">[ALERT] DRIFT DETECTED</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #721c24;">Model performance has degraded significantly. Retraining recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write("**Reasons:**")
                    st.write(f"• R² decreased by {r2_degradation:.4f} (threshold: {drift_threshold:.4f})")
                    st.write(f"• Current R²: {current_r2:.4f} vs Baseline: {baseline_r2:.4f}")
                    
                    st.write("**Recommended Actions:**")
                    st.write("1. Retrain model with latest data: `python -m src.main`")
                    st.write("2. Review feature importance for changes")
                    st.write("3. Investigate data quality issues")
                    st.write("4. Check for distribution shifts")
                else:
                    st.markdown("""
                    <div style="background-color: #d4edda; border: 2px solid #28a745; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">
                        <h3 style="color: #155724; margin: 0;">[OK] NO DRIFT DETECTED</h3>
                        <p style="margin: 0.5rem 0 0 0; color: #155724;">Model is performing within acceptable range.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.write(f"**R² Degradation:** {r2_degradation:.4f} (threshold: {drift_threshold:.4f})")
                    st.write("**Status:** Continue monitoring regularly")
                
                # visualization
                st.markdown("---")
                st.subheader("Predictions vs Actual")
                
                comparison_df = pd.DataFrame({
                    'Quarter': test_df['year_quarter'].values,
                    'Actual': y_test,
                    'Predicted': predictions.astype(int),
                    'Error': y_test - predictions
                })
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=comparison_df['Quarter'],
                    y=comparison_df['Actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='#FF4B4B', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=comparison_df['Quarter'],
                    y=comparison_df['Predicted'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='#4B4BFF', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'Predictions vs Actual - {test_year}',
                    xaxis_title='Quarter',
                    yaxis_title='Cases',
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # detailed table
                with st.expander("View Detailed Comparison"):
                    st.dataframe(comparison_df.style.format({
                        'Actual': '{:,.0f}',
                        'Predicted': '{:,.0f}',
                        'Error': '{:+,.0f}'
                    }), use_container_width=True)
                
            except Exception as e:
                st.error(f"[ERROR] Error: {str(e)}")
    
    # MLflow link
    st.markdown("---")
    st.info("[METRICS] For detailed experiment tracking, view [MLflow UI](http://localhost:5000)")


def show_custom_data_page(model, metadata, feature_engineer):
    """show custom data input page."""
    st.header("[FILE] Custom Data Input")
    
    st.info(" Upload your own data or input values manually to generate forecasts.")
    
    tab1, tab2 = st.tabs(["Upload CSV", "Manual Input"])
    
    with tab1:
        st.subheader("Upload CSV File")
        
        st.markdown("""
        **Required columns:** `year`, `quarter`, `casos_est`  
        **Optional:** Any SST indices or additional features
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                custom_df = pd.read_csv(uploaded_file)
                st.success("[OK] File uploaded successfully!")
                
                st.subheader("Preview")
                st.dataframe(custom_df.head(10), use_container_width=True)
                
                if st.button("Generate Features & Predict", type="primary"):
                    with st.spinner("Processing..."):
                        # create features
                        custom_df, _ = feature_engineer.create_features(custom_df)
                        
                        # ensure features match
                        available_features = [f for f in metadata['features'] if f in custom_df.columns]
                        
                        if len(available_features) < len(metadata['features']) * 0.8:
                            st.warning("[WARNING] Many features are missing. Predictions may be inaccurate.")
                        
                        # predict
                        X = custom_df[available_features].fillna(0)
                        predictions = model.predict(X)
                        
                        # display results
                        result_df = custom_df[['year', 'quarter', 'casos_est']].copy()
                        result_df['predicted'] = predictions.astype(int)
                        result_df['error'] = result_df['casos_est'] - result_df['predicted']
                        
                        st.subheader("Prediction Results")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # download
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            " Download Results",
                            csv,
                            "predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                        
            except Exception as e:
                st.error(f"[ERROR] Error processing file: {str(e)}")
    
    with tab2:
        st.subheader("Manual Input")
        st.info("Enter quarterly data points for prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.number_input("Year", min_value=2010, max_value=2030, value=2026)
        with col2:
            quarter = st.selectbox("Quarter", [1, 2, 3, 4])
        with col3:
            casos_est = st.number_input("Estimated Cases", min_value=0, value=10000)
        
        st.markdown("**Optional: SST Indices**")
        col1, col2 = st.columns(2)
        with col1:
            nino34 = st.number_input("Niño 3.4 Index", value=0.0, format="%.2f")
        with col2:
            nino12 = st.number_input("Niño 1+2 Index", value=0.0, format="%.2f")
        
        if st.button("Predict", type="primary"):
            st.info(" Manual prediction feature coming soon. Use CSV upload for now.")


def show_model_info(model, metadata):
    """show model information page."""
    st.header("[INFO] Model Information")
    
    # model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.write(f"**Type:** {metadata['model_name']}")
        st.write(f"**Saved:** {metadata['saved_at']}")
        st.write(f"**Test Year:** {metadata['test_year']}")
        st.write(f"**Training Years:** {metadata['train_years'][0]} - {metadata['train_years'][-1]}")
    
    with col2:
        st.subheader("Performance Metrics")
        st.write(f"**R² Score:** {metadata['metrics']['r2']:.4f}")
        st.write(f"**MAE:** {metadata['metrics']['mae']:,.2f}")
        st.write(f"**RMSE:** {metadata['metrics']['rmse']:,.2f}")
    
    st.markdown("---")
    
    # features
    st.subheader("Features Used")
    st.write(f"Total: {metadata['num_features']} features")
    
    with st.expander("View All Features"):
        features_df = pd.DataFrame({
            'Feature Name': metadata['features']
        })
        st.dataframe(features_df, use_container_width=True)
    
    # mlflow link
    st.markdown("---")
    st.subheader("MLflow Monitoring")
    st.info("[METRICS] View detailed experiment tracking and model monitoring in MLflow UI")
    st.code("mlflow ui", language="bash")
    st.write("Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
