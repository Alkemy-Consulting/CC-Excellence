"""
Enhanced forecasting execution module that integrates all the advanced model implementations.
Provides a unified interface for running Prophet, ARIMA, SARIMA, and Holt-Winters models with advanced features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    from .prophet_enhanced import run_prophet_forecast
    from .arima_enhanced import run_arima_forecast  
    from .sarima_enhanced import run_sarima_forecast
    from .holtwinters_enhanced import run_holtwinters_forecast
    from .config import MODEL_LABELS, ERROR_MESSAGES
except ImportError as e:
    st.error(f"Error importing enhanced modules: {e}")


def run_enhanced_forecast(
    data: pd.DataFrame,
    date_column: str,
    target_column: str,
    model_type: str,
    model_config: Dict[str, Any],
    forecast_config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, go.Figure]]:
    """
    Unified interface for running enhanced forecasting models.
    
    Args:
        data: Input DataFrame with date and target columns
        date_column: Name of the date column
        target_column: Name of the target variable column
        model_type: Type of model to run ('Prophet', 'ARIMA', 'SARIMA', 'Holt-Winters')
        model_config: Model-specific configuration parameters
        forecast_config: Forecast configuration (periods, confidence interval, etc.)
        
    Returns:
        Tuple of (forecast_df, metrics, plots)
    """
    try:
        # Prepare common configuration
        config = {
            'date_column': date_column,
            'target_column': target_column,
            'forecast_periods': forecast_config.get('forecast_periods', 30),
            'confidence_interval': forecast_config.get('confidence_interval', 0.95),
            'train_size': forecast_config.get('train_size', 0.8),
            **model_config
        }
        
        # Route to appropriate model
        if model_type == "Prophet":
            return run_prophet_forecast(data, config)
        elif model_type == "ARIMA":
            return run_arima_forecast(data, config)
        elif model_type == "SARIMA":
            return run_sarima_forecast(data, config)
        elif model_type == "Holt-Winters":
            return run_holtwinters_forecast(data, config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    except Exception as e:
        st.error(f"Error in {model_type} forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}


def run_auto_select_forecast(
    data: pd.DataFrame,
    date_column: str,
    target_column: str,
    model_configs: Dict[str, Dict[str, Any]],
    forecast_config: Dict[str, Any]
) -> Tuple[str, pd.DataFrame, Dict[str, Any], Dict[str, go.Figure]]:
    """
    Run multiple models and select the best performer automatically.
    
    Args:
        data: Input DataFrame
        date_column: Name of the date column
        target_column: Name of the target variable column
        model_configs: Dictionary of model configurations
        forecast_config: Forecast configuration
        
    Returns:
        Tuple of (best_model_name, forecast_df, metrics, plots)
    """
    try:
        models_to_test = ['Prophet', 'ARIMA', 'SARIMA', 'Holt-Winters']
        model_results = {}
        
        st.info("🔍 Testing multiple models to find the best performer...")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, model_type in enumerate(models_to_test):
            try:
                status_text.text(f"Testing {model_type}...")
                progress_bar.progress((i + 1) / len(models_to_test))
                
                # Get model config or use defaults
                model_config = model_configs.get(model_type, {})
                
                # Run model
                forecast_df, metrics, plots = run_enhanced_forecast(
                    data, date_column, target_column, model_type, model_config, forecast_config
                )
                
                if not forecast_df.empty and metrics:
                    model_results[model_type] = {
                        'forecast': forecast_df,
                        'metrics': metrics,
                        'plots': plots,
                        'score': calculate_model_score(metrics)
                    }
                    st.success(f"✅ {model_type} completed successfully")
                else:
                    st.warning(f"⚠️ {model_type} failed to produce results")
                    
            except Exception as e:
                st.warning(f"⚠️ {model_type} failed: {str(e)}")
                continue
        
        progress_bar.progress(1.0)
        status_text.text("Model comparison complete!")
        
        if not model_results:
            st.error("❌ No models succeeded. Please check your data and configurations.")
            return "None", pd.DataFrame(), {}, {}
        
        # Select best model based on scores
        best_model = min(model_results.keys(), key=lambda k: model_results[k]['score'])
        
        # Display comparison results
        st.subheader("📊 Model Comparison Results")
        
        comparison_data = []
        for model, results in model_results.items():
            metrics = results['metrics']
            comparison_data.append({
                'Model': model,
                'Score': f"{results['score']:.4f}",
                'AIC': f"{metrics.get('aic', 'N/A'):.4f}" if isinstance(metrics.get('aic'), (int, float)) else 'N/A',
                'RMSE': f"{metrics.get('val_rmse', metrics.get('train_rmse', 'N/A')):.4f}" if isinstance(metrics.get('val_rmse', metrics.get('train_rmse')), (int, float)) else 'N/A',
                'MAPE': f"{metrics.get('val_mape', metrics.get('train_mape', 'N/A')):.2%}" if isinstance(metrics.get('val_mape', metrics.get('train_mape')), (int, float)) else 'N/A',
                'Status': '🏆 Best' if model == best_model else '✅ Good'
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        st.success(f"🏆 Best model selected: **{best_model}** (Score: {model_results[best_model]['score']:.4f})")
        
        return (
            best_model,
            model_results[best_model]['forecast'],
            model_results[best_model]['metrics'],
            model_results[best_model]['plots']
        )
        
    except Exception as e:
        st.error(f"Error in auto-select forecasting: {str(e)}")
        return "None", pd.DataFrame(), {}, {}


def calculate_model_score(metrics: Dict[str, Any]) -> float:
    """
    Calculate a composite score for model comparison.
    Lower scores are better.
    
    Args:
        metrics: Dictionary of model metrics
        
    Returns:
        Composite score (lower is better)
    """
    try:
        score = 0.0
        weight_sum = 0.0
        
        # AIC (lower is better) - weight: 0.3
        if 'aic' in metrics and isinstance(metrics['aic'], (int, float)) and not np.isnan(metrics['aic']):
            score += 0.3 * metrics['aic']
            weight_sum += 0.3
        
        # Validation RMSE (lower is better) - weight: 0.4
        rmse = metrics.get('val_rmse', metrics.get('train_rmse'))
        if rmse is not None and isinstance(rmse, (int, float)) and not np.isnan(rmse):
            score += 0.4 * rmse
            weight_sum += 0.4
        
        # Validation MAPE (lower is better) - weight: 0.3
        mape = metrics.get('val_mape', metrics.get('train_mape'))
        if mape is not None and isinstance(mape, (int, float)) and not np.isnan(mape):
            score += 0.3 * mape * 100  # Convert to percentage
            weight_sum += 0.3
        
        # Normalize by total weight
        if weight_sum > 0:
            score = score / weight_sum
        else:
            score = float('inf')  # No valid metrics
        
        return score
        
    except Exception:
        return float('inf')


def display_forecast_results(
    model_name: str,
    forecast_df: pd.DataFrame,
    metrics: Dict[str, Any],
    plots: Dict[str, go.Figure],
    show_diagnostics: bool = True
):
    """
    Display comprehensive forecast results with visualizations and metrics.
    
    Args:
        model_name: Name of the model used
        forecast_df: Forecast results DataFrame
        metrics: Model performance metrics
        plots: Dictionary of plot figures
        show_diagnostics: Whether to show diagnostic plots
    """
    try:
        st.subheader(f"📈 {model_name} Forecast Results")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Forecast", "📋 Metrics", "🔍 Diagnostics", "📑 Export"])
        
        with tab1:
            # Main forecast plot
            if 'forecast' in plots:
                st.plotly_chart(plots['forecast'], use_container_width=True)
            
            # Forecast data table
            if not forecast_df.empty:
                st.subheader("📅 Forecast Data")
                
                # Format the forecast data for display
                display_df = forecast_df.copy()
                if 'ds' in display_df.columns:
                    display_df['ds'] = pd.to_datetime(display_df['ds']).dt.strftime('%Y-%m-%d')
                
                for col in ['yhat', 'yhat_lower', 'yhat_upper']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].round(2)
                
                st.dataframe(display_df, use_container_width=True)
        
        with tab2:
            # Performance metrics
            st.subheader("📊 Model Performance Metrics")
            
            if metrics:
                # Create metrics columns
                col1, col2, col3 = st.columns(3)
                
                # Training metrics
                with col1:
                    st.markdown("**Training Metrics**")
                    if 'train_mae' in metrics:
                        st.metric("MAE", f"{metrics['train_mae']:.4f}")
                    if 'train_rmse' in metrics:
                        st.metric("RMSE", f"{metrics['train_rmse']:.4f}")
                    if 'train_mape' in metrics:
                        st.metric("MAPE", f"{metrics['train_mape']:.2%}")
                
                # Validation metrics
                with col2:
                    st.markdown("**Validation Metrics**")
                    if 'val_mae' in metrics:
                        st.metric("MAE", f"{metrics['val_mae']:.4f}")
                    if 'val_rmse' in metrics:
                        st.metric("RMSE", f"{metrics['val_rmse']:.4f}")
                    if 'val_mape' in metrics:
                        st.metric("MAPE", f"{metrics['val_mape']:.2%}")
                
                # Information criteria
                with col3:
                    st.markdown("**Information Criteria**")
                    if 'aic' in metrics:
                        st.metric("AIC", f"{metrics['aic']:.4f}")
                    if 'bic' in metrics:
                        st.metric("BIC", f"{metrics['bic']:.4f}")
                    if 'aicc' in metrics:
                        st.metric("AICc", f"{metrics['aicc']:.4f}")
                
                # Full metrics table
                st.subheader("📋 All Metrics")
                metrics_df = pd.DataFrame([
                    {"Metric": k, "Value": f"{v:.6f}" if isinstance(v, (int, float)) and not np.isnan(v) else str(v)}
                    for k, v in metrics.items()
                ])
                st.dataframe(metrics_df, use_container_width=True)
            else:
                st.warning("No metrics available")
        
        with tab3:
            # Diagnostic plots
            if show_diagnostics and plots:
                st.subheader("🔍 Model Diagnostics")
                
                # Residuals analysis
                if 'residuals' in plots:
                    st.subheader("📉 Residuals Analysis")
                    st.plotly_chart(plots['residuals'], use_container_width=True)
                
                # Components/decomposition
                if 'components' in plots:
                    st.subheader("🔄 Time Series Components")
                    st.plotly_chart(plots['components'], use_container_width=True)
                
                # ACF/PACF for ARIMA/SARIMA
                if 'acf_pacf' in plots:
                    st.subheader("📊 Autocorrelation Analysis")
                    st.plotly_chart(plots['acf_pacf'], use_container_width=True)
                
                # Model diagnostics summary
                if 'diagnostics' in plots:
                    st.subheader("📋 Diagnostics Summary")
                    st.plotly_chart(plots['diagnostics'], use_container_width=True)
            else:
                st.info("Diagnostic plots not available for this model")
        
        with tab4:
            # Export options
            st.subheader("💾 Export Forecast Results")
            
            if not forecast_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # CSV export
                    csv_data = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv_data,
                        file_name=f"{model_name.lower()}_forecast.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Excel export with multiple sheets
                    excel_buffer = create_excel_export(forecast_df, metrics, model_name)
                    if excel_buffer:
                        st.download_button(
                            label="📊 Download Excel",
                            data=excel_buffer,
                            file_name=f"{model_name.lower()}_forecast_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                with col3:
                    # JSON export for API integration
                    import json
                    json_data = {
                        'model': model_name,
                        'forecast': forecast_df.to_dict('records'),
                        'metrics': {k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else str(v) 
                                  for k, v in metrics.items()},
                        'generated_at': pd.Timestamp.now().isoformat()
                    }
                    
                    st.download_button(
                        label="📋 Download JSON",
                        data=json.dumps(json_data, indent=2),
                        file_name=f"{model_name.lower()}_forecast.json",
                        mime="application/json"
                    )
            else:
                st.warning("No forecast data available for export")
    
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")


def create_excel_export(forecast_df: pd.DataFrame, metrics: Dict[str, Any], model_name: str) -> bytes:
    """Create comprehensive Excel export with multiple sheets."""
    try:
        import io
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
        from openpyxl.styles import Font, PatternFill
        
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Forecast data
            forecast_df.to_excel(writer, sheet_name='Forecast', index=False)
            
            # Metrics
            if metrics:
                metrics_df = pd.DataFrame([
                    {"Metric": k, "Value": v} for k, v in metrics.items()
                ])
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Summary sheet
            summary_data = {
                'Model': [model_name],
                'Forecast_Periods': [len(forecast_df)],
                'Generated_At': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        return buffer.getvalue()
        
    except Exception as e:
        st.warning(f"Error creating Excel export: {str(e)}")
        return b""
