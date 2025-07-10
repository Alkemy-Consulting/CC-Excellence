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
    from .prophet_enhanced import run_prophet_model
    from .arima_enhanced import run_arima_model  
    from .sarima_enhanced import run_sarima_forecast
    from .holtwinters_enhanced import run_holtwinters_forecast
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    try:
        # Fallback for direct execution
        from prophet_enhanced import run_prophet_model
        from arima_enhanced import run_arima_model  
        from sarima_enhanced import run_sarima_forecast
        from holtwinters_enhanced import run_holtwinters_forecast
        ENHANCED_MODELS_AVAILABLE = True
    except ImportError as e:
        st.warning(f"⚠️ Some enhanced modules unavailable: {e}")
        ENHANCED_MODELS_AVAILABLE = False
        
        # Create placeholder functions to prevent crashes
        def run_prophet_model(*args, **kwargs):
            st.error("Prophet enhanced module not available")
            return None
            
        def run_arima_model(*args, **kwargs):
            st.error("ARIMA enhanced module not available")
            return None
            
        def run_sarima_forecast(*args, **kwargs):
            st.error("SARIMA enhanced module not available")
            return pd.DataFrame(), {}, {}
            
        def run_holtwinters_forecast(*args, **kwargs):
            st.error("Holt-Winters enhanced module not available")
            return pd.DataFrame(), {}, {}


# Wrapper functions to match expected interface
def run_prophet_forecast(data: pd.DataFrame, date_col: str, target_col: str, 
                        model_config: Dict[str, Any], base_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Wrapper function for Prophet to match forecast engine interface.
    """
    try:
        # Prepare data for Prophet
        prophet_df = data.copy()
        prophet_df['ds'] = pd.to_datetime(prophet_df[date_col])
        prophet_df['y'] = prophet_df[target_col]
        prophet_df = prophet_df[['ds', 'y']].copy()
        
        # Simple Prophet model creation
        from prophet import Prophet
        
        # Extract Prophet parameters
        model_params = {
            'seasonality_mode': model_config.get('seasonality_mode', 'additive'),
            'yearly_seasonality': model_config.get('yearly_seasonality', True),
            'weekly_seasonality': model_config.get('weekly_seasonality', True),
            'daily_seasonality': model_config.get('daily_seasonality', False),
            'changepoint_prior_scale': model_config.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': model_config.get('seasonality_prior_scale', 10.0),
        }
        
        # Create and fit Prophet model
        model = Prophet(**model_params)
        model.fit(prophet_df)
        
        # Create future dataframe
        forecast_periods = base_config.get('forecast_periods', 30)
        future = model.make_future_dataframe(periods=forecast_periods, freq='D')
        
        # Make forecast
        forecast = model.predict(future)
        
        # Get only future predictions
        forecast_df = forecast.tail(forecast_periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        # Calculate basic metrics
        metrics = {}
        try:
            train_forecast = forecast.head(len(prophet_df))
            actual = prophet_df['y'].values
            predicted = train_forecast['yhat'].values
            
            min_len = min(len(actual), len(predicted))
            actual = actual[-min_len:]
            predicted = predicted[-min_len:]
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            metrics['train_mae'] = mean_absolute_error(actual, predicted)
            metrics['train_rmse'] = np.sqrt(mean_squared_error(actual, predicted))
            
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                metrics['train_mape'] = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]))
            
        except Exception:
            pass
        
        # Create basic visualization
        plots = {}
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=prophet_df['ds'],
                y=prophet_df['y'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='Confidence Interval'
            ))
            
            fig.update_layout(
                title='Prophet Forecast',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            plots['forecast'] = fig
            
        except Exception:
            pass
        
        return forecast_df, metrics, plots
        
    except Exception as e:
        st.error(f"Prophet forecast error: {str(e)}")
        return pd.DataFrame(), {}, {}


def run_arima_forecast(data: pd.DataFrame, date_col: str, target_col: str,
                      model_config: Dict[str, Any], base_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Wrapper function for ARIMA to match forecast engine interface.
    """
    try:
        # Prepare data
        ts_data = data.copy()
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        ts_data = ts_data.sort_values(date_col)
        
        series = pd.Series(
            ts_data[target_col].values,
            index=pd.to_datetime(ts_data[date_col])
        )
        
        if series.isnull().any():
            series = series.interpolate(method='linear')
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            p = model_config.get('p', 1)
            d = model_config.get('d', 1) 
            q = model_config.get('q', 1)
            
            auto_order = model_config.get('auto_order', True)
            if auto_order:
                try:
                    from pmdarima import auto_arima
                    auto_model = auto_arima(
                        series, 
                        start_p=0, start_q=0,
                        max_p=3, max_q=3,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore'
                    )
                    p, d, q = auto_model.order
                except Exception:
                    pass
            
            model = ARIMA(series, order=(p, d, q))
            fitted_model = model.fit()
            
            forecast_periods = base_config.get('forecast_periods', 30)
            forecast_result = fitted_model.forecast(steps=forecast_periods, alpha=0.05)
            forecast_values = forecast_result
            
            try:
                forecast_ci = fitted_model.get_forecast(steps=forecast_periods).conf_int()
                conf_lower = forecast_ci.iloc[:, 0].values
                conf_upper = forecast_ci.iloc[:, 1].values
            except Exception:
                residuals_std = np.std(fitted_model.resid)
                conf_lower = forecast_values - 1.96 * residuals_std
                conf_upper = forecast_values + 1.96 * residuals_std
            
            last_date = series.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
            
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast_values,
                'yhat_lower': conf_lower,
                'yhat_upper': conf_upper
            })
            
            metrics = {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'order': (p, d, q)
            }
            
            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                fitted_values = fitted_model.fittedvalues
                actual_values = series.values
                
                min_len = min(len(actual_values), len(fitted_values))
                actual_aligned = actual_values[-min_len:]
                fitted_aligned = fitted_values[-min_len:]
                
                metrics.update({
                    'train_mae': mean_absolute_error(actual_aligned, fitted_aligned),
                    'train_rmse': np.sqrt(mean_squared_error(actual_aligned, fitted_aligned))
                })
                
                non_zero_mask = actual_aligned != 0
                if np.any(non_zero_mask):
                    metrics['train_mape'] = np.mean(np.abs((actual_aligned[non_zero_mask] - fitted_aligned[non_zero_mask]) / actual_aligned[non_zero_mask]))
                    
            except Exception:
                pass
            
            plots = {}
            try:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=series.index,
                    y=series.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['ds'],
                    y=forecast_df['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f'ARIMA({p},{d},{q}) Forecast',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    hovermode='x unified'
                )
                
                plots['forecast'] = fig
                
            except Exception:
                pass
            
            return forecast_df, metrics, plots
            
        except Exception as e:
            st.error(f"ARIMA model error: {str(e)}")
            return pd.DataFrame(), {}, {}
        
    except Exception as e:
        st.error(f"ARIMA forecast error: {str(e)}")
        return pd.DataFrame(), {}, {}


def run_enhanced_forecast(data: pd.DataFrame, date_col: str, target_col: str, 
                         model_name: str, model_config: Dict[str, Any], 
                         base_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Enhanced forecast execution with proper state management.
    """
    
    # Validate inputs
    if data is None or data.empty:
        raise ValueError("No data provided for forecasting")
    
    if date_col not in data.columns or target_col not in data.columns:
        raise ValueError(f"Required columns not found: {date_col}, {target_col}")
    
    # Ensure data is properly formatted
    data = data.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    
    # Remove any duplicate dates
    data = data.drop_duplicates(subset=[date_col], keep='last')
    
    # Handle missing values
    data[target_col] = data[target_col].fillna(method='ffill').fillna(method='bfill')
    
    try:
        # Route to appropriate model with error handling
        if model_name.lower() == 'prophet':
            return run_prophet_forecast(data, date_col, target_col, model_config, base_config)
        elif model_name.lower() == 'arima':
            return run_arima_forecast(data, date_col, target_col, model_config, base_config)
        elif model_name == 'SARIMA':
            # Pass all required columns to SARIMA
            return run_sarima_forecast(data, date_col, target_col, model_config, base_config)
        elif model_name == 'Holt-Winters':
            if ENHANCED_MODELS_AVAILABLE:
                forecast_df, metrics, plots = run_holtwinters_forecast(
                    data, date_col, target_col, model_config
                )
            else:
                st.error("Holt-Winters enhanced module is not available.")
                return None
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    except Exception as e:
        # Return empty results with error information
        empty_df = pd.DataFrame()
        error_metrics = {
            'error': str(e),
            'model': model_name,
            'status': 'failed'
        }
        empty_plots = {}
        
        return empty_df, error_metrics, empty_plots


def run_auto_select_forecast(
    data: pd.DataFrame,
    date_column: str,
    target_column: str,
    model_configs: Dict[str, Dict[str, Any]],
    forecast_config: Dict[str, Any]
) -> Tuple[str, pd.DataFrame, Dict[str, Any], Dict[str, go.Figure]]:
    """
    Run multiple models and select the best performer automatically.
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
                'RMSE': f"{metrics.get('train_rmse', 'N/A'):.4f}" if isinstance(metrics.get('train_rmse'), (int, float)) else 'N/A',
                'MAPE': f"{metrics.get('train_mape', 'N/A'):.2%}" if isinstance(metrics.get('train_mape'), (int, float)) else 'N/A',
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
    """
    try:
        score = 0.0
        weight_sum = 0.0
        
        # AIC (lower is better) - weight: 0.3
        if 'aic' in metrics and isinstance(metrics['aic'], (int, float)) and not np.isnan(metrics['aic']):
            score += 0.3 * metrics['aic']
            weight_sum += 0.3
        
        # Training RMSE (lower is better) - weight: 0.4
        rmse = metrics.get('train_rmse')
        if rmse is not None and isinstance(rmse, (int, float)) and not np.isnan(rmse):
            score += 0.4 * rmse
            weight_sum += 0.4
        
        # Training MAPE (lower is better) - weight: 0.3
        mape = metrics.get('train_mape')
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
    """
    try:
        st.subheader(f"📈 {model_name} Forecast Results")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Forecast", "📋 Metrics", "📑 Export"])
        
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
                col1, col2 = st.columns(2)
                
                # Training metrics
                with col1:
                    st.markdown("**Training Metrics**")
                    if 'train_mae' in metrics:
                        st.metric("MAE", f"{metrics['train_mae']:.4f}")
                    if 'train_rmse' in metrics:
                        st.metric("RMSE", f"{metrics['train_rmse']:.4f}")
                    if 'train_mape' in metrics:
                        st.metric("MAPE", f"{metrics['train_mape']:.2%}")
                
                # Information criteria
                with col2:
                    st.markdown("**Information Criteria**")
                    if 'aic' in metrics:
                        st.metric("AIC", f"{metrics['aic']:.4f}")
                    if 'bic' in metrics:
                        st.metric("BIC", f"{metrics['bic']:.4f}")
                    if 'order' in metrics:
                        st.metric("Model Order", str(metrics['order']))
                
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
            # Export options
            st.subheader("💾 Export Forecast Results")
            
            if not forecast_df.empty:
                col1, col2 = st.columns(2)
                
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
                    # JSON export for API integration
                    import json
                    
                    # Custom JSON serializer for handling timestamps and numpy types
                    def json_serializer(obj):
                        if isinstance(obj, pd.Timestamp):
                            return obj.isoformat()
                        elif isinstance(obj, np.datetime64):
                            return pd.Timestamp(obj).isoformat()
                        elif isinstance(obj, (np.integer, np.floating)):
                            return float(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif pd.isna(obj):
                            return None
                        return str(obj)
                    
                    # Prepare forecast data for JSON serialization
                    forecast_records = []
                    for _, row in forecast_df.iterrows():
                        record = {}
                        for col, value in row.items():
                            try:
                                record[col] = json_serializer(value)
                            except:
                                record[col] = str(value)
                        forecast_records.append(record)
                    
                    json_data = {
                        'model': model_name,
                        'forecast': forecast_records,
                        'metrics': {k: float(v) if isinstance(v, (int, float)) and not np.isnan(v) else str(v) 
                                  for k, v in metrics.items()},
                        'generated_at': pd.Timestamp.now().isoformat()
                    }
                    
                    st.download_button(
                        label="📋 Download JSON",
                        data=json.dumps(json_data, indent=2, default=json_serializer),
                        file_name=f"{model_name.lower()}_forecast.json",
                        mime="application/json"
                    )
            else:
                st.warning("No forecast data available for export")
    
    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
