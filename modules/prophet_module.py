from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.error("Prophet not installed. Please install: pip install prophet")

from .config import PROPHET_DEFAULTS

def run_prophet_forecast(df: pd.DataFrame, date_col: str, target_col: str, 
                        model_config: Dict[str, Any], base_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Enhanced Prophet forecasting with auto-tuning capabilities
    """
    if not PROPHET_AVAILABLE:
        return pd.DataFrame(), {}, {}
    
    try:
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Check if auto-tuning is enabled
        if model_config.get('auto_tune', False):
            st.info("🔍 Starting Prophet auto-tuning...")
            best_params, tuning_results = auto_tune_prophet(
                prophet_df, 
                model_config.get('tuning_horizon', 30),
                model_config.get('tuning_parallel', 'processes')
            )
            
            # Update model config with best parameters
            model_config.update(best_params)
            st.success(f"✅ Auto-tuning completed! Best MAPE: {tuning_results['best_mape']:.3f}")
        
        # Initialize Prophet model
        prophet_params = {
            'changepoint_prior_scale': model_config.get('changepoint_prior_scale', PROPHET_DEFAULTS['changepoint_prior_scale']),
            'seasonality_prior_scale': model_config.get('seasonality_prior_scale', PROPHET_DEFAULTS['seasonality_prior_scale']),
            'seasonality_mode': model_config.get('seasonality_mode', 'additive'),
            'uncertainty_samples': model_config.get('uncertainty_samples', PROPHET_DEFAULTS['uncertainty_samples']),
            'yearly_seasonality': model_config.get('yearly_seasonality', 'auto'),
            'weekly_seasonality': model_config.get('weekly_seasonality', 'auto'),
            'daily_seasonality': model_config.get('daily_seasonality', 'auto'),
        }
        
        if model_config.get('mcmc_samples', 0) > 0:
            prophet_params['mcmc_samples'] = model_config['mcmc_samples']
        
        model = Prophet(**prophet_params)
        
        # Add custom seasonalities
        for seasonality in model_config.get('custom_seasonalities', []):
            model.add_seasonality(
                name=seasonality['name'],
                period=seasonality['period'],
                fourier_order=seasonality['fourier_order']
            )
        
        # Add holidays if specified
        if model_config.get('holidays_country'):
            import holidays
            country_holidays = holidays.country_holidays(model_config['holidays_country'])
            holidays_df = pd.DataFrame([
                {'ds': date, 'holiday': name}
                for date, name in country_holidays.items()
                if prophet_df['ds'].min() <= pd.to_datetime(date) <= prophet_df['ds'].max() + timedelta(days=base_config.get('forecast_periods', 30))
            ])
            if not holidays_df.empty:
                model.add_country_holidays(country_name=model_config['holidays_country'])
        
        # Fit the model
        with st.spinner("🔄 Training Prophet model..."):
            model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=base_config.get('forecast_periods', 30))
        
        # Handle logistic growth
        if model_config.get('growth') == 'logistic':
            cap_value = model_config.get('cap', prophet_df['y'].max() * 1.2)
            future['cap'] = cap_value
            prophet_df['cap'] = cap_value
            model.fit(prophet_df)  # Refit with cap
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Prepare output DataFrame
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_df.columns = [date_col, 'forecast', 'lower_bound', 'upper_bound']
        
        # Calculate metrics on training data
        train_forecast = forecast[:-base_config.get('forecast_periods', 30)]
        actual_values = prophet_df['y'].values
        predicted_values = train_forecast['yhat'].values
        
        metrics = calculate_forecast_metrics(actual_values, predicted_values)
        
        # Create plots
        plots = create_prophet_plots(model, forecast, prophet_df, base_config.get('forecast_periods', 30))
        
        # Add auto-tuning results to plots if available
        if model_config.get('auto_tune', False) and 'tuning_results' in locals():
            plots['tuning_results'] = tuning_results
        
        return forecast_df, metrics, plots
        
    except Exception as e:
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}

def auto_tune_prophet(df: pd.DataFrame, horizon: int = 30, 
                     parallel: str = 'processes') -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Auto-tune Prophet parameters using cross-validation
    """
    # Parameter grid for tuning
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    }
    
    best_params = {}
    best_mape = float('inf')
    results = []
    
    total_combinations = len(param_grid['changepoint_prior_scale']) * len(param_grid['seasonality_prior_scale'])
    progress_bar = st.progress(0)
    current_combination = 0
    
    for cps in param_grid['changepoint_prior_scale']:
        for sps in param_grid['seasonality_prior_scale']:
            current_combination += 1
            progress_bar.progress(current_combination / total_combinations)
            
            try:
                # Create and fit model
                model = Prophet(
                    changepoint_prior_scale=cps,
                    seasonality_prior_scale=sps,
                    uncertainty_samples=0  # Disable for faster tuning
                )
                model.fit(df)
                
                # Cross-validation
                cv_results = cross_validation(
                    model, 
                    horizon=f'{horizon} days',
                    initial=f'{len(df)//2} days',
                    period=f'{horizon//3} days',
                    parallel=parallel
                )
                
                # Calculate performance metrics
                cv_metrics = performance_metrics(cv_results)
                mape = cv_metrics['mape'].mean()
                
                results.append({
                    'changepoint_prior_scale': cps,
                    'seasonality_prior_scale': sps,
                    'mape': mape,
                    'mae': cv_metrics['mae'].mean(),
                    'rmse': cv_metrics['rmse'].mean()
                })
                
                if mape < best_mape:
                    best_mape = mape
                    best_params = {
                        'changepoint_prior_scale': cps,
                        'seasonality_prior_scale': sps
                    }
                    
            except Exception as e:
                st.warning(f"Parameter combination failed: cps={cps}, sps={sps}")
                continue
    
    progress_bar.empty()
    
    tuning_results = {
        'best_params': best_params,
        'best_mape': best_mape,
        'all_results': results
    }
    
    return best_params, tuning_results

def calculate_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate various forecast accuracy metrics"""
    metrics = {}
    
    # Remove any NaN values
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {'error': 'No valid data points for metric calculation'}
    
    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(actual - predicted))
    
    # Root Mean Square Error
    metrics['rmse'] = np.sqrt(np.mean((actual - predicted) ** 2))
    
    # Mean Absolute Percentage Error
    non_zero_mask = actual != 0
    if np.any(non_zero_mask):
        metrics['mape'] = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
    else:
        metrics['mape'] = np.inf
    
    # Mean Error (Bias)
    metrics['me'] = np.mean(actual - predicted)
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return metrics

def create_prophet_plots(model, forecast, train_df, forecast_periods):
    """Create Prophet-specific plots"""
    plots = {}
    
    try:
        # Main forecast plot
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=train_df['ds'],
            y=train_df['y'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        forecast_start_idx = len(train_df)
        forecast_data = forecast.iloc[forecast_start_idx:]
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red')
        ))
        
        # Confidence intervals
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data['ds'],
            y=forecast_data['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor='rgba(255,0,0,0.2)'
        ))
        
        fig_forecast.update_layout(
            title='Prophet Forecast',
            xaxis_title='Date',
            yaxis_title='Value',
            height=500
        )
        
        plots['forecast'] = fig_forecast
        
        # Components plot
        if hasattr(model, 'predict_seasonal_components'):
            components = model.predict_seasonal_components(forecast)
            fig_components = go.Figure()
            
            # Trend
            if 'trend' in components.columns:
                fig_components.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=components['trend'],
                    mode='lines',
                    name='Trend'
                ))
            
            # Weekly seasonality
            if 'weekly' in components.columns:
                fig_components.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=components['weekly'],
                    mode='lines',
                    name='Weekly'
                ))
            
            # Yearly seasonality
            if 'yearly' in components.columns:
                fig_components.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=components['yearly'],
                    mode='lines',
                    name='Yearly'
                ))
            
            fig_components.update_layout(
                title='Forecast Components',
                xaxis_title='Date',
                yaxis_title='Component Value',
                height=400
            )
            
            plots['components'] = fig_components
        
    except Exception as e:
        st.warning(f"Could not create some plots: {str(e)}")
    
    return plots
