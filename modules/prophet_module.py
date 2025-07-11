from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import warnings
warnings.filterwarnings('ignore')

def create_prophet_forecast_chart(model, forecast_df, actual_data, date_col, target_col, confidence_interval=0.8):
    """
    Create a standardized Prophet forecast chart following project conventions
    """
    try:
        # Calculate confidence percentage for display
        confidence_percentage = int(confidence_interval * 100)
        
        # Ensure proper datetime conversion for all data
        forecast_df = forecast_df.copy()
        actual_data = actual_data.copy()
        
        # Convert date columns to datetime if they aren't already
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        actual_data[date_col] = pd.to_datetime(actual_data[date_col])
        
        # Create the main plot
        fig = go.Figure()
        
        # Add actual values (white points for training period)
        fig.add_trace(go.Scatter(
            x=actual_data[date_col],
            y=actual_data[target_col],
            mode='markers',
            name='Actual Values',
            marker=dict(color='white', size=4, line=dict(color='black', width=1)),
            hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add predictions (blue line)
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat'],
            mode='lines',
            name='Predictions',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Prediction</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add uncertainty interval (blue shade) - 80% interval
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 100, 255, 0.2)',
            name=f'{confidence_percentage}% Confidence Interval',
            hovertemplate=f'<b>{confidence_percentage}% Confidence Interval</b><br>Date: %{{x}}<br>Upper: %{{text}}<br>Lower: %{{y:.2f}}<extra></extra>',
            text=forecast_df['yhat_upper'].round(2)
        ))
        
        # Add trend line (red line)
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2),
            hovertemplate='<b>Trend</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Add changepoints (vertical lines) - Disabled due to compatibility issues
        # Note: Changepoints display has been temporarily disabled to prevent type errors
        # This can be re-enabled once the underlying issue is resolved
        if False:  # hasattr(model, 'changepoints') and len(model.changepoints) > 0:
            try:
                pass
            except Exception as cp_error:
                pass

        # Apply same chart formatting as Historical Time Series with Trend Analysis (Tab 1)
        fig.update_layout(
            title="Prophet Forecast Results",
            xaxis_title="Date",
            yaxis_title=target_col,
            height=500,  # Same height as PLOT_CONFIG['height']
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",  # Align legend to the right
                x=0.95  # Position at 95% of the width (right side)
            ),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=2, label="2Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    x=0.02,  # Position range selector at left (2% from left edge)
                    xanchor="left",  # Anchor to the left
                    y=1.02,  # Same height as legend
                    yanchor="bottom"
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Prophet forecast chart: {str(e)}")
        return None

def run_prophet_forecast(df: pd.DataFrame, date_col: str, target_col: str, 
                        model_config: dict, base_config: dict):
    """
    Run Prophet forecast with enhanced visualization
    """
    try:
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        # Split data for training
        train_size = base_config.get('train_size', 0.8)
        split_point = int(len(prophet_df) * train_size)
        train_df = prophet_df[:split_point]
        test_df = prophet_df[split_point:]
        
        # Get confidence interval from base_config, default to 80%
        confidence_interval = base_config.get('confidence_interval', 0.95)
        # Convert from 95% to Prophet's interval_width (80% is common default)
        interval_width = 0.8 if confidence_interval == 0.95 else confidence_interval * 0.85
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=model_config.get('yearly_seasonality', 'auto'),
            weekly_seasonality=model_config.get('weekly_seasonality', 'auto'),
            daily_seasonality=model_config.get('daily_seasonality', 'auto'),
            seasonality_mode=model_config.get('seasonality_mode', 'additive'),
            changepoint_prior_scale=model_config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=model_config.get('seasonality_prior_scale', 10.0),
            interval_width=interval_width
        )
        
        # Add holidays if specified
        if model_config.get('add_holidays', False):
            import holidays
            country_holidays = holidays.US()  # Default to US holidays
            holiday_df = pd.DataFrame({
                'holiday': list(country_holidays.keys()),
                'ds': pd.to_datetime(list(country_holidays.keys())),
            })
            holiday_df = holiday_df[holiday_df['ds'].between(train_df['ds'].min(), train_df['ds'].max())]
            if not holiday_df.empty:
                model.add_country_holidays(country_name='US')
        
        # Fit the model
        model.fit(train_df)
        
        # Create future dataframe for forecasting
        forecast_periods = base_config.get('forecast_periods', 30)
        future = model.make_future_dataframe(periods=forecast_periods)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Calculate metrics on test set if available
        metrics = {}
        if not test_df.empty:
            # Get predictions for test period
            test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
            if not test_forecast.empty:
                actual_values = test_df['y'].values
                predicted_values = test_forecast['yhat'].values
                
                # Align lengths
                min_len = min(len(actual_values), len(predicted_values))
                actual_values = actual_values[:min_len]
                predicted_values = predicted_values[:min_len]
                
                if len(actual_values) > 0:
                    # Calculate metrics
                    mae = np.mean(np.abs(actual_values - predicted_values))
                    mse = np.mean((actual_values - predicted_values) ** 2)
                    rmse = np.sqrt(mse)
                    
                    # MAPE with zero-division protection
                    mask = actual_values != 0
                    if mask.sum() > 0:
                        mape = np.mean(np.abs((actual_values[mask] - predicted_values[mask]) / actual_values[mask])) * 100
                    else:
                        mape = 100.0
                    
                    # R²
                    ss_res = np.sum((actual_values - predicted_values) ** 2)
                    ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    metrics = {
                        'mape': float(mape),
                        'mae': float(mae),
                        'rmse': float(rmse),
                        'r2': float(r2)
                    }
        
        # Provide fallback metrics if calculation fails
        if not metrics:
            metrics = {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
        
        # Create the standardized forecast chart with confidence interval
        forecast_plot = create_prophet_forecast_chart(model, forecast, prophet_df, 'ds', 'y', interval_width)
        
        # Prepare plots dictionary
        plots = {}
        if forecast_plot:
            plots['forecast_plot'] = forecast_plot
        
        # Create forecast dataframe for output
        forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_output.columns = [date_col, f'{target_col}_forecast', f'{target_col}_lower', f'{target_col}_upper']
        
        return forecast_output, metrics, plots
        
    except Exception as e:
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}
