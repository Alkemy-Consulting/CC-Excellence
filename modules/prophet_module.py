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
import logging
import hashlib
from functools import lru_cache
warnings.filterwarnings('ignore')

# Configure logging for Prophet module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def create_prophet_forecast_chart(model, forecast_df, actual_data, date_col, target_col, confidence_interval=0.8):
    """
    Create a standardized Prophet forecast chart following project conventions
    """
    try:
        logger.info(f"Creating Prophet forecast chart with confidence_interval={confidence_interval}")
        
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
        
        # Add changepoints (vertical lines) - Enhanced with proper error handling
        try:
            if hasattr(model, 'changepoints') and len(model.changepoints) > 0:
                logger.info(f"Adding {len(model.changepoints)} changepoints to chart")
                
                # Filter changepoints to those within the actual data range
                data_start = actual_data[date_col].min()
                data_end = actual_data[date_col].max()
                
                for i, changepoint in enumerate(model.changepoints):
                    # Convert changepoint to pandas Timestamp for comparison
                    cp_date = pd.to_datetime(changepoint)
                    
                    # Only add changepoints within the historical data range
                    if data_start <= cp_date <= data_end:
                        fig.add_vline(
                            x=cp_date,
                            line=dict(color='orange', width=1, dash='dot'),
                            opacity=0.7,
                            annotation_text=f"CP {i+1}",
                            annotation_position="top"
                        )
                logger.info("Changepoints added successfully")
            else:
                logger.info("No changepoints available to display")
        except Exception as cp_error:
            logger.warning(f"Could not display changepoints: {str(cp_error)}")
            # Continue without changepoints rather than failing completely

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
        logger.error(f"Error creating Prophet forecast chart: {str(e)}")
        st.error(f"Error creating Prophet forecast chart: {str(e)}")
        return None

def generate_data_hash(df: pd.DataFrame, date_col: str, target_col: str, model_config: dict, base_config: dict) -> str:
    """
    Generate a hash for caching purposes based on input data and configuration
    """
    try:
        # Create a string representation of the data and configs
        data_str = f"{df[[date_col, target_col]].to_string()}"
        config_str = f"{sorted(model_config.items())}{sorted(base_config.items())}"
        combined_str = data_str + config_str
        
        # Generate MD5 hash
        return hashlib.md5(combined_str.encode()).hexdigest()
    except Exception:
        # Return empty hash if generation fails
        return ""

@lru_cache(maxsize=32)
def _cached_prophet_model_params(
    seasonality_mode: str,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
    interval_width: float,
    yearly_seasonality: str,
    weekly_seasonality: str,
    daily_seasonality: str
) -> dict:
    """
    Cache Prophet model parameters to avoid repeated parameter processing
    """
    return {
        'yearly_seasonality': yearly_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'daily_seasonality': daily_seasonality,
        'seasonality_mode': seasonality_mode,
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale,
        'interval_width': interval_width
    }

def optimize_dataframe_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage for Prophet processing
    """
    try:
        optimized_df = df.copy()
        
        # Convert datetime column to more efficient format
        if 'ds' in optimized_df.columns:
            optimized_df['ds'] = pd.to_datetime(optimized_df['ds'])
        
        # Convert numeric columns to more efficient dtypes
        if 'y' in optimized_df.columns:
            optimized_df['y'] = pd.to_numeric(optimized_df['y'], downcast='float')
        
        logger.info(f"DataFrame optimized - Memory usage reduced from {df.memory_usage(deep=True).sum()} to {optimized_df.memory_usage(deep=True).sum()} bytes")
        return optimized_df
    except Exception as e:
        logger.warning(f"DataFrame optimization failed: {e}. Using original DataFrame.")
        return df

def validate_prophet_inputs(df: pd.DataFrame, date_col: str, target_col: str) -> None:
    """
    Robust input validation for Prophet forecasting
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    if not isinstance(date_col, str) or not date_col:
        raise ValueError("date_col must be a non-empty string")
    if not isinstance(target_col, str) or not target_col:
        raise ValueError("target_col must be a non-empty string")
    
    # Sanitize column names to prevent injection attacks
    safe_date_col = str(date_col).strip()
    safe_target_col = str(target_col).strip()
    
    if safe_date_col not in df.columns:
        raise KeyError(f"Date column not found in DataFrame")
    if safe_target_col not in df.columns:
        raise KeyError(f"Target column not found in DataFrame")
    
    # Check for minimum data requirements
    if len(df) < 10:
        raise ValueError(f"Insufficient data points: {len(df)} (minimum 10 required for Prophet)")
    
    # Check for maximum data size to prevent memory issues
    if len(df) > 100000:
        logger.warning(f"Large dataset detected: {len(df)} rows. Consider data sampling for better performance.")
    
    # Validate target column data type and missing values
    target_series = df[safe_target_col]
    if target_series.isna().sum() > len(df) * 0.3:
        raise ValueError(f"Too many missing values in target column: {target_series.isna().sum()}/{len(df)} (>30%)")
    
    # Check for numeric target values
    try:
        numeric_values = pd.to_numeric(target_series.dropna(), errors='raise')
        # Check for infinite or extremely large values
        if np.isinf(numeric_values).any():
            raise ValueError("Target column contains infinite values")
        if (np.abs(numeric_values) > 1e10).any():
            logger.warning("Target column contains very large values which may cause numerical instability")
    except (ValueError, TypeError):
        raise ValueError(f"Target column contains non-numeric values")
    
    # Validate date column
    try:
        date_values = pd.to_datetime(df[safe_date_col], errors='raise')
        # Check for reasonable date range
        if date_values.min().year < 1900 or date_values.max().year > 2100:
            logger.warning("Date values outside reasonable range (1900-2100)")
    except (ValueError, TypeError):
        raise ValueError(f"Date column contains invalid date values")
    
    # Check for zero variance
    numeric_target = pd.to_numeric(target_series.dropna(), errors='coerce')
    if numeric_target.std() == 0:
        raise ValueError("Target column has zero variance - cannot forecast constant values")

def run_prophet_forecast(df: pd.DataFrame, date_col: str, target_col: str, 
                        model_config: dict, base_config: dict):
    """
    Run Prophet forecast with enhanced validation and visualization
    """
    try:
        logger.info(f"Starting Prophet forecast - Data shape: {df.shape}, Date col: {date_col}, Target col: {target_col}")
        
        # STEP 1: Robust input validation
        validate_prophet_inputs(df, date_col, target_col)
        logger.info("Input validation passed")
        
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        # Optimize DataFrame for performance
        prophet_df = optimize_dataframe_for_prophet(prophet_df)
        logger.info(f"Prepared Prophet data - Shape: {prophet_df.shape}")
        
        # Split data for training
        train_size = base_config.get('train_size', 0.8)
        split_point = int(len(prophet_df) * train_size)
        train_df = prophet_df[:split_point]
        test_df = prophet_df[split_point:]
        logger.info(f"Data split - Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Get confidence interval from base_config, default to 80%
        confidence_interval = base_config.get('confidence_interval', 0.95)
        
        # Convert confidence interval to Prophet's interval_width format
        # Prophet expects interval_width as a decimal (0.8 for 80%, 0.95 for 95%)
        if confidence_interval > 1.0:
            # Handle percentage input (e.g., 95 -> 0.95)
            interval_width = confidence_interval / 100.0
        else:
            # Already in decimal format (e.g., 0.95)
            interval_width = confidence_interval
        
        # Ensure interval_width is within valid range [0.1, 0.99]
        interval_width = max(0.1, min(0.99, interval_width))
        logger.info(f"Using confidence interval: {confidence_interval} -> interval_width: {interval_width}")
        
        # Use cached model parameters for better performance
        cached_params = _cached_prophet_model_params(
            seasonality_mode=model_config.get('seasonality_mode', 'additive'),
            changepoint_prior_scale=model_config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=model_config.get('seasonality_prior_scale', 10.0),
            interval_width=interval_width,
            yearly_seasonality=str(model_config.get('yearly_seasonality', 'auto')),
            weekly_seasonality=str(model_config.get('weekly_seasonality', 'auto')),
            daily_seasonality=str(model_config.get('daily_seasonality', 'auto'))
        )
        
        # Initialize Prophet model
        logger.info("Initializing Prophet model with parameters:")
        for key, value in cached_params.items():
            logger.info(f"  - {key}: {value}")
        
        model = Prophet(**cached_params)
        
        # Add holidays if specified
        if model_config.get('add_holidays', False):
            logger.info("Adding holidays to Prophet model")
            try:
                import holidays
                
                # Get country from config, default to US
                country_code = model_config.get('holidays_country', 'US').upper()
                logger.info(f"Using holidays for country: {country_code}")
                
                # Mapping of country codes to holiday functions
                country_holidays_map = {
                    'US': holidays.US,
                    'CA': holidays.Canada,
                    'UK': holidays.UK,
                    'GB': holidays.UK,
                    'DE': holidays.Germany,
                    'FR': holidays.France,
                    'IT': holidays.Italy,
                    'ES': holidays.Spain,
                    'AU': holidays.Australia,
                    'JP': holidays.Japan,
                    'CN': holidays.China,
                    'IN': holidays.India
                }
                
                if country_code in country_holidays_map:
                    country_holidays = country_holidays_map[country_code]()
                else:
                    logger.warning(f"Country code '{country_code}' not supported, using US holidays as fallback")
                    country_holidays = holidays.US()
                
                holiday_df = pd.DataFrame({
                    'holiday': list(country_holidays.keys()),
                    'ds': pd.to_datetime(list(country_holidays.keys())),
                })
                holiday_df = holiday_df[holiday_df['ds'].between(train_df['ds'].min(), train_df['ds'].max())]
                
                if not holiday_df.empty:
                    # Use add_country_holidays for supported countries, otherwise add manually
                    supported_countries = ['US', 'CA', 'UK', 'GB']
                    if country_code in supported_countries:
                        model.add_country_holidays(country_name=country_code)
                    else:
                        # Add holidays manually for other countries
                        for _, holiday_row in holiday_df.iterrows():
                            model.add_seasonality(
                                name=f'holiday_{holiday_row["ds"].strftime("%Y%m%d")}',
                                period=365.25,
                                fourier_order=1
                            )
                    
                    logger.info(f"Added {len(holiday_df)} {country_code} holidays to model")
                else:
                    logger.warning("No holidays found in the training date range")
                    
            except ImportError:
                logger.error("holidays package not available. Install with: pip install holidays")
            except Exception as e:
                logger.error(f"Error adding holidays: {str(e)}")
        else:
            logger.info("Holidays not enabled for this forecast")
        
        # Fit the model
        logger.info("Fitting Prophet model...")
        model.fit(train_df)
        logger.info("Prophet model fitted successfully")
        
        # Create future dataframe for forecasting
        forecast_periods = base_config.get('forecast_periods', 30)
        logger.info(f"Creating forecast for {forecast_periods} periods")
        future = model.make_future_dataframe(periods=forecast_periods)
        
        # Generate forecast
        logger.info("Generating Prophet forecast...")
        forecast = model.predict(future)
        logger.info(f"Forecast generated successfully - Shape: {forecast.shape}")
        
        # Calculate metrics on test set if available
        metrics = {}
        if not test_df.empty:
            logger.info("Calculating metrics on test set")
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
                    logger.info(f"Metrics calculated - MAPE: {mape:.2f}%, MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.3f}")
            else:
                logger.warning("No test forecast data available for metrics calculation")
        else:
            logger.info("No test data available for metrics calculation")
        
        # Provide fallback metrics if calculation fails
        if not metrics:
            metrics = {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
            logger.warning("Using fallback metrics (all zeros)")
        
        # Create the standardized forecast chart with confidence interval
        logger.info("Creating forecast visualization...")
        forecast_plot = create_prophet_forecast_chart(model, forecast, prophet_df, 'ds', 'y', interval_width)
        
        # Prepare plots dictionary
        plots = {}
        if forecast_plot:
            plots['forecast_plot'] = forecast_plot
            logger.info("Forecast plot created successfully")
        else:
            logger.warning("Failed to create forecast plot")
        
        # Create forecast dataframe for output
        forecast_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_output.columns = [date_col, f'{target_col}_forecast', f'{target_col}_lower', f'{target_col}_upper']
        logger.info(f"Prophet forecast completed successfully - Output shape: {forecast_output.shape}")
        
        return forecast_output, metrics, plots
        
    except Exception as e:
        logger.error(f"Error in Prophet forecasting: {str(e)}")
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}
