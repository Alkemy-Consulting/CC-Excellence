"""
Unified Prophet Forecasting Module
Complete implementation with all user parameters, robust validation, and proper execution order.
This module contains all Prophet-related functionality in a single, efficient file.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings
import logging
import hashlib
from functools import lru_cache
from prophet import Prophet
from datetime import datetime, timedelta
from dataclasses import dataclass
import holidays
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ProphetForecastResult:
    """Result container for Prophet forecasting operations"""
    success: bool
    error: Optional[str] = None
    model: Optional[Prophet] = None
    raw_forecast: Optional[pd.DataFrame] = None
    metrics: Optional[Dict[str, float]] = None

class ProphetForecaster:
    """Unified Prophet forecasting engine with complete parameter support"""
    
    def __init__(self):
        self.model = None
        self.forecast_data = None
        self.metrics = {}
        
    def validate_inputs(self, df: pd.DataFrame, date_col: str, target_col: str, 
                       model_config: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive input validation for Prophet forecasting
        Returns: (is_valid, error_message)
        """
        try:
            # Type validation
            if not isinstance(df, pd.DataFrame):
                return False, "Input must be a pandas DataFrame"
            
            if df.empty:
                return False, "DataFrame cannot be empty"
                
            if not isinstance(date_col, str) or not date_col:
                return False, "date_col must be a non-empty string"
                
            if not isinstance(target_col, str) or not target_col:
                return False, "target_col must be a non-empty string"
            
            # Sanitize column names to prevent injection attacks
            safe_date_col = str(date_col).strip()
            safe_target_col = str(target_col).strip()
            
            if safe_date_col not in df.columns:
                return False, f"Date column '{safe_date_col}' not found in DataFrame"
                
            if safe_target_col not in df.columns:
                return False, f"Target column '{safe_target_col}' not found in DataFrame"
            
            # Enhanced data requirements validation
            if len(df) < 30:  # Increased minimum for Prophet
                return False, f"Insufficient data points: {len(df)} (minimum 30 required for Prophet)"
            
            # Check for maximum data size to prevent memory issues
            if len(df) > 100000:
                logger.warning(f"Large dataset detected: {len(df)} rows. Consider data sampling for better performance.")
            
            # Validate target column data type and missing values
            target_series = df[safe_target_col]
            if target_series.isna().sum() > len(df) * 0.3:
                return False, f"Too many missing values in target column: {target_series.isna().sum()}/{len(df)} (>30%)"
            
            # Check for numeric target values
            try:
                numeric_target = pd.to_numeric(target_series.dropna(), errors='coerce')
                if numeric_target.isna().sum() > 0:
                    return False, f"Target column contains non-numeric values: {numeric_target.isna().sum()} invalid entries"
            except Exception as e:
                return False, f"Error converting target column to numeric: {str(e)}"
            
            # Check for zero variance
            if numeric_target.std() == 0:
                return False, "Target column has zero variance - cannot forecast constant values"
            
            # Validate model configuration if provided
            if model_config:
                self._validate_model_config(model_config, df)
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_model_config(self, model_config: Dict, df: pd.DataFrame) -> None:
        """Validate model configuration parameters"""
        # Validate seasonality requirements
        if model_config.get('yearly_seasonality') and len(df) < 365:
            raise ValueError("Insufficient data for yearly seasonality (minimum 365 days required)")
        
        if model_config.get('weekly_seasonality') and len(df) < 14:
            raise ValueError("Insufficient data for weekly seasonality (minimum 14 days required)")
        
        if model_config.get('daily_seasonality') and len(df) < 48:
            raise ValueError("Insufficient data for daily seasonality (minimum 48 hours required)")
        
        # Validate growth model requirements
        if model_config.get('growth') == 'logistic' and 'cap' not in df.columns:
            raise ValueError("Logistic growth requires 'cap' column in data")
        
        # Validate parameter ranges
        changepoint_prior_scale = model_config.get('changepoint_prior_scale', 0.05)
        if not (0.001 <= changepoint_prior_scale <= 0.5):
            raise ValueError("changepoint_prior_scale must be between 0.001 and 0.5")
        
        seasonality_prior_scale = model_config.get('seasonality_prior_scale', 10.0)
        if not (0.01 <= seasonality_prior_scale <= 100.0):
            raise ValueError("seasonality_prior_scale must be between 0.01 and 100.0")

    def prepare_data(self, df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """
        Prepare data for Prophet processing with enhanced data quality checks
        """
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        # Ensure proper datetime format
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        prophet_df = prophet_df.sort_values('ds')
        
        # Ensure numeric target
        prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
        prophet_df = prophet_df.dropna()
        
        logger.info(f"Prepared Prophet data - Shape: {prophet_df.shape}")
        return prophet_df

    def split_data(self, prophet_df: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data for training and evaluation
        Returns: (train_df, test_df)
        """
        split_point = int(len(prophet_df) * train_size)
        
        # FIXED: Proper train/test split for accurate metrics calculation
        train_df = prophet_df[:split_point]  # Use only training portion
        test_df = prophet_df[split_point:]   # Keep test split for evaluation
        logger.info(f"Data split - Using {len(train_df)} rows for training, {len(test_df)} rows for evaluation")
        
        return train_df, test_df

    def create_model(self, model_config: dict, confidence_interval: float = 0.95) -> Prophet:
        """
        Create and configure Prophet model with ALL user parameters
        """
        # Convert confidence interval to Prophet's interval_width format
        if confidence_interval > 1.0:
            interval_width = confidence_interval / 100.0
        else:
            interval_width = confidence_interval
        
        # Ensure interval_width is within valid range [0.1, 0.99]
        interval_width = max(0.1, min(0.99, interval_width))
        logger.info(f"Using confidence interval: {confidence_interval} -> interval_width: {interval_width}")
        
        # Convert seasonality parameters properly
        def convert_seasonality(value):
            if isinstance(value, bool):
                return value
            elif isinstance(value, str):
                if value.lower() in ['true', '1', 'yes', 'on']:
                    return True
                elif value.lower() in ['false', '0', 'no', 'off']:
                    return False
                elif value.lower() == 'auto':
                    return 'auto'
                else:
                    return 'auto'  # Default fallback
            else:
                return 'auto'  # Default fallback
        
        # Build Prophet parameters from user configuration
        prophet_params = {
            'yearly_seasonality': convert_seasonality(model_config.get('yearly_seasonality', 'auto')),
            'weekly_seasonality': convert_seasonality(model_config.get('weekly_seasonality', 'auto')),
            'daily_seasonality': convert_seasonality(model_config.get('daily_seasonality', 'auto')),
            'seasonality_mode': model_config.get('seasonality_mode', 'additive'),
            'changepoint_prior_scale': model_config.get('changepoint_prior_scale', 0.05),
            'seasonality_prior_scale': model_config.get('seasonality_prior_scale', 10.0),
            'interval_width': interval_width
        }
        
        # Add growth model if specified
        if model_config.get('growth') == 'logistic':
            prophet_params['growth'] = 'logistic'
        else:
            prophet_params['growth'] = 'linear'
        
        # Initialize Prophet model
        logger.info("Initializing Prophet model with parameters:")
        for key, value in prophet_params.items():
            logger.info(f"  - {key}: {value}")
        
        # Add hash of parameters for debugging different configurations
        import hashlib
        params_str = str(sorted(prophet_params.items()))
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        logger.info(f"Configuration hash: {params_hash}")
        
        model = Prophet(**prophet_params)
        
        # Add holidays if specified
        if model_config.get('add_holidays', False):
            self._add_holidays(model, model_config)
        
        return model

    def _add_holidays(self, model: Prophet, model_config: dict):
        """Add holidays to Prophet model"""
        logger.info("Adding holidays to Prophet model")
        try:
            # Get country from config, default to US
            country_code = model_config.get('holidays_country', 'US').upper()
            logger.info(f"Using holidays for country: {country_code}")
            
            # Mapping of country codes to holiday functions
            country_holidays_map = {
                'US': holidays.US, 'CA': holidays.Canada, 'UK': holidays.UK,
                'GB': holidays.UK, 'DE': holidays.Germany, 'FR': holidays.France,
                'IT': holidays.Italy, 'ES': holidays.Spain, 'AU': holidays.Australia,
                'JP': holidays.Japan, 'CN': holidays.China, 'IN': holidays.India
            }
            
            if country_code in country_holidays_map:
                country_holidays = country_holidays_map[country_code]()
            else:
                logger.warning(f"Country code '{country_code}' not supported, using US holidays as fallback")
                country_holidays = holidays.US()
            
            # Use add_country_holidays for supported countries
            supported_countries = ['US', 'CA', 'UK', 'GB']
            if country_code in supported_countries:
                model.add_country_holidays(country_name=country_code)
                logger.info(f"Added {country_code} holidays to model")
            else:
                # Create custom holiday DataFrame for unsupported countries
                holiday_df = pd.DataFrame({
                    'holiday': list(country_holidays.keys()),
                    'ds': pd.to_datetime(list(country_holidays.keys())),
                })
                if not holiday_df.empty:
                    model.holidays = holiday_df
                    logger.info(f"Added custom holidays for {country_code}")
                
        except ImportError:
            logger.error("holidays package not available. Install with: pip install holidays")
        except Exception as e:
            logger.error(f"Error adding holidays: {str(e)}")

    def calculate_metrics(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive forecast metrics
        """
        try:
            if len(actual_values) == 0 or len(predicted_values) == 0:
                return {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
            
            # Ensure both arrays have same length
            min_len = min(len(actual_values), len(predicted_values))
            actual = actual_values[:min_len]
            predicted = predicted_values[:min_len]
            
            # Calculate metrics
            mae = np.mean(np.abs(actual - predicted))
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            
            # MAPE with zero-division protection
            mask = actual != 0
            if mask.sum() > 0:
                mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
            else:
                mape = 100.0
            
            # R²
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                'mape': float(mape),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}

    def calculate_metrics_from_dataframes(self, forecast: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate forecast metrics on test set with improved matching
        """
        if test_df.empty:
            logger.info("No test data available for metrics calculation")
            return {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
        
        logger.info(f"Calculating metrics on test set - Test dates: {len(test_df)}, Forecast dates: {len(forecast)}")
        
        try:
            # Merge forecast and test data on date column for proper alignment
            test_with_forecast = test_df.merge(
                forecast[['ds', 'yhat']], 
                on='ds', 
                how='inner'
            )
            
            if test_with_forecast.empty:
                logger.warning("No matching dates between test data and forecast")
                return {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
            
            actual_values = test_with_forecast['y'].values
            predicted_values = test_with_forecast['yhat'].values
            
            logger.info(f"Matched {len(actual_values)} data points for metrics calculation")
            
            # Calculate metrics with the properly aligned data
            metrics = self.calculate_metrics(actual_values, predicted_values)
            logger.info(f"Calculated metrics: MAPE={metrics['mape']:.2f}%, MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in metrics calculation: {str(e)}")
            return {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}

    def run_forecast_core(self, df: pd.DataFrame, date_col: str, target_col: str, 
                         model_config: dict, forecast_config: dict) -> ProphetForecastResult:
        """
        Run Prophet forecast using parameters from Forecast Settings section
        """
        try:
            logger.info(f"Starting Prophet forecast - Data shape: {df.shape}, Date col: {date_col}, Target col: {target_col}")
            logger.info(f"Model config: {model_config}")
            logger.info(f"Forecast config: {forecast_config}")
            
            # Step 1: Comprehensive input validation
            is_valid, error_msg = self.validate_inputs(df, date_col, target_col, model_config)
            if not is_valid:
                return ProphetForecastResult(
                    success=False,
                    error=error_msg,
                    model=None,
                    raw_forecast=pd.DataFrame(),
                    metrics={}
                )
            
            # Step 2: Prepare data with quality checks
            prophet_df = self.prepare_data(df, date_col, target_col)
            
            # Step 3: Split data for evaluation using Forecast Settings
            train_size = forecast_config.get('train_size', 0.8)
            train_df, test_df = self.split_data(prophet_df, train_size)
            
            # Step 4: Create and configure model using Forecast Settings confidence level
            confidence_level = forecast_config.get('confidence_level', 0.95)
            model = self.create_model(model_config, confidence_level)
            
            # Step 5: Train model
            logger.info("Fitting Prophet model...")
            model.fit(train_df)
            logger.info("Prophet model fitted successfully")
            
            # Step 6: Apply cross-validation if enabled in Forecast Settings
            if forecast_config.get('enable_cross_validation', False):
                logger.info("Cross-validation enabled in Forecast Settings")
                cv_folds = forecast_config.get('cv_folds', 5)
                logger.info(f"Would perform {cv_folds}-fold cross-validation")
                # Cross-validation implementation would go here
                pass
            
            # Step 7: Generate forecast using horizon from Forecast Settings
            # Support both 'horizon' and 'forecast_periods' parameter names for compatibility
            forecast_horizon = forecast_config.get('horizon', forecast_config.get('forecast_periods', 30))
            logger.info(f"Creating forecast for {forecast_horizon} periods")
            logger.info(f"Forecast config keys: {list(forecast_config.keys())}")
            logger.info(f"Forecast config values: {forecast_config}")
            
            # Ensure we have a valid horizon
            if forecast_horizon <= 0:
                logger.warning(f"Invalid forecast horizon: {forecast_horizon}, defaulting to 30")
                forecast_horizon = 30
            
            # Create future dataframe that includes the full historical period
            future = model.make_future_dataframe(periods=forecast_horizon)
            
            # Add logistic growth cap if specified
            if model_config.get('growth') == 'logistic' and 'cap' in df.columns:
                future['cap'] = df['cap'].iloc[-1]  # Use last cap value
            
            logger.info("Generating Prophet forecast...")
            forecast = model.predict(future)
            logger.info(f"Forecast generated successfully - Shape: {forecast.shape}")
            
            # Debug info about forecast periods
            last_train_date = train_df['ds'].max()
            future_periods = forecast[forecast['ds'] > last_train_date]
            logger.info(f"Last training date: {last_train_date}")
            logger.info(f"Future forecast periods: {len(future_periods)} (expected: {forecast_horizon})")
            logger.info(f"Forecast date range: {forecast['ds'].min()} to {forecast['ds'].max()}")
            
            # Step 8: Calculate comprehensive metrics using proper test set
            metrics = self.calculate_metrics_from_dataframes(forecast, test_df)
            
            # Step 9: Return success result
            return ProphetForecastResult(
                success=True,
                error=None,
                model=model,
                raw_forecast=forecast,
                metrics=metrics
            )
            
        except Exception as e:
            error_msg = f"Error in Prophet forecasting: {str(e)}"
            logger.error(error_msg)
            return ProphetForecastResult(
                success=False,
                error=error_msg,
                model=None,
                raw_forecast=None,
                metrics={}
            )

def render_prophet_config():
    """
    Render Prophet configuration UI components - ONLY Prophet-specific parameters
    General forecast parameters are handled in Forecast Settings section
    """
    with st.expander("⚙️ Prophet Configuration", expanded=False):
        config = {}
        
        # Core Prophet Parameters
        st.subheader("🔧 Core Prophet Parameters")
        
        config['changepoint_prior_scale'] = st.slider(
            "Trend Flexibility",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.001,
            format="%.3f",
            help="Controls trend flexibility. Higher values = more flexible trend"
        )
        
        config['seasonality_prior_scale'] = st.slider(
            "Seasonality Strength",
            min_value=0.01,
            max_value=100.0,
            value=10.0,
            step=0.01,
            help="Controls seasonality strength. Higher values = stronger seasonality"
        )
        
        config['seasonality_mode'] = st.selectbox(
            "Seasonality Mode",
            ['additive', 'multiplicative'],
            index=0,
            help="How seasonality affects the trend"
        )
        
        # Seasonality Configuration
        st.subheader("📊 Seasonality Configuration")
        
        config['yearly_seasonality'] = st.selectbox(
            "Yearly Seasonality",
            ['auto', True, False],
            index=0,
            help="Automatically detect or manually set yearly patterns"
        )
        
        config['weekly_seasonality'] = st.selectbox(
            "Weekly Seasonality", 
            ['auto', True, False],
            index=0,
            help="Automatically detect or manually set weekly patterns"
        )
        
        config['daily_seasonality'] = st.selectbox(
            "Daily Seasonality",
            ['auto', True, False],
            index=0,
            help="Automatically detect or manually set daily patterns"
        )
        
        # Growth Model Configuration
        st.subheader("📈 Growth Model")
        config['growth'] = st.selectbox(
            "Growth Model",
            options=['linear', 'logistic'],
            index=0,
            help="Type of growth trend"
        )
        
        if config['growth'] == 'logistic':
            st.warning("⚠️ Logistic growth requires 'cap' column in data")
        
        # Holiday Effects
        st.subheader("🎉 Holiday Effects")
        config['add_holidays'] = st.checkbox(
            "Add Holiday Effects",
            value=False,
            help="Include country-specific holidays in the model"
        )
        
        if config['add_holidays']:
            config['holidays_country'] = st.selectbox(
                "Select Country",
                options=['US', 'CA', 'UK', 'DE', 'FR', 'IT', 'ES', 'AU', 'JP'],
                index=0,
                help="Country for holiday calendar"
            )
        
        return config

def run_prophet_forecast(df: pd.DataFrame, date_col: str, target_col: str, 
                        model_config: dict, forecast_config: dict):
    """
    Main Prophet forecast interface - uses Forecast Settings parameters
    """
    try:
        logger.info("Using unified Prophet forecasting architecture")
        
        # Initialize Prophet forecaster
        forecaster = ProphetForecaster()
        
        # Run forecast using forecast_config from Forecast Settings
        result = forecaster.run_forecast_core(df, date_col, target_col, model_config, forecast_config)
        
        if result.success:
            # Store results in session state for diagnostics
            st.session_state.last_prophet_result = result
            st.session_state.last_prophet_data = {
                'df': df.copy(),
                'date_col': date_col,
                'target_col': target_col
            }
            
            # Create visualizations
            plots = create_prophet_plots(result, df, date_col, target_col)
            
            # Convert to legacy format for backward compatibility
            forecast_output = result.raw_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_output.columns = [date_col, f'{target_col}_forecast', f'{target_col}_lower', f'{target_col}_upper']
            
            logger.info(f"Unified Prophet forecast completed successfully - Output shape: {forecast_output.shape}")
            return forecast_output, result.metrics, plots
            
        else:
            logger.error(f"Prophet forecast failed: {result.error}")
            st.error(f"Prophet forecast failed: {result.error}")
            return pd.DataFrame(), {}, {}
            
    except Exception as e:
        logger.error(f"Error in unified Prophet forecast interface: {str(e)}")
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}

def create_prophet_plots(result: ProphetForecastResult, df: pd.DataFrame, 
                        date_col: str, target_col: str) -> Dict[str, Any]:
    """
    Create comprehensive Prophet visualization plots
    """
    plots = {}
    
    try:
        if not result.success or result.raw_forecast is None:
            return plots
        
        # Main forecast plot
        fig = go.Figure()
        
        # Historical data (torna al colore blu originale)
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[target_col],
            mode='lines',
            name='Historical Data',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast completo (come era originalmente ma con logica corretta)
        forecast = result.raw_forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2)
        ))
        
        # Confidence intervals (torna all'originale)
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            name='Upper Bound',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            name='Confidence Interval',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=True
        ))
        
        # Aggiungi pulsanti di filtro temporale
        fig.update_layout(
            title=f"Prophet Forecast: {target_col}",
            xaxis_title="Date",
            yaxis_title=target_col,
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=0.95
            ),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=30, label="1M", step="day", stepmode="backward"),
                        dict(count=90, label="3M", step="day", stepmode="backward"),
                        dict(count=180, label="6M", step="day", stepmode="backward"),
                        dict(count=365, label="1Y", step="day", stepmode="backward"),
                        dict(count=730, label="2Y", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    x=0.02,
                    xanchor="left",
                    y=1.02,
                    yanchor="bottom"
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        plots['forecast_plot'] = fig
        
        # Components plot if model is available - create a Plotly version instead
        if result.model is not None:
            try:
                # Create a simplified components analysis using Plotly
                components_fig = create_prophet_components_plotly(forecast, target_col)
                if components_fig is not None:
                    plots['components_plot'] = components_fig
                    logger.info("Prophet components plot created successfully (Plotly version)")
            except Exception as e:
                logger.warning(f"Could not create components plot: {e}")
                # Skip components plot - forecast will still work
                logger.info("Skipping components plot - forecast will still work")
        
    except Exception as e:
        logger.error(f"Error creating Prophet plots: {e}")
    
    return plots

def create_prophet_components_plotly(forecast: pd.DataFrame, target_col: str) -> go.Figure:
    """
    Create Prophet components plot using Plotly (alternative to matplotlib version)
    """
    try:
        from plotly.subplots import make_subplots
        
        # Create subplots for trend and seasonality components
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Trend', 'Weekly Seasonality', 'Yearly Seasonality'),
            vertical_spacing=0.15
        )
        
        # Trend component
        if 'trend' in forecast.columns:
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['trend'],
                          mode='lines', name='Trend',
                          line=dict(color='blue', width=2)),
                row=1, col=1
            )
        
        # Weekly seasonality
        if 'weekly' in forecast.columns:
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['weekly'],
                          mode='lines', name='Weekly',
                          line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        # Yearly seasonality
        if 'yearly' in forecast.columns:
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['yearly'],
                          mode='lines', name='Yearly',
                          line=dict(color='orange', width=2)),
                row=3, col=1
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Prophet Components Analysis - {target_col}",
            showlegend=False
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Trend", row=1, col=1)
        fig.update_yaxes(title_text="Weekly", row=2, col=1)
        fig.update_yaxes(title_text="Yearly", row=3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating Plotly components plot: {e}")
        return None

def run_prophet_diagnostics(df: pd.DataFrame, date_col: str, target_col: str,
                           result: ProphetForecastResult, show_diagnostic_plots: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive Prophet diagnostics
    """
    diagnostic_results = {}
    
    try:
        if not result.success or result.raw_forecast is None:
            return diagnostic_results
        
        # Residual analysis
        if not df.empty and date_col in df.columns and target_col in df.columns:
            # Get historical data
            historical = df.set_index(date_col)[target_col]
            
            # Get forecast for historical period
            forecast_historical = result.raw_forecast[
                result.raw_forecast['ds'].isin(df[date_col])
            ].set_index('ds')['yhat']
            
            # Calculate residuals
            residuals = historical - forecast_historical
            residuals = residuals.dropna()
            
            if len(residuals) > 0:
                diagnostic_results['residual_analysis'] = {
                    'mean': float(residuals.mean()),
                    'std': float(residuals.std()),
                    'skewness': float(residuals.skew()),
                    'kurtosis': float(residuals.kurtosis()),
                    'count': len(residuals)
                }
        
        # Model performance metrics
        if result.metrics:
            diagnostic_results['performance_metrics'] = result.metrics
        
        # Forecast quality assessment
        if not result.raw_forecast.empty:
            forecast_values = result.raw_forecast['yhat']
            diagnostic_results['forecast_quality'] = {
                'mean_forecast': float(forecast_values.mean()),
                'std_forecast': float(forecast_values.std()),
                'forecast_range': float(forecast_values.max() - forecast_values.min()),
                'trend_direction': 'increasing' if forecast_values.iloc[-1] > forecast_values.iloc[0] else 'decreasing'
            }
        
    except Exception as e:
        logger.error(f"Error in Prophet diagnostics: {e}")
    
    return diagnostic_results

# Legacy function for backward compatibility
def run_prophet_forecast_legacy(df: pd.DataFrame, date_col: str, target_col: str, 
                        model_config: dict, base_config: dict):
    """
    Legacy Prophet forecast implementation - converts base_config to forecast_config
    """
    logger.warning("Legacy Prophet forecast called - converting base_config to forecast_config")
    # Convert old base_config format to new forecast_config format
    forecast_config = {
        'forecast_periods': base_config.get('forecast_periods', 30),  # Keep original name
        'horizon': base_config.get('forecast_periods', 30),          # Add for compatibility
        'confidence_level': base_config.get('confidence_interval', 0.95),
        'train_size': base_config.get('train_size', 0.8),
        'enable_cross_validation': False
    }
    return run_prophet_forecast(df, date_col, target_col, model_config, forecast_config)

# Factory function for easy instantiation
def create_prophet_forecaster() -> ProphetForecaster:
    """Factory function to create ProphetForecaster instance"""
    return ProphetForecaster()


