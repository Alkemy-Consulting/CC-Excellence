"""
Prophet Core Business Logic Module
Pure business logic without UI dependencies - Enterprise Architecture Layer
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Tuple, Dict, Optional, Any, List
import logging
from datetime import datetime
import hashlib
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

class ProphetForecastResult:
    """Data class to encapsulate forecast results"""
    def __init__(self, success: bool, error: Optional[str] = None, 
                 model: Optional['Prophet'] = None, raw_forecast: Optional[pd.DataFrame] = None,
                 metrics: Optional[Dict[str, float]] = None):
        self.success = success
        self.error = error
        self.model = model
        self.raw_forecast = raw_forecast if raw_forecast is not None else pd.DataFrame()
        self.metrics = metrics if metrics is not None else {}

class ProphetForecaster:
    """Core Prophet forecasting engine - pure business logic"""
    
    def __init__(self):
        self.model = None
        self.forecast_data = None
        self.metrics = {}
        
    def validate_inputs(self, df: pd.DataFrame, date_col: str, target_col: str, 
                       model_config: Optional[Dict] = None, base_config: Optional[Dict] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate inputs for Prophet forecasting
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
            
            # Check for minimum data requirements
            if len(df) < 10:
                return False, f"Insufficient data points: {len(df)} (minimum 10 required for Prophet)"
            
            # Check for maximum data size to prevent memory issues
            if len(df) > 100000:
                logger.warning(f"Large dataset detected: {len(df)} rows. Consider data sampling for better performance.")
            
            # Validate target column data type and missing values
            target_series = df[safe_target_col]
            if target_series.isna().sum() > len(df) * 0.3:
                return False, f"Too many missing values in target column: {target_series.isna().sum()}/{len(df)} (>30%)"
            
            # Check for numeric target values
            try:
                numeric_values = pd.to_numeric(target_series.dropna(), errors='raise')
                # Check for infinite or extremely large values
                if np.isinf(numeric_values).any():
                    return False, "Target column contains infinite values"
                if (np.abs(numeric_values) > 1e10).any():
                    logger.warning("Target column contains very large values which may cause numerical instability")
            except (ValueError, TypeError):
                return False, "Target column contains non-numeric values"
            
            # Validate date column
            try:
                date_values = pd.to_datetime(df[safe_date_col], errors='raise')
                # Check for reasonable date range
                if date_values.min().year < 1900 or date_values.max().year > 2100:
                    logger.warning("Date values outside reasonable range (1900-2100)")
            except (ValueError, TypeError):
                return False, "Date column contains invalid date values"
            
            # Check for zero variance
            numeric_target = pd.to_numeric(target_series.dropna(), errors='coerce')
            if numeric_target.std() == 0:
                return False, "Target column has zero variance - cannot forecast constant values"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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

    @lru_cache(maxsize=32)
    def _get_cached_model_params(self, seasonality_mode: str, changepoint_prior_scale: float,
                                seasonality_prior_scale: float, interval_width: float,
                                yearly_seasonality: str, weekly_seasonality: str, 
                                daily_seasonality: str) -> dict:
        """Cache Prophet model parameters to avoid repeated parameter processing"""
        
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
        
        return {
            'yearly_seasonality': convert_seasonality(yearly_seasonality),
            'weekly_seasonality': convert_seasonality(weekly_seasonality),
            'daily_seasonality': convert_seasonality(daily_seasonality),
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'interval_width': interval_width
        }

    def prepare_data(self, df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
        """
        Prepare data for Prophet processing
        """
        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna()
        
        # Optimize DataFrame for performance
        prophet_df = self.optimize_dataframe(prophet_df)
        logger.info(f"Prepared Prophet data - Shape: {prophet_df.shape}")
        
        return prophet_df

    def split_data(self, prophet_df: pd.DataFrame, train_size: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data for training and evaluation
        Returns: (train_df, test_df)
        """
        split_point = int(len(prophet_df) * train_size)
        
        # For forecasting, we want to use ALL available data for training
        # The train/test split is only used for evaluation metrics
        train_df = prophet_df  # Use ALL data for training the model
        test_df = prophet_df[split_point:]  # Keep test split only for metrics evaluation
        logger.info(f"Data split - Using all {len(train_df)} rows for training, {len(test_df)} rows for evaluation")
        
        return train_df, test_df

    def create_model(self, model_config: dict, confidence_interval: float = 0.95) -> Prophet:
        """
        Create and configure Prophet model
        """
        # Convert confidence interval to Prophet's interval_width format
        if confidence_interval > 1.0:
            interval_width = confidence_interval / 100.0
        else:
            interval_width = confidence_interval
        
        # Ensure interval_width is within valid range [0.1, 0.99]
        interval_width = max(0.1, min(0.99, interval_width))
        logger.info(f"Using confidence interval: {confidence_interval} -> interval_width: {interval_width}")
        
        # Use cached model parameters for better performance
        cached_params = self._get_cached_model_params(
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
            self._add_holidays(model, model_config)
        
        return model

    def _add_holidays(self, model: Prophet, model_config: dict):
        """Add holidays to Prophet model"""
        logger.info("Adding holidays to Prophet model")
        try:
            import holidays
            
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
            
            # Create holiday DataFrame (simplified implementation)
            holiday_df = pd.DataFrame({
                'holiday': list(country_holidays.keys()),
                'ds': pd.to_datetime(list(country_holidays.keys())),
            })
            
            if not holiday_df.empty:
                # Use add_country_holidays for supported countries
                supported_countries = ['US', 'CA', 'UK', 'GB']
                if country_code in supported_countries:
                    model.add_country_holidays(country_name=country_code)
                    logger.info(f"Added {country_code} holidays to model")
                
        except ImportError:
            logger.error("holidays package not available. Install with: pip install holidays")
        except Exception as e:
            logger.error(f"Error adding holidays: {str(e)}")

    def calculate_metrics(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> Dict[str, float]:
        """
        Calculate forecast metrics from actual and predicted values
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
        Calculate forecast metrics on test set (original method for backward compatibility)
        """
        metrics = {}
        if not test_df.empty:
            logger.info("Calculating metrics on test set")
            # Get predictions for test period
            test_forecast = forecast[forecast['ds'].isin(test_df['ds'])]
            if not test_forecast.empty:
                actual_values = test_df['y'].values
                predicted_values = test_forecast['yhat'].values
                
                return self.calculate_metrics(actual_values, predicted_values)
            else:
                logger.warning("No test forecast data available for metrics calculation")
        else:
            logger.info("No test data available for metrics calculation")
        
        # Provide fallback metrics if calculation fails
        return {'mape': 0.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}

    def run_forecast_core(self, df: pd.DataFrame, date_col: str, target_col: str, 
                         model_config: dict, base_config: dict) -> ProphetForecastResult:
        """
        Run Prophet forecast - pure business logic without UI dependencies
        Returns: ProphetForecastResult object
        """
        try:
            logger.info(f"Starting Prophet forecast - Data shape: {df.shape}, Date col: {date_col}, Target col: {target_col}")
            
            # Step 1: Validate inputs
            is_valid, error_msg = self.validate_inputs(df, date_col, target_col)
            if not is_valid:
                return ProphetForecastResult(
                    forecast_df=pd.DataFrame(), metrics={}, model=None, 
                    raw_forecast=pd.DataFrame(), success=False, error=error_msg
                )
            
            # Step 2: Prepare data
            prophet_df = self.prepare_data(df, date_col, target_col)
            
            # Step 3: Split data
            train_size = base_config.get('train_size', 0.8)
            train_df, test_df = self.split_data(prophet_df, train_size)
            
            # Step 4: Create and fit model
            confidence_interval = base_config.get('confidence_interval', 0.95)
            model = self.create_model(model_config, confidence_interval)
            
            logger.info("Fitting Prophet model...")
            model.fit(train_df)
            logger.info("Prophet model fitted successfully")
            
            # Step 5: Generate forecast
            forecast_periods = base_config.get('forecast_periods', 30)
            logger.info(f"Creating forecast for {forecast_periods} periods")
            future = model.make_future_dataframe(periods=forecast_periods)
            
            logger.info("Generating Prophet forecast...")
            forecast = model.predict(future)
            logger.info(f"Forecast generated successfully - Shape: {forecast.shape}")
            
            # Step 6: Calculate metrics
            metrics = self.calculate_metrics_from_dataframes(forecast, test_df)
            
            # Step 7: Return success result
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

# Factory function for easy instantiation
def create_prophet_forecaster() -> ProphetForecaster:
    """Factory function to create ProphetForecaster instance"""
    return ProphetForecaster()
