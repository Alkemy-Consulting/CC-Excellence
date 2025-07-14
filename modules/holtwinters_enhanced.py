"""
Enhanced Holt-Winters Exponential Smoothing module with advanced features.
Provides comprehensive forecasting capabilities with diagnostics and visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional, Tuple, List, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    import joblib
    import io
except ImportError as e:
    st.error(f"Missing required packages for Holt-Winters: {e}")
    st.stop()

from src.modules.utils.config import (
    MODEL_LABELS, HOLTWINTERS_DEFAULTS, FORECAST_DEFAULTS,
    VISUALIZATION_CONFIG, ERROR_MESSAGES
)


class HoltWintersEnhanced:
    """Enhanced Holt-Winters Exponential Smoothing with advanced features."""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.forecast_result = None
        self.training_data = None
        self.validation_data = None
        self.model_params = {}
        self.diagnostics = {}
        
    def detect_seasonal_period(self, data: pd.Series, max_period: int = 365) -> int:
        """Detect seasonal period automatically."""
        try:
            # Try different periods and find the best one
            periods_to_test = [7, 12, 24, 52, 365]  # Common periods
            periods_to_test = [p for p in periods_to_test if p <= len(data) // 2 and p <= max_period]
            
            if not periods_to_test:
                return min(12, len(data) // 4)
            
            best_period = periods_to_test[0]
            best_score = float('inf')
            
            for period in periods_to_test:
                try:
                    # Test with simple exponential smoothing
                    temp_model = ExponentialSmoothing(
                        data, 
                        trend='add', 
                        seasonal='add', 
                        seasonal_periods=period
                    ).fit(optimized=True)
                    
                    # Use AIC as scoring metric
                    if hasattr(temp_model, 'aic') and temp_model.aic < best_score:
                        best_score = temp_model.aic
                        best_period = period
                        
                except Exception:
                    continue
                    
            return best_period
            
        except Exception:
            return min(12, len(data) // 4)
    
    def prepare_data(self, data: pd.DataFrame, date_col: str, target_col: str,
                    train_size: float = 0.8) -> Tuple[pd.Series, pd.Series]:
        """Prepare data for Holt-Winters modeling."""
        try:
            # Ensure data is sorted by date
            df = data.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # Create time series
            ts = df.set_index(date_col)[target_col]
            ts = ts.asfreq(ts.index.inferred_freq or 'D')
            
            # Handle missing values
            if ts.isnull().any():
                ts = ts.interpolate(method='linear')
            
            # Split data
            split_point = int(len(ts) * train_size)
            train_data = ts[:split_point]
            validation_data = ts[split_point:] if split_point < len(ts) else None
            
            self.training_data = train_data
            self.validation_data = validation_data
            
            return train_data, validation_data
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None
    
    def fit_model(self, data: pd.Series, config: Dict[str, Any]) -> bool:
        """Fit Holt-Winters model with given configuration."""
        try:
            # Extract parameters
            trend = config.get('trend', 'add')
            seasonal = config.get('seasonal', 'add')
            seasonal_periods = config.get('seasonal_periods', None)
            damped_trend = config.get('damped_trend', False)
            use_boxcox = config.get('use_boxcox', False)
            remove_bias = config.get('remove_bias', False)
            
            # Auto-detect seasonal period if not provided
            if seasonal_periods is None or seasonal_periods == 'auto':
                seasonal_periods = self.detect_seasonal_period(data)
            
            # Store model parameters
            self.model_params = {
                'trend': trend,
                'seasonal': seasonal,
                'seasonal_periods': seasonal_periods,
                'damped_trend': damped_trend,
                'use_boxcox': use_boxcox,
                'remove_bias': remove_bias
            }
            
            # Create and fit model
            self.model = ExponentialSmoothing(
                data,
                trend=trend if trend != 'none' else None,
                seasonal=seasonal if seasonal != 'none' else None,
                seasonal_periods=seasonal_periods if seasonal != 'none' else None,
                damped_trend=damped_trend,
                use_boxcox=use_boxcox
            )
            
            self.fitted_model = self.model.fit(
                optimized=True,
                remove_bias=remove_bias,
                use_brute=True
            )
            
            return True
            
        except Exception as e:
            st.error(f"Error fitting Holt-Winters model: {str(e)}")
            return False
    
    def generate_forecast(self, periods: int, confidence_interval: float = 0.95) -> pd.DataFrame:
        """Generate forecast with confidence intervals."""
        try:
            if self.fitted_model is None:
                raise ValueError("Model must be fitted before generating forecasts")
            
            # Generate forecast
            forecast = self.fitted_model.forecast(periods)
            
            # Get prediction intervals - use more compatible approach
            try:
                # Try new method first (newer statsmodels)
                if hasattr(self.fitted_model, 'get_prediction'):
                    pred_int = self.fitted_model.get_prediction(
                        start=len(self.training_data),
                        end=len(self.training_data) + periods - 1
                    ).summary_frame(alpha=1-confidence_interval)
                    lower_bound = pred_int['mean_ci_lower'].values
                    upper_bound = pred_int['mean_ci_upper'].values
                else:
                    # Fallback: use forecast method with prediction intervals
                    forecast_result = self.fitted_model.forecast(periods, return_conf_int=True)
                    if isinstance(forecast_result, tuple) and len(forecast_result) == 2:
                        forecast, conf_int = forecast_result
                        lower_bound = conf_int[:, 0]
                        upper_bound = conf_int[:, 1]
                    else:
                        # If no confidence intervals available, use simple approximation
                        forecast_std = np.std(self.fitted_model.resid) if hasattr(self.fitted_model, 'resid') else np.std(forecast) * 0.1
                        margin = 1.96 * forecast_std  # 95% confidence interval approximation
                        lower_bound = forecast - margin
                        upper_bound = forecast + margin
            except Exception:
                # Final fallback: simple approximation
                forecast_std = np.std(forecast) * 0.1
                margin = 1.96 * forecast_std
                lower_bound = forecast - margin
                upper_bound = forecast + margin
            
            # Ensure arrays are 1D
            if hasattr(forecast, 'values'):
                forecast = forecast.values
            if hasattr(lower_bound, 'values'):
                lower_bound = lower_bound.values
            if hasattr(upper_bound, 'values'):
                upper_bound = upper_bound.values
            
            # Create forecast DataFrame
            forecast_dates = pd.date_range(
                start=self.training_data.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=self.training_data.index.inferred_freq or 'D'
            )
            
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast,
                'yhat_lower': lower_bound,
                'yhat_upper': upper_bound
            })
            
            self.forecast_result = forecast_df
            return forecast_df
            
        except Exception as e:
            st.error(f"Error generating forecast: {str(e)}")
            return pd.DataFrame()
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate model performance metrics."""
        metrics = {}
        
        try:
            if self.fitted_model is None or self.validation_data is None:
                return metrics
            
            # Generate in-sample predictions
            fitted_values = self.fitted_model.fittedvalues
            
            # Calculate in-sample metrics
            if len(fitted_values) > 0:
                train_actual = self.training_data.values
                train_pred = fitted_values.values
                
                # Align lengths
                min_len = min(len(train_actual), len(train_pred))
                train_actual = train_actual[-min_len:]
                train_pred = train_pred[-min_len:]
                
                metrics['train_mae'] = mean_absolute_error(train_actual, train_pred)
                metrics['train_mse'] = mean_squared_error(train_actual, train_pred)
                metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
                
                # MAPE calculation with zero handling
                non_zero_mask = train_actual != 0
                if np.any(non_zero_mask):
                    metrics['train_mape'] = mean_absolute_percentage_error(
                        train_actual[non_zero_mask], 
                        train_pred[non_zero_mask]
                    )
                else:
                    metrics['train_mape'] = np.nan
            
            # Calculate validation metrics if validation data exists
            if self.validation_data is not None and len(self.validation_data) > 0:
                val_forecast = self.fitted_model.forecast(len(self.validation_data))
                val_actual = self.validation_data.values
                val_pred = val_forecast.values
                
                metrics['val_mae'] = mean_absolute_error(val_actual, val_pred)
                metrics['val_mse'] = mean_squared_error(val_actual, val_pred)
                metrics['val_rmse'] = np.sqrt(metrics['val_mse'])
                
                # MAPE calculation with zero handling
                non_zero_mask = val_actual != 0
                if np.any(non_zero_mask):
                    metrics['val_mape'] = mean_absolute_percentage_error(
                        val_actual[non_zero_mask], 
                        val_pred[non_zero_mask]
                    )
                else:
                    metrics['val_mape'] = np.nan
            
            # Model information criteria
            if hasattr(self.fitted_model, 'aic'):
                metrics['aic'] = self.fitted_model.aic
            if hasattr(self.fitted_model, 'bic'):
                metrics['bic'] = self.fitted_model.bic
            if hasattr(self.fitted_model, 'aicc'):
                metrics['aicc'] = self.fitted_model.aicc
                
        except Exception as e:
            st.warning(f"Error calculating some metrics: {str(e)}")
        
        return metrics
    
    def perform_diagnostics(self) -> Dict[str, Any]:
        """Perform comprehensive model diagnostics."""
        diagnostics = {}
        
        try:
            if self.fitted_model is None:
                return diagnostics
            
            # Get residuals
            residuals = self.fitted_model.resid
            
            # Basic residual statistics
            diagnostics['residual_stats'] = {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            }
            
            # Normality tests
            try:
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                diagnostics['normality_test'] = {
                    'shapiro_stat': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except Exception:
                diagnostics['normality_test'] = {'error': 'Could not perform normality test'}
            
            # Ljung-Box test for autocorrelation
            try:
                lb_stat, lb_p = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=False)
                diagnostics['autocorrelation_test'] = {
                    'ljung_box_stat': lb_stat.iloc[-1] if hasattr(lb_stat, 'iloc') else lb_stat,
                    'ljung_box_p_value': lb_p.iloc[-1] if hasattr(lb_p, 'iloc') else lb_p,
                    'has_autocorrelation': (lb_p.iloc[-1] if hasattr(lb_p, 'iloc') else lb_p) < 0.05
                }
            except Exception:
                diagnostics['autocorrelation_test'] = {'error': 'Could not perform autocorrelation test'}
            
            # Model parameters
            diagnostics['model_parameters'] = self.model_params.copy()
            if hasattr(self.fitted_model, 'params'):
                diagnostics['fitted_parameters'] = dict(self.fitted_model.params)
            
            self.diagnostics = diagnostics;
            
        except Exception as e:
            st.warning(f"Error in diagnostics: {str(e)}")
        
        return diagnostics
    
    def create_visualizations(self) -> Dict[str, go.Figure]:
        """Create comprehensive visualizations."""
        plots = {}
        
        try:
            # 1. Forecast plot
            if self.forecast_result is not None and not self.forecast_result.empty:
                fig_forecast = self.create_forecast_plot()
                plots['forecast'] = fig_forecast
            
            # 2. Residuals analysis
            if self.fitted_model is not None:
                fig_residuals = self.create_residuals_plot()
                plots['residuals'] = fig_residuals
            
            # 3. Components plot
            if self.fitted_model is not None:
                fig_components = self.create_components_plot()
                plots['components'] = fig_components
            
            # 4. Diagnostics plot
            if self.fitted_model is not None:
                fig_diagnostics = self.create_diagnostics_plot()
                plots['diagnostics'] = fig_diagnostics
                
        except Exception as e:
            st.warning(f"Error creating visualizations: {str(e)}")
        
        return plots
    
    def create_forecast_plot(self) -> go.Figure:
        """Create forecast visualization."""
        fig = go.Figure()
        
        try:
            # Historical data
            fig.add_trace(go.Scatter(
                x=self.training_data.index,
                y=self.training_data.values,
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Validation data if available
            if self.validation_data is not None:
                fig.add_trace(go.Scatter(
                    x=self.validation_data.index,
                    y=self.validation_data.values,
                    mode='lines',
                    name='Actual (Validation)',
                    line=dict(color='green')
                ))
            
            # Forecast
            if self.forecast_result is not None:
                fig.add_trace(go.Scatter(
                    x=self.forecast_result['ds'],
                    y=self.forecast_result['yhat'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                # Confidence intervals
                fig.add_trace(go.Scatter(
                    x=self.forecast_result['ds'],
                    y=self.forecast_result['yhat_upper'],
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=self.forecast_result['ds'],
                    y=self.forecast_result['yhat_lower'],
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    name='Confidence Interval'
                ))
            
            fig.update_layout(
                title='Holt-Winters Forecast',
                xaxis_title='Date',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
        except Exception as e:
            st.warning(f"Error creating forecast plot: {str(e)}")
        
        return fig
    
    def create_residuals_plot(self) -> go.Figure:
        """Create residuals analysis plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Residuals vs Time', 'Residuals Distribution', 
                          'Q-Q Plot', 'Autocorrelation']
        )
        
        try:
            residuals = self.fitted_model.resid
            
            # Residuals vs time
            fig.add_trace(go.Scatter(
                x=residuals.index,
                y=residuals.values,
                mode='lines+markers',
                name='Residuals'
            ), row=1, col=1)
            
            # Residuals distribution
            fig.add_trace(go.Histogram(
                x=residuals.values,
                nbinsx=30,
                name='Distribution'
            ), row=1, col=2)
            
            # Q-Q plot
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals.values)
            
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot'
            ), row=2, col=1)
            
            # Add reference line for Q-Q plot
            fig.add_trace(go.Scatter(
                x=theoretical_quantiles,
                y=theoretical_quantiles,
                mode='lines',
                name='Reference Line',
                line=dict(dash='dash')
            ), row=2, col=1)
            
            fig.update_layout(
                title='Residuals Analysis',
                showlegend=False
            )
            
        except Exception as e:
            st.warning(f"Error creating residuals plot: {str(e)}")
        
        return fig
    
    def create_components_plot(self) -> go.Figure:
        """Create components decomposition plot."""
        fig = go.Figure()
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                self.training_data, 
                model='additive', 
                period=self.model_params.get('seasonal_periods', 12)
            )
            
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                vertical_spacing=0.05
            )
            
            # Original
            fig.add_trace(go.Scatter(
                x=self.training_data.index,
                y=self.training_data.values,
                mode='lines',
                name='Original'
            ), row=1, col=1)
            
            # Trend
            fig.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend.values,
                mode='lines',
                name='Trend'
            ), row=2, col=1)
            
            # Seasonal
            fig.add_trace(go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal.values,
                mode='lines',
                name='Seasonal'
            ), row=3, col=1)
            
            # Residual
            fig.add_trace(go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid.values,
                mode='lines',
                name='Residual'
            ), row=4, col=1)
            
            fig.update_layout(
                title='Time Series Decomposition',
                showlegend=False,
                height=800
            )
            
        except Exception as e:
            st.warning(f"Error creating components plot: {str(e)}")
        
        return fig
    
    def create_diagnostics_plot(self) -> go.Figure:
        """Create model diagnostics visualization."""
        fig = go.Figure()
        
        try:
            if not self.diagnostics:
                self.perform_diagnostics()
            
            # Create a simple metrics table visualization
            metrics = self.calculate_metrics()
            
            metric_names = list(metrics.keys())
            metric_values = [f"{v:.4f}" if isinstance(v, (int, float)) and not np.isnan(v) else str(v) 
                           for v in metrics.values()]
            
            fig.add_trace(go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[metric_names, metric_values])
            ))
            
            fig.update_layout(
                title='Model Diagnostics and Metrics'
            )
            
        except Exception as e:
            st.warning(f"Error creating diagnostics plot: {str(e)}")
        
        return fig
    
    def export_results(self, format_type: str = 'csv') -> bytes:
        """Export forecast results."""
        try:
            if self.forecast_result is None or self.forecast_result.empty:
                raise ValueError("No forecast results to export")
            
            if format_type.lower() == 'csv':
                output = io.StringIO()
                self.forecast_result.to_csv(output, index=False)
                return output.getvalue().encode()
            
            elif format_type.lower() == 'excel':
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    self.forecast_result.to_excel(writer, sheet_name='Forecast', index=False)
                    
                    # Add metrics if available
                    metrics = self.calculate_metrics()
                    if metrics:
                        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                    
                    # Add diagnostics if available
                    if self.diagnostics:
                        diag_data = []
                        for key, value in self.diagnostics.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    diag_data.append([f"{key}_{subkey}", str(subvalue)])
                            else:
                                diag_data.append([key, str(value)])
                        
                        if diag_data:
                            diag_df = pd.DataFrame(diag_data, columns=['Diagnostic', 'Value'])
                            diag_df.to_excel(writer, sheet_name='Diagnostics', index=False)
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
                
        except Exception as e:
            st.error(f"Error exporting results: {str(e)}")
            return b""


def run_holtwinters_forecast(df: pd.DataFrame, date_col: str, target_col: str,
                           model_config: Dict[str, Any], base_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Enhanced Holt-Winters forecast with proper metrics calculation
    """
    try:
        print(f"DEBUG Holt-Winters: Starting with config: {model_config}")
        print(f"DEBUG Holt-Winters: Data shape: {df.shape}")
        
        # Ensure we have the required configuration
        config = {**base_config, **model_config}
        
        # Validate seasonal periods
        seasonal_periods = config.get('seasonal_periods', 12)
        if len(df) < seasonal_periods * 2:
            raise ValueError(f"Insufficient data for Holt-Winters: need at least {seasonal_periods * 2} points, got {len(df)}")
        
        # Initialize model
        model = HoltWintersEnhanced()
        
        # Prepare data
        train_data, val_data = model.prepare_data(
            df, 
            date_col, 
            target_col,
            config.get('train_size', 0.8)
        )
        
        if train_data is None:
            return pd.DataFrame(), {}, {}
        
        # Fit model
        if not model.fit_model(train_data, config):
            return pd.DataFrame(), {}, {}
        
        # Generate forecast
        forecast_df = model.generate_forecast(
            config.get('forecast_periods', 30),
            config.get('confidence_interval', 0.95)
        )
        
        # Calculate metrics
        metrics = model.calculate_metrics()
        
        # Perform diagnostics
        model.perform_diagnostics()
        
        # Create visualizations
        plots = model.create_visualizations()
        
        # CRITICAL: Ensure metrics calculation includes MAPE
        if not forecast_df.empty and len(val_data) > 0:
            actual_values = val_data.values
            fitted_values = model.fitted_model.fittedvalues[-len(val_data):]
            
            # Align lengths for metric calculation
            min_len = min(len(actual_values), len(fitted_values))
            actual_aligned = actual_values[-min_len:]
            fitted_aligned = fitted_values[-min_len:]
            
            # Calculate metrics with proper error handling
            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                # MAPE calculation with zero-division protection
                def calculate_mape(actual, predicted):
                    actual, predicted = np.array(actual), np.array(predicted)
                    mask = actual != 0  # Avoid division by zero
                    if mask.sum() == 0:
                        return 100.0  # Return 100% error if all actual values are zero
                    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
                
                metrics = {
                    'mape': float(calculate_mape(actual_aligned, fitted_aligned)),
                    'mae': float(mean_absolute_error(actual_aligned, fitted_aligned)),
                    'rmse': float(np.sqrt(mean_squared_error(actual_aligned, fitted_aligned))),
                    'r2': float(r2_score(actual_aligned, fitted_aligned))
                }
                
                print(f"DEBUG Holt-Winters: Calculated metrics: {metrics}")
                
                # Validate MAPE
                if not np.isfinite(metrics['mape']) or metrics['mape'] < 0:
                    metrics['mape'] = 100.0  # Fallback value
                    
            except Exception as metric_error:
                print(f"DEBUG Holt-Winters: Metrics calculation error: {metric_error}")
                # Provide fallback metrics
                metrics = {
                    'mape': 100.0,  # High error as fallback
                    'mae': 0.0,
                    'rmse': 0.0,
                    'r2': 0.0
                }
        else:
            print(f"DEBUG Holt-Winters: Cannot calculate metrics - forecast_df empty: {forecast_df.empty}")
            metrics = {
                'mape': 100.0,  # High error as fallback
                'mae': 0.0,
                'rmse': 0.0,
                'r2': 0.0
            }
        
        print(f"DEBUG Holt-Winters: Final metrics returned: {metrics}")
        return forecast_df, metrics, plots
        
    except Exception as e:
        print(f"DEBUG Holt-Winters: Exception occurred: {str(e)}")
        import traceback
        print(f"DEBUG Holt-Winters: Traceback: {traceback.format_exc()}")
        
        # Return empty results with fallback metrics
        return pd.DataFrame(), {
            'mape': 100.0,
            'mae': 0.0,
            'rmse': 0.0,
            'r2': 0.0
        }, {}
