"""
Enhanced SARIMA (Seasonal ARIMA) module with advanced features.
Provides comprehensive seasonal time series forecasting with auto-tuning and diagnostics.
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
    import pmdarima as pm
    from pmdarima import auto_arima
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    import joblib
    import io
except ImportError as e:
    st.error(f"Missing required packages for SARIMA: {e}")
    st.stop()

from .config import (
    MODEL_LABELS, SARIMA_DEFAULTS, FORECAST_DEFAULTS,
    VISUALIZATION_CONFIG, ERROR_MESSAGES
)


class SARIMAEnhanced:
    """Enhanced SARIMA with auto-tuning and comprehensive diagnostics."""
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.forecast_result = None
        self.training_data = None
        self.validation_data = None
        self.model_params = {}
        self.diagnostics = {}
        self.seasonal_period = None
        
    def detect_seasonal_period(self, data: pd.Series, max_period: int = 365) -> int:
        """Detect seasonal period using multiple methods."""
        try:
            # Method 1: Auto-correlation analysis
            autocorr_periods = []
            
            # Calculate autocorrelations for different lags
            possible_periods = [7, 12, 24, 30, 52, 365]  # Common periods
            possible_periods = [p for p in possible_periods if p <= len(data) // 3 and p <= max_period]
            
            if not possible_periods:
                return min(12, len(data) // 4)
            
            # Method 2: Using pmdarima's estimate_seasonal_period
            try:
                estimated_period = pm.arima.utils.ndiffs(data, alpha=0.05, test='adf', max_d=2)
                if 1 <= estimated_period <= max_period:
                    autocorr_periods.append(estimated_period)
            except Exception:
                pass
            
            # Method 3: Simple periodogram analysis
            try:
                from scipy.fft import fft, fftfreq
                
                # Remove trend first
                detrended = data - data.rolling(window=min(12, len(data)//4)).mean()
                detrended = detrended.dropna()
                
                if len(detrended) > 10:
                    # FFT analysis
                    fft_vals = np.abs(fft(detrended.values))
                    freqs = fftfreq(len(detrended))
                    
                    # Find dominant frequencies
                    dominant_freq_idx = np.argsort(fft_vals[1:len(fft_vals)//2])[-3:]
                    
                    for idx in dominant_freq_idx:
                        if freqs[idx] > 0:
                            period = int(1 / freqs[idx])
                            if 2 <= period <= max_period:
                                autocorr_periods.append(period)
            except Exception:
                pass
            
            # Choose the most reasonable period
            if autocorr_periods:
                # Prefer common business periods
                business_periods = [p for p in autocorr_periods if p in [7, 12, 24, 30, 52, 365]]
                if business_periods:
                    return min(business_periods)
                else:
                    return min(autocorr_periods)
            else:
                # Default based on data frequency
                if len(data) >= 365:
                    return 52  # Weekly pattern in daily data
                elif len(data) >= 52:
                    return 12  # Monthly pattern
                else:
                    return min(7, len(data) // 4)
                    
        except Exception:
            return min(12, len(data) // 4)
    
    def check_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """Check stationarity using multiple tests."""
        results = {}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(data.dropna())
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < 0.05
            }
            
            # KPSS test
            try:
                kpss_result = kpss(data.dropna())
                results['kpss'] = {
                    'statistic': kpss_result[0],
                    'p_value': kpss_result[1],
                    'critical_values': kpss_result[3],
                    'is_stationary': kpss_result[1] > 0.05
                }
            except Exception:
                results['kpss'] = {'error': 'Could not perform KPSS test'}
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def prepare_data(self, data: pd.DataFrame, date_col: str, target_col: str,
                    train_size: float = 0.8) -> Tuple[pd.Series, pd.Series]:
        """Prepare data for SARIMA modeling."""
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
            
            # Detect seasonal period
            self.seasonal_period = self.detect_seasonal_period(ts)
            
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
    
    def auto_tune_sarima(self, data: pd.Series, seasonal_period: int) -> Tuple[int, int, int, int, int, int, int]:
        """Auto-tune SARIMA parameters using pmdarima."""
        try:
            # Use auto_arima for parameter selection
            model = auto_arima(
                data,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                seasonal=True,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                m=seasonal_period,
                max_d=2, max_D=1,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
            
            order = model.order
            seasonal_order = model.seasonal_order
            
            return (order[0], order[1], order[2], 
                   seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_order[3])
            
        except Exception as e:
            st.warning(f"Auto-tuning failed: {str(e)}. Using default parameters.")
            # Return default parameters
            return (1, 1, 1, 1, 1, 1, seasonal_period)
    
    def fit_model(self, data: pd.Series, config: Dict[str, Any]) -> bool:
        """Fit SARIMA model with given configuration."""
        try:
            # Extract parameters
            auto_tune = config.get('auto_tune', True)
            seasonal_period = config.get('seasonal_periods', self.seasonal_period)
            
            if auto_tune:
                # Auto-tune parameters
                p, d, q, P, D, Q, s = self.auto_tune_sarima(data, seasonal_period)
            else:
                # Use manual parameters
                p = config.get('p', 1)
                d = config.get('d', 1)
                q = config.get('q', 1)
                P = config.get('P', 1)
                D = config.get('D', 1)
                Q = config.get('Q', 1)
                s = seasonal_period
            
            # Store model parameters
            self.model_params = {
                'order': (p, d, q),
                'seasonal_order': (P, D, Q, s),
                'auto_tuned': auto_tune
            }
            
            # Create and fit model
            self.model = SARIMAX(
                data,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = self.model.fit(disp=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error fitting SARIMA model: {str(e)}")
            return False
    
    def generate_forecast(self, periods: int, confidence_interval: float = 0.95) -> pd.DataFrame:
        """Generate forecast with confidence intervals."""
        try:
            if self.fitted_model is None:
                raise ValueError("Model must be fitted before generating forecasts")
            
            # Generate forecast
            forecast_result = self.fitted_model.get_forecast(steps=periods)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=1-confidence_interval)
            
            # Create forecast DataFrame
            forecast_dates = pd.date_range(
                start=self.training_data.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq=self.training_data.index.inferred_freq or 'D'
            )
            
            forecast_df = pd.DataFrame({
                'ds': forecast_dates,
                'yhat': forecast.values,
                'yhat_lower': conf_int.iloc[:, 0].values,
                'yhat_upper': conf_int.iloc[:, 1].values
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
            if self.fitted_model is None:
                return metrics
            
            # Model information criteria
            metrics['aic'] = self.fitted_model.aic
            metrics['bic'] = self.fitted_model.bic
            if hasattr(self.fitted_model, 'aicc'):
                metrics['aicc'] = self.fitted_model.aicc
            
            # In-sample metrics
            fitted_values = self.fitted_model.fittedvalues
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
            
            # Validation metrics if validation data exists
            if self.validation_data is not None and len(self.validation_data) > 0:
                val_forecast_result = self.fitted_model.get_forecast(steps=len(self.validation_data))
                val_forecast = val_forecast_result.predicted_mean
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
                lb_result = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
                diagnostics['autocorrelation_test'] = {
                    'ljung_box_stat': lb_result['lb_stat'].iloc[-1],
                    'ljung_box_p_value': lb_result['lb_pvalue'].iloc[-1],
                    'has_autocorrelation': lb_result['lb_pvalue'].iloc[-1] < 0.05
                }
            except Exception:
                diagnostics['autocorrelation_test'] = {'error': 'Could not perform autocorrelation test'}
            
            # Stationarity check
            diagnostics['stationarity'] = self.check_stationarity(self.training_data)
            
            # Model parameters
            diagnostics['model_parameters'] = self.model_params.copy()
            if hasattr(self.fitted_model, 'params'):
                diagnostics['fitted_parameters'] = dict(self.fitted_model.params)
            
            # Model summary statistics
            if hasattr(self.fitted_model, 'summary'):
                try:
                    summary = self.fitted_model.summary()
                    diagnostics['model_summary'] = str(summary)
                except Exception:
                    pass
            
            self.diagnostics = diagnostics
            
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
            
            # 3. ACF/PACF plots
            if self.fitted_model is not None:
                fig_acf_pacf = self.create_acf_pacf_plot()
                plots['acf_pacf'] = fig_acf_pacf
            
            # 4. Components plot
            if self.fitted_model is not None:
                fig_components = self.create_components_plot()
                plots['components'] = fig_components
            
            # 5. Diagnostics plot
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
                title='SARIMA Forecast',
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
                          'Q-Q Plot', 'Residuals ACF']
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
            
            # Simple ACF for residuals (approximation)
            # Calculate autocorrelation manually for simplicity
            max_lags = min(20, len(residuals) // 4)
            lags = range(max_lags)
            acf_values = [residuals.autocorr(lag=lag) for lag in lags]
            
            fig.add_trace(go.Bar(
                x=list(lags),
                y=acf_values,
                name='ACF'
            ), row=2, col=2)
            
            fig.update_layout(
                title='Residuals Analysis',
                showlegend=False
            )
            
        except Exception as e:
            st.warning(f"Error creating residuals plot: {str(e)}")
        
        return fig
    
    def create_acf_pacf_plot(self) -> go.Figure:
        """Create ACF and PACF plots."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Autocorrelation Function (ACF)', 'Partial Autocorrelation Function (PACF)']
        )
        
        try:
            # Calculate ACF and PACF manually for original data
            data = self.training_data.dropna()
            max_lags = min(40, len(data) // 4)
            
            # ACF
            lags = range(max_lags)
            acf_values = [data.autocorr(lag=lag) for lag in lags]
            
            fig.add_trace(go.Bar(
                x=list(lags),
                y=acf_values,
                name='ACF'
            ), row=1, col=1)
            
            # PACF (simplified calculation)
            # This is a basic approximation - for full PACF use statsmodels
            pacf_values = []
            for lag in lags:
                if lag == 0:
                    pacf_values.append(1.0)
                else:
                    # Simple approximation
                    corr = data.autocorr(lag=lag)
                    pacf_values.append(corr)
            
            fig.add_trace(go.Bar(
                x=list(lags),
                y=pacf_values,
                name='PACF (Approx)'
            ), row=1, col=2)
            
            # Add confidence bands (approximate)
            conf_level = 1.96 / np.sqrt(len(data))
            fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=1, col=2)
            
            fig.update_layout(
                title='ACF and PACF Analysis',
                showlegend=False
            )
            
        except Exception as e:
            st.warning(f"Error creating ACF/PACF plot: {str(e)}")
        
        return fig
    
    def create_components_plot(self) -> go.Figure:
        """Create seasonal decomposition plot."""
        fig = go.Figure()
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                self.training_data, 
                model='additive', 
                period=self.seasonal_period
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
                title='Seasonal Decomposition',
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
            
            # Create a metrics table visualization
            metrics = self.calculate_metrics()
            
            metric_names = list(metrics.keys())
            metric_values = [f"{v:.4f}" if isinstance(v, (int, float)) and not np.isnan(v) else str(v) 
                           for v in metrics.values()]
            
            fig.add_trace(go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[metric_names, metric_values])
            ))
            
            fig.update_layout(
                title='SARIMA Model Diagnostics and Metrics'
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


def run_sarima_forecast(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, go.Figure]]:
    """
    Main function to run SARIMA forecasting with enhanced features.
    
    Args:
        data: Input DataFrame with date and target columns
        config: Configuration dictionary with model parameters
        
    Returns:
        Tuple of (forecast_df, metrics, plots)
    """
    try:
        # Initialize model
        model = SARIMAEnhanced()
        
        # Prepare data
        train_data, val_data = model.prepare_data(
            data, 
            config['date_column'], 
            config['target_column'],
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
        
        return forecast_df, metrics, plots
        
    except Exception as e:
        st.error(f"Error in SARIMA forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}
