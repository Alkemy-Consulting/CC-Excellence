"""
Extended Diagnostic Plots Module for Prophet Forecasting
Enterprise-level diagnostic visualization and analysis tools
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .prophet_core import ProphetForecastResult

logger = logging.getLogger(__name__)

class ProphetDiagnosticConfig:
    """Configuration for diagnostic plots"""
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'grid': '#e6e6e6',
            'background': '#fafafa'
        }
        self.plot_height = 400
        self.subplot_spacing = 0.08
        self.font_size = 12
        self.line_width = 2

class ProphetDiagnosticAnalyzer:
    """Advanced diagnostic analysis for Prophet forecasts"""
    
    def __init__(self, config: Optional[ProphetDiagnosticConfig] = None):
        self.config = config or ProphetDiagnosticConfig()
    
    def analyze_forecast_quality(self, forecast_result: ProphetForecastResult, 
                                actual_data: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
        """
        Comprehensive forecast quality analysis
        """
        try:
            analysis = {
                'forecast_coverage': self._analyze_forecast_coverage(forecast_result, actual_data, date_col),
                'residual_analysis': self._analyze_residuals(forecast_result, actual_data, date_col, target_col),
                'trend_analysis': self._analyze_trend_quality(forecast_result),
                'seasonality_analysis': self._analyze_seasonality_quality(forecast_result),
                'uncertainty_analysis': self._analyze_uncertainty_quality(forecast_result),
                'changepoint_analysis': self._analyze_changepoints(forecast_result, actual_data, date_col, target_col)
            }
            
            # Overall quality score
            analysis['quality_score'] = self._calculate_quality_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in forecast quality analysis: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_forecast_coverage(self, forecast_result: ProphetForecastResult, 
                                  actual_data: pd.DataFrame, date_col: str) -> Dict[str, Any]:
        """Analyze how well forecast covers the actual data period"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            return {'coverage_ratio': 0, 'gaps': []}
        
        forecast_dates = pd.to_datetime(forecast_result.raw_forecast['ds'])
        actual_dates = pd.to_datetime(actual_data[date_col])
        
        # Check coverage
        forecast_start = forecast_dates.min()
        forecast_end = forecast_dates.max()
        actual_start = actual_dates.min()
        actual_end = actual_dates.max()
        
        overlap_start = max(forecast_start, actual_start)
        overlap_end = min(forecast_end, actual_end)
        
        if overlap_start <= overlap_end:
            overlap_days = (overlap_end - overlap_start).days
            actual_days = (actual_end - actual_start).days
            coverage_ratio = overlap_days / actual_days if actual_days > 0 else 0
        else:
            coverage_ratio = 0
            overlap_days = 0
        
        return {
            'coverage_ratio': coverage_ratio,
            'forecast_start': forecast_start,
            'forecast_end': forecast_end,
            'actual_start': actual_start,
            'actual_end': actual_end,
            'overlap_days': overlap_days
        }
    
    def _analyze_residuals(self, forecast_result: ProphetForecastResult, 
                          actual_data: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
        """Advanced residual analysis"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            return {'error': 'No forecast data available'}
        
        # Merge forecast with actual data
        forecast_df = forecast_result.raw_forecast[['ds', 'yhat']].copy()
        actual_df = actual_data[[date_col, target_col]].copy()
        actual_df.columns = ['ds', 'y_actual']
        
        merged = pd.merge(forecast_df, actual_df, on='ds', how='inner')
        if merged.empty:
            return {'error': 'No overlapping data for residual analysis'}
        
        # Calculate residuals
        residuals = merged['y_actual'] - merged['yhat']
        
        # Statistical tests
        _, shapiro_p = stats.shapiro(residuals.values[:min(5000, len(residuals))])  # Shapiro-Wilk test
        _, ljungbox_p = self._ljung_box_test(residuals)
        durbin_watson = self._durbin_watson_test(residuals)
        
        return {
            'residuals': residuals,
            'mean_residual': float(residuals.mean()),
            'std_residual': float(residuals.std()),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
            'normality_p_value': float(shapiro_p),
            'autocorrelation_p_value': float(ljungbox_p),
            'durbin_watson_statistic': float(durbin_watson),
            'is_normally_distributed': shapiro_p > 0.05,
            'has_autocorrelation': ljungbox_p < 0.05,
            'dates': merged['ds']
        }
    
    def _ljung_box_test(self, residuals: pd.Series, lags: int = 10) -> Tuple[float, float]:
        """Ljung-Box test for autocorrelation"""
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(residuals, lags=lags, return_df=False)
            return float(result['lb_stat'].iloc[-1]), float(result['lb_pvalue'].iloc[-1])
        except ImportError:
            # Fallback implementation
            n = len(residuals)
            autocorrs = [residuals.autocorr(lag=i) for i in range(1, lags + 1)]
            lb_stat = n * (n + 2) * sum([(autocorrs[i]**2) / (n - i - 1) for i in range(lags)])
            p_value = 1 - stats.chi2.cdf(lb_stat, lags)
            return float(lb_stat), float(p_value)
    
    def _durbin_watson_test(self, residuals: pd.Series) -> float:
        """Durbin-Watson test for serial correlation"""
        diff = residuals.diff().dropna()
        return float(np.sum(diff**2) / np.sum(residuals**2))
    
    def _analyze_trend_quality(self, forecast_result: ProphetForecastResult) -> Dict[str, Any]:
        """Analyze trend component quality"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            return {'error': 'No forecast data available'}
        
        if 'trend' not in forecast_result.raw_forecast.columns:
            return {'error': 'No trend component available'}
        
        trend = forecast_result.raw_forecast['trend']
        
        # Trend characteristics
        trend_slope = np.polyfit(range(len(trend)), trend, 1)[0]
        trend_volatility = trend.std()
        trend_range = trend.max() - trend.min()
        
        # Detect trend changes
        trend_changes = np.diff(trend)
        change_points = np.where(np.abs(trend_changes) > 2 * trend_changes.std())[0]
        
        return {
            'trend_slope': float(trend_slope),
            'trend_volatility': float(trend_volatility),
            'trend_range': float(trend_range),
            'trend_direction': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable',
            'significant_changes': len(change_points),
            'change_points': change_points.tolist()
        }
    
    def _analyze_seasonality_quality(self, forecast_result: ProphetForecastResult) -> Dict[str, Any]:
        """Analyze seasonality components quality"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            return {'error': 'No forecast data available'}
        
        seasonality_analysis = {}
        
        # Analyze each seasonality component
        for component in ['yearly', 'weekly', 'daily']:
            if component in forecast_result.raw_forecast.columns:
                values = forecast_result.raw_forecast[component]
                seasonality_analysis[component] = {
                    'amplitude': float(values.max() - values.min()),
                    'std': float(values.std()),
                    'mean': float(values.mean()),
                    'strength': float(values.std() / (forecast_result.raw_forecast['yhat'].std() + 1e-8))
                }
        
        return seasonality_analysis
    
    def _analyze_uncertainty_quality(self, forecast_result: ProphetForecastResult) -> Dict[str, Any]:
        """Analyze uncertainty intervals quality"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            return {'error': 'No forecast data available'}
        
        forecast = forecast_result.raw_forecast
        if not all(col in forecast.columns for col in ['yhat_lower', 'yhat_upper', 'yhat']):
            return {'error': 'No uncertainty intervals available'}
        
        # Interval characteristics
        interval_width = forecast['yhat_upper'] - forecast['yhat_lower']
        relative_width = interval_width / (forecast['yhat'].abs() + 1e-8)
        
        return {
            'mean_interval_width': float(interval_width.mean()),
            'std_interval_width': float(interval_width.std()),
            'mean_relative_width': float(relative_width.mean()),
            'max_interval_width': float(interval_width.max()),
            'min_interval_width': float(interval_width.min()),
            'interval_symmetry': float(np.corrcoef(
                forecast['yhat'] - forecast['yhat_lower'],
                forecast['yhat_upper'] - forecast['yhat']
            )[0, 1])
        }
    
    def _analyze_changepoints(self, forecast_result: ProphetForecastResult, 
                             actual_data: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
        """Analyze changepoints quality and effectiveness"""
        if not forecast_result.success or forecast_result.model is None:
            return {'error': 'No model available'}
        
        model = forecast_result.model
        if not hasattr(model, 'changepoints') or len(model.changepoints) == 0:
            return {'changepoints_count': 0, 'effectiveness': 'no_changepoints'}
        
        # Changepoint analysis
        changepoints = pd.to_datetime(model.changepoints)
        actual_dates = pd.to_datetime(actual_data[date_col])
        
        # Find changepoints within actual data range
        data_start = actual_dates.min()
        data_end = actual_dates.max()
        valid_changepoints = changepoints[(changepoints >= data_start) & (changepoints <= data_end)]
        
        # Calculate spacing safely
        if len(changepoints) > 1:
            # Convert to pandas timestamps for proper calculation
            changepoints_ts = pd.to_datetime(changepoints)
            diffs = changepoints_ts.diff().dropna()
            avg_spacing_days = diffs.dt.total_seconds().mean() / 86400
        else:
            avg_spacing_days = 0
        
        return {
            'changepoints_count': len(model.changepoints),
            'valid_changepoints_count': len(valid_changepoints),
            'changepoints_dates': [cp.strftime('%Y-%m-%d') for cp in valid_changepoints],
            'changepoint_spacing_days': float(avg_spacing_days)
        }
    
    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall forecast quality score (0-100)"""
        try:
            score_components = []
            
            # Coverage score (0-25 points)
            coverage = analysis.get('forecast_coverage', {}).get('coverage_ratio', 0)
            coverage_score = min(25, coverage * 25)
            score_components.append(coverage_score)
            
            # Residual analysis score (0-25 points)
            residual_analysis = analysis.get('residual_analysis', {})
            if 'error' not in residual_analysis:
                residual_score = 0
                # Normality (0-10 points)
                if residual_analysis.get('is_normally_distributed', False):
                    residual_score += 10
                # No autocorrelation (0-10 points)
                if not residual_analysis.get('has_autocorrelation', True):
                    residual_score += 10
                # Low residual std (0-5 points)
                std_residual = residual_analysis.get('std_residual', float('inf'))
                if std_residual < 1:
                    residual_score += 5
            else:
                residual_score = 0
            score_components.append(residual_score)
            
            # Trend quality score (0-25 points)
            trend_analysis = analysis.get('trend_analysis', {})
            if 'error' not in trend_analysis:
                trend_score = max(0, 25 - trend_analysis.get('significant_changes', 10))
            else:
                trend_score = 0
            score_components.append(trend_score)
            
            # Uncertainty quality score (0-25 points)
            uncertainty_analysis = analysis.get('uncertainty_analysis', {})
            if 'error' not in uncertainty_analysis:
                relative_width = uncertainty_analysis.get('mean_relative_width', float('inf'))
                uncertainty_score = max(0, 25 - min(25, relative_width * 100))
            else:
                uncertainty_score = 0
            score_components.append(uncertainty_score)
            
            return float(sum(score_components))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {str(e)}")
            return 0.0

class ProphetDiagnosticPlots:
    """Extended diagnostic plots for Prophet forecasts"""
    
    def __init__(self, config: Optional[ProphetDiagnosticConfig] = None):
        self.config = config or ProphetDiagnosticConfig()
        self.analyzer = ProphetDiagnosticAnalyzer(config)
    
    def create_comprehensive_diagnostic_report(self, forecast_result: ProphetForecastResult, 
                                             actual_data: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, go.Figure]:
        """
        Create comprehensive diagnostic report with multiple plots
        """
        try:
            plots = {}
            
            # Perform analysis
            analysis = self.analyzer.analyze_forecast_quality(forecast_result, actual_data, date_col, target_col)
            
            # Create individual diagnostic plots
            plots['residual_analysis'] = self.create_residual_analysis_plot(analysis.get('residual_analysis', {}))
            plots['trend_decomposition'] = self.create_trend_decomposition_plot(forecast_result)
            plots['seasonality_analysis'] = self.create_seasonality_analysis_plot(forecast_result)
            plots['uncertainty_analysis'] = self.create_uncertainty_analysis_plot(forecast_result)
            plots['quality_dashboard'] = self.create_quality_dashboard(analysis)
            plots['forecast_validation'] = self.create_forecast_validation_plot(forecast_result, actual_data, date_col, target_col)
            
            return plots
            
        except Exception as e:
            logger.error(f"Error creating diagnostic report: {str(e)}")
            return {'error': go.Figure().add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5)}
    
    def create_residual_analysis_plot(self, residual_analysis: Dict[str, Any]) -> go.Figure:
        """Create comprehensive residual analysis plot"""
        if 'error' in residual_analysis:
            fig = go.Figure()
            fig.add_annotation(text=f"Residual Analysis Error: {residual_analysis['error']}", 
                             x=0.5, y=0.5, showarrow=False)
            return fig
        
        residuals = residual_analysis.get('residuals', pd.Series())
        dates = residual_analysis.get('dates', pd.Series())
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Residuals Over Time', 'Residuals Distribution', 
                           'Q-Q Plot', 'Autocorrelation'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Residuals over time
        if len(residuals) > 0 and len(dates) > 0:
            fig.add_trace(
                go.Scatter(x=dates, y=residuals, mode='markers', 
                          name='Residuals', marker=dict(color=self.config.colors['primary'], size=4)),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color=self.config.colors['warning'], 
                         row=1, col=1)
        
        # 2. Histogram of residuals
        if len(residuals) > 0:
            fig.add_trace(
                go.Histogram(x=residuals, nbinsx=30, name='Distribution',
                           marker=dict(color=self.config.colors['secondary'])),
                row=1, col=2
            )
        
        # 3. Q-Q plot
        if len(residuals) > 0:
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
            sample_quantiles = np.sort(residuals)
            
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers',
                          name='Q-Q Plot', marker=dict(color=self.config.colors['success'])),
                row=2, col=1
            )
            
            # Add diagonal line
            min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
            max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
            fig.add_trace(
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                          line=dict(dash='dash', color=self.config.colors['warning']),
                          name='Perfect Fit', showlegend=False),
                row=2, col=1
            )
        
        # 4. Autocorrelation function
        if len(residuals) > 1:
            lags = range(1, min(21, len(residuals) // 4))
            autocorrs = [residuals.autocorr(lag=lag) for lag in lags]
            
            fig.add_trace(
                go.Bar(x=list(lags), y=autocorrs, name='Autocorrelation',
                       marker=dict(color=self.config.colors['info'])),
                row=2, col=2
            )
            
            # Add significance bounds
            bound = 1.96 / np.sqrt(len(residuals))
            fig.add_hline(y=bound, line_dash="dash", line_color=self.config.colors['warning'], 
                         row=2, col=2)
            fig.add_hline(y=-bound, line_dash="dash", line_color=self.config.colors['warning'], 
                         row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Residual Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_trend_decomposition_plot(self, forecast_result: ProphetForecastResult) -> go.Figure:
        """Create trend decomposition and analysis plot"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            fig = go.Figure()
            fig.add_annotation(text="No forecast data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        forecast = forecast_result.raw_forecast
        
        # Create subplots
        components = ['yhat', 'trend']
        if 'yearly' in forecast.columns:
            components.append('yearly')
        if 'weekly' in forecast.columns:
            components.append('weekly')
        
        fig = make_subplots(
            rows=len(components), cols=1,
            subplot_titles=[comp.title() for comp in components],
            vertical_spacing=self.config.subplot_spacing
        )
        
        colors = [self.config.colors['primary'], self.config.colors['secondary'], 
                 self.config.colors['success'], self.config.colors['info']]
        
        for i, component in enumerate(components):
            if component in forecast.columns:
                fig.add_trace(
                    go.Scatter(x=forecast['ds'], y=forecast[component], 
                              mode='lines', name=component.title(),
                              line=dict(color=colors[i % len(colors)], width=self.config.line_width)),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title="Trend Decomposition Analysis",
            height=self.config.plot_height * len(components),
            showlegend=False
        )
        
        return fig
    
    def create_seasonality_analysis_plot(self, forecast_result: ProphetForecastResult) -> go.Figure:
        """Create seasonality analysis plot"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            fig = go.Figure()
            fig.add_annotation(text="No forecast data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        forecast = forecast_result.raw_forecast
        seasonal_components = [col for col in ['yearly', 'weekly', 'daily'] if col in forecast.columns]
        
        if not seasonal_components:
            fig = go.Figure()
            fig.add_annotation(text="No seasonal components available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        fig = make_subplots(
            rows=len(seasonal_components), cols=1,
            subplot_titles=[f'{comp.title()} Seasonality' for comp in seasonal_components],
            vertical_spacing=self.config.subplot_spacing
        )
        
        colors = [self.config.colors['success'], self.config.colors['info'], self.config.colors['warning']]
        
        for i, component in enumerate(seasonal_components):
            fig.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast[component], 
                          mode='lines', name=f'{component.title()} Seasonality',
                          line=dict(color=colors[i % len(colors)], width=self.config.line_width)),
                row=i+1, col=1
            )
        
        fig.update_layout(
            title="Seasonality Components Analysis",
            height=self.config.plot_height * len(seasonal_components),
            showlegend=False
        )
        
        return fig
    
    def create_uncertainty_analysis_plot(self, forecast_result: ProphetForecastResult) -> go.Figure:
        """Create uncertainty intervals analysis plot"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            fig = go.Figure()
            fig.add_annotation(text="No forecast data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        forecast = forecast_result.raw_forecast
        required_cols = ['yhat', 'yhat_lower', 'yhat_upper']
        
        if not all(col in forecast.columns for col in required_cols):
            fig = go.Figure()
            fig.add_annotation(text="No uncertainty intervals available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Calculate interval metrics
        interval_width = forecast['yhat_upper'] - forecast['yhat_lower']
        relative_width = interval_width / (forecast['yhat'].abs() + 1e-8) * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Uncertainty Intervals', 'Relative Interval Width (%)'],
            vertical_spacing=self.config.subplot_spacing
        )
        
        # Upper plot: uncertainty intervals
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                      mode='lines', name='Upper Bound', 
                      line=dict(color=self.config.colors['secondary'], width=1)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                      mode='lines', name='Lower Bound', fill='tonexty',
                      fillcolor=f"rgba{(*[int(self.config.colors['primary'][i:i+2], 16) for i in (1, 3, 5)], 0.2)}",
                      line=dict(color=self.config.colors['secondary'], width=1)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                      mode='lines', name='Forecast',
                      line=dict(color=self.config.colors['primary'], width=self.config.line_width)),
            row=1, col=1
        )
        
        # Lower plot: relative width
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=relative_width, 
                      mode='lines', name='Relative Width (%)',
                      line=dict(color=self.config.colors['warning'], width=self.config.line_width)),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Uncertainty Analysis",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_quality_dashboard(self, analysis: Dict[str, Any]) -> go.Figure:
        """Create forecast quality dashboard"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Quality Score', 'Coverage Analysis', 'Residual Statistics', 'Trend Quality'],
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "scatter"}]]
            )
            
            # Quality score gauge
            quality_score = analysis.get('quality_score', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=quality_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Quality Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.config.colors['primary']},
                        'steps': [
                            {'range': [0, 50], 'color': self.config.colors['warning']},
                            {'range': [50, 75], 'color': self.config.colors['secondary']},
                            {'range': [75, 100], 'color': self.config.colors['success']}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Coverage analysis
            coverage_data = analysis.get('forecast_coverage', {})
            if 'error' not in coverage_data:
                coverage_ratio = coverage_data.get('coverage_ratio', 0) * 100
                fig.add_trace(
                    go.Bar(x=['Coverage'], y=[coverage_ratio], 
                          marker=dict(color=self.config.colors['success']),
                          name='Coverage %'),
                    row=1, col=2
                )
            
            # Residual statistics
            residual_data = analysis.get('residual_analysis', {})
            if 'error' not in residual_data:
                stats_names = ['Normality', 'No Autocorr', 'Low Std']
                stats_values = [
                    100 if residual_data.get('is_normally_distributed', False) else 0,
                    100 if not residual_data.get('has_autocorrelation', True) else 0,
                    100 if residual_data.get('std_residual', float('inf')) < 1 else 0
                ]
                
                colors = [self.config.colors['success'] if v > 50 else self.config.colors['warning'] 
                         for v in stats_values]
                
                fig.add_trace(
                    go.Bar(x=stats_names, y=stats_values, 
                          marker=dict(color=colors),
                          name='Residual Quality'),
                    row=2, col=1
                )
            
            # Trend quality over time
            trend_data = analysis.get('trend_analysis', {})
            if 'error' not in trend_data:
                trend_info = [
                    ['Slope', trend_data.get('trend_slope', 0)],
                    ['Volatility', trend_data.get('trend_volatility', 0)],
                    ['Changes', trend_data.get('significant_changes', 0)]
                ]
                
                fig.add_trace(
                    go.Scatter(x=[info[0] for info in trend_info], 
                              y=[info[1] for info in trend_info],
                              mode='markers+lines',
                              marker=dict(size=10, color=self.config.colors['info']),
                              name='Trend Metrics'),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Forecast Quality Dashboard",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating quality dashboard: {str(e)}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating dashboard: {str(e)}", x=0.5, y=0.5, showarrow=False)
            return fig
    
    def create_forecast_validation_plot(self, forecast_result: ProphetForecastResult, 
                                      actual_data: pd.DataFrame, date_col: str, target_col: str) -> go.Figure:
        """Create forecast validation and comparison plot"""
        if not forecast_result.success or forecast_result.raw_forecast is None:
            fig = go.Figure()
            fig.add_annotation(text="No forecast data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        forecast = forecast_result.raw_forecast
        
        # Merge with actual data for validation
        actual_df = actual_data[[date_col, target_col]].copy()
        actual_df.columns = ['ds', 'y_actual']
        merged = pd.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                         actual_df, on='ds', how='left')
        
        fig = go.Figure()
        
        # Add actual values
        actual_mask = merged['y_actual'].notna()
        if actual_mask.any():
            fig.add_trace(
                go.Scatter(x=merged.loc[actual_mask, 'ds'], 
                          y=merged.loc[actual_mask, 'y_actual'],
                          mode='markers', name='Actual Values',
                          marker=dict(color=self.config.colors['success'], size=4))
            )
        
        # Add forecast
        fig.add_trace(
            go.Scatter(x=merged['ds'], y=merged['yhat'],
                      mode='lines', name='Forecast',
                      line=dict(color=self.config.colors['primary'], width=self.config.line_width))
        )
        
        # Add confidence intervals
        fig.add_trace(
            go.Scatter(x=merged['ds'], y=merged['yhat_upper'],
                      mode='lines', name='Upper Bound', 
                      line=dict(color=self.config.colors['secondary'], width=1))
        )
        
        fig.add_trace(
            go.Scatter(x=merged['ds'], y=merged['yhat_lower'],
                      mode='lines', name='Lower Bound', fill='tonexty',
                      fillcolor=f"rgba{(*[int(self.config.colors['primary'][i:i+2], 16) for i in (1, 3, 5)], 0.2)}",
                      line=dict(color=self.config.colors['secondary'], width=1))
        )
        
        # Add validation split line if there's both historical and future data
        if actual_mask.any():
            last_actual_date = merged.loc[actual_mask, 'ds'].max()
            fig.add_vline(x=last_actual_date, line_dash="dash", 
                         line_color=self.config.colors['warning'],
                         annotation_text="Forecast Start")
        
        fig.update_layout(
            title="Forecast Validation and Comparison",
            xaxis_title="Date",
            yaxis_title="Value",
            height=self.config.plot_height,
            hovermode='x unified'
        )
        
        return fig

# Factory functions
def create_diagnostic_analyzer(config: Optional[ProphetDiagnosticConfig] = None) -> ProphetDiagnosticAnalyzer:
    """Factory function to create diagnostic analyzer"""
    return ProphetDiagnosticAnalyzer(config)

def create_diagnostic_plots(config: Optional[ProphetDiagnosticConfig] = None) -> ProphetDiagnosticPlots:
    """Factory function to create diagnostic plots"""
    return ProphetDiagnosticPlots(config)
