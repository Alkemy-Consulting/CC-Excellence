"""
Prophet Presentation Layer Module
Handles all visualization and UI components - Enterprise Architecture Layer
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional, Any, List
import logging
from datetime import datetime

from src.modules.forecasting.prophet_core import ProphetForecastResult

logger = logging.getLogger(__name__)

class ProphetVisualizationConfig:
    """Configuration class for Prophet visualizations"""
    def __init__(self):
        self.colors = {
            'actual': 'white',
            'prediction': 'blue', 
            'confidence': 'rgba(0, 100, 255, 0.2)',
            'trend': 'red',
            'changepoints': 'orange'
        }
        self.height = 500
        self.show_changepoints = True
        self.show_rangeslider = True

class ProphetPlotGenerator:
    """Generates all Prophet-related visualizations"""
    
    def __init__(self, config: Optional[ProphetVisualizationConfig] = None):
        self.config = config or ProphetVisualizationConfig()
    
    def prepare_chart_data(self, model, forecast_df: pd.DataFrame, actual_data: pd.DataFrame, 
                          date_col: str, target_col: str, confidence_interval: float = 0.8) -> Dict[str, Any]:
        """
        Prepare data for chart creation - pure data processing
        Returns: Dictionary with processed data for plotting
        """
        try:
            logger.info(f"Preparing chart data with confidence_interval={confidence_interval}")
            
            # Calculate confidence percentage for display
            confidence_percentage = int(confidence_interval * 100)
            
            # Ensure proper datetime conversion for all data
            forecast_df = forecast_df.copy()
            actual_data = actual_data.copy()
            
            # Convert date columns to datetime if they aren't already
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            actual_data[date_col] = pd.to_datetime(actual_data[date_col])
            
            # Filter changepoints to those within the actual data range
            changepoints_data = []
            if hasattr(model, 'changepoints') and len(model.changepoints) > 0 and self.config.show_changepoints:
                data_start = actual_data[date_col].min()
                data_end = actual_data[date_col].max()
                
                for i, changepoint in enumerate(model.changepoints):
                    cp_date = pd.to_datetime(changepoint)
                    if data_start <= cp_date <= data_end:
                        changepoints_data.append({
                            'date': cp_date,
                            'label': f"CP {i+1}",
                            'index': i
                        })
            
            return {
                'forecast_df': forecast_df,
                'actual_data': actual_data,
                'date_col': date_col,
                'target_col': target_col,
                'confidence_percentage': confidence_percentage,
                'changepoints': changepoints_data,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error preparing chart data: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_forecast_chart(self, chart_data: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Create Prophet forecast chart - pure visualization logic
        """
        try:
            if not chart_data.get('success', False):
                logger.error(f"Cannot create chart: {chart_data.get('error', 'Unknown error')}")
                return None
            
            forecast_df = chart_data['forecast_df']
            actual_data = chart_data['actual_data']
            date_col = chart_data['date_col']
            target_col = chart_data['target_col']
            confidence_percentage = chart_data['confidence_percentage']
            changepoints = chart_data['changepoints']
            
            # Create the main plot
            fig = go.Figure()
            
            # Add actual values (white points for training period)
            fig.add_trace(go.Scatter(
                x=actual_data[date_col],
                y=actual_data[target_col],
                mode='markers',
                name='Actual Values',
                marker=dict(color=self.config.colors['actual'], size=4, line=dict(color='black', width=1)),
                hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
            
            # Add predictions (blue line)
            fig.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat'],
                mode='lines',
                name='Predictions',
                line=dict(color=self.config.colors['prediction'], width=2),
                hovertemplate='<b>Prediction</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
            
            # Add uncertainty interval (blue shade)
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
                fillcolor=self.config.colors['confidence'],
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
                line=dict(color=self.config.colors['trend'], width=2),
                hovertemplate='<b>Trend</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
            
            # Add changepoints using add_shape() for pandas >= 2.0 compatibility
            for cp in changepoints:
                fig.add_shape(
                    type="line",
                    x0=cp['date'], x1=cp['date'],
                    y0=0, y1=1,
                    yref="paper",
                    line=dict(color=self.config.colors['changepoints'], width=1, dash='dot'),
                    opacity=0.7
                )
                # Add annotation separately for the changepoint label
                fig.add_annotation(
                    x=cp['date'],
                    y=1.02,
                    yref="paper", 
                    text=cp['label'],
                    showarrow=False,
                    font=dict(color=self.config.colors['changepoints'], size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=self.config.colors['changepoints'],
                    borderwidth=1
                )
            
            if changepoints:
                logger.info(f"{len(changepoints)} changepoints added successfully")
            
            # Apply chart formatting
            fig.update_layout(
                title="Prophet Forecast Results",
                xaxis_title="Date",
                yaxis_title=target_col,
                height=self.config.height,
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
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=2, label="2Y", step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ]),
                        x=0.02,
                        xanchor="left",
                        y=1.02,
                        yanchor="bottom"
                    ),
                    rangeslider=dict(visible=self.config.show_rangeslider),
                    type="date"
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating Prophet forecast chart: {str(e)}")
            return None
    
    def create_components_chart(self, model, forecast_df: pd.DataFrame) -> Optional[go.Figure]:
        """
        Create Prophet components chart (trend, seasonality)
        """
        try:
            # Create subplots for components
            n_components = 2  # trend + yearly
            if 'weekly' in forecast_df.columns:
                n_components += 1
            if 'daily' in forecast_df.columns:
                n_components += 1
            
            fig = make_subplots(
                rows=n_components, cols=1,
                subplot_titles=['Trend', 'Yearly', 'Weekly', 'Daily'][:n_components],
                vertical_spacing=0.1
            )
            
            # Add trend component
            fig.add_trace(
                go.Scatter(x=forecast_df['ds'], y=forecast_df['trend'], 
                          name='Trend', line=dict(color='red')),
                row=1, col=1
            )
            
            # Add yearly seasonality if available
            if 'yearly' in forecast_df.columns:
                fig.add_trace(
                    go.Scatter(x=forecast_df['ds'], y=forecast_df['yearly'], 
                              name='Yearly', line=dict(color='blue')),
                    row=2, col=1
                )
            
            # Add weekly seasonality if available
            row_idx = 3
            if 'weekly' in forecast_df.columns and n_components >= 3:
                fig.add_trace(
                    go.Scatter(x=forecast_df['ds'], y=forecast_df['weekly'], 
                              name='Weekly', line=dict(color='green')),
                    row=row_idx, col=1
                )
                row_idx += 1
            
            # Add daily seasonality if available  
            if 'daily' in forecast_df.columns and n_components >= 4:
                fig.add_trace(
                    go.Scatter(x=forecast_df['ds'], y=forecast_df['daily'], 
                              name='Daily', line=dict(color='orange')),
                    row=row_idx, col=1
                )
            
            fig.update_layout(
                height=150 * n_components,
                title="Prophet Model Components",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating components chart: {str(e)}")
            return None
    
    def create_residuals_chart(self, forecast_df: pd.DataFrame, actual_data: pd.DataFrame,
                              date_col: str, target_col: str) -> Optional[go.Figure]:
        """
        Create residuals analysis chart
        """
        try:
            # Merge forecast with actual data for residuals calculation
            actual_data = actual_data.copy()
            actual_data.columns = ['ds', 'y'] if len(actual_data.columns) == 2 else actual_data.columns
            
            # Get overlapping period
            merged = pd.merge(forecast_df[['ds', 'yhat']], actual_data, on='ds', how='inner')
            if merged.empty:
                logger.warning("No overlapping data for residuals calculation")
                return None
            
            # Calculate residuals
            merged['residuals'] = merged['y'] - merged['yhat']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Residuals Over Time', 'Residuals Distribution', 
                               'Q-Q Plot', 'Residuals vs Fitted'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Residuals over time
            fig.add_trace(
                go.Scatter(x=merged['ds'], y=merged['residuals'], 
                          mode='markers', name='Residuals'),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # Residuals histogram
            fig.add_trace(
                go.Histogram(x=merged['residuals'], name='Distribution', nbinsx=20),
                row=1, col=2
            )
            
            # Residuals vs fitted
            fig.add_trace(
                go.Scatter(x=merged['yhat'], y=merged['residuals'], 
                          mode='markers', name='Residuals vs Fitted'),
                row=2, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)
            
            fig.update_layout(
                height=600,
                title="Residuals Analysis",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating residuals chart: {str(e)}")
            return None

class ProphetPlotFactory:
    """Factory class for creating Prophet visualization objects"""
    
    @staticmethod
    def create_plot_generator(config: Optional[ProphetVisualizationConfig] = None) -> ProphetPlotGenerator:
        """Create a ProphetPlotGenerator instance"""
        return ProphetPlotGenerator(config)
    
    @staticmethod
    def create_default_config() -> ProphetVisualizationConfig:
        """Create default visualization configuration"""
        return ProphetVisualizationConfig()

# Helper functions for backward compatibility
def create_prophet_plots(forecast_result: ProphetForecastResult, actual_data: pd.DataFrame,
                        date_col: str, target_col: str, 
                        config: Optional[ProphetVisualizationConfig] = None) -> Dict[str, Optional[go.Figure]]:
    """
    Create all Prophet plots from forecast result
    Returns: Dictionary with plot names as keys and Figure objects as values
    """
    plot_generator = ProphetPlotFactory.create_plot_generator(config)
    plots = {}
    
    try:
        if forecast_result.success and forecast_result.model is not None:
            # Prepare chart data
            chart_data = plot_generator.prepare_chart_data(
                forecast_result.model, forecast_result.raw_forecast, 
                actual_data, date_col, target_col
            )
            
            # Create forecast chart
            plots['forecast_plot'] = plot_generator.create_forecast_chart(chart_data)
            
            # Create components chart  
            plots['components_plot'] = plot_generator.create_components_chart(
                forecast_result.model, forecast_result.raw_forecast
            )
            
            # Create residuals chart
            plots['residuals_plot'] = plot_generator.create_residuals_chart(
                forecast_result.raw_forecast, actual_data, date_col, target_col
            )
            
        else:
            logger.error(f"Cannot create plots - forecast failed: {forecast_result.error}")
            
    except Exception as e:
        logger.error(f"Error creating Prophet plots: {str(e)}")
    
    return plots
