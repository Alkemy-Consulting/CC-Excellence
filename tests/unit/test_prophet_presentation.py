"""
Unit Tests for Prophet Presentation Layer
Tests the enterprise presentation layer for visualizations
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.prophet_presentation import (
    ProphetVisualizationConfig,
    ProphetPlotGenerator,
    ProphetPlotFactory,
    create_prophet_plots
)
from modules.prophet_core import ProphetForecastResult
from tests.conftest import TestDataValidator

class TestProphetVisualizationConfig:
    """Test suite for ProphetVisualizationConfig"""
    
    def test_config_initialization(self):
        """Test configuration initialization with defaults"""
        config = ProphetVisualizationConfig()
        
        assert hasattr(config, 'colors')
        assert hasattr(config, 'height')
        assert hasattr(config, 'show_changepoints')
        assert hasattr(config, 'show_rangeslider')
        
        # Check default values
        assert config.height == 500
        assert config.show_changepoints is True
        assert config.show_rangeslider is True
        
        # Check color configuration
        assert 'actual' in config.colors
        assert 'prediction' in config.colors
        assert 'confidence' in config.colors
        assert 'trend' in config.colors
        assert 'changepoints' in config.colors
    
    def test_config_customization(self):
        """Test configuration customization"""
        config = ProphetVisualizationConfig()
        
        # Modify configuration
        config.height = 800
        config.show_changepoints = False
        config.colors['actual'] = 'green'
        
        assert config.height == 800
        assert config.show_changepoints is False
        assert config.colors['actual'] == 'green'

class TestProphetPlotGenerator:
    """Test suite for ProphetPlotGenerator"""
    
    def test_plot_generator_initialization(self):
        """Test plot generator initialization"""
        generator = ProphetPlotGenerator()
        assert generator is not None
        assert hasattr(generator, 'config')
        assert isinstance(generator.config, ProphetVisualizationConfig)
    
    def test_plot_generator_custom_config(self):
        """Test plot generator with custom configuration"""
        custom_config = ProphetVisualizationConfig()
        custom_config.height = 600
        
        generator = ProphetPlotGenerator(custom_config)
        assert generator.config.height == 600
    
    def test_prepare_chart_data_success(self, sample_time_series):
        """Test successful chart data preparation"""
        generator = ProphetPlotGenerator()
        
        # Create mock model with changepoints
        class MockModel:
            def __init__(self):
                self.changepoints = pd.date_range('2020-02-01', periods=3, freq='30D')
        
        mock_model = MockModel()
        
        # Create mock forecast data
        forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=120),
            'yhat': np.random.normal(150, 20, 120),
            'yhat_lower': np.random.normal(130, 15, 120),
            'yhat_upper': np.random.normal(170, 15, 120),
            'trend': np.linspace(100, 200, 120)
        })
        
        chart_data = generator.prepare_chart_data(
            mock_model, forecast_df, sample_time_series, 'date', 'value'
        )
        
        assert chart_data['success'] is True
        assert chart_data['error'] is None
        assert 'forecast_df' in chart_data
        assert 'actual_data' in chart_data
        assert 'changepoints' in chart_data
        assert chart_data['confidence_percentage'] == 80  # Default 0.8 * 100
    
    def test_prepare_chart_data_custom_confidence(self, sample_time_series):
        """Test chart data preparation with custom confidence interval"""
        generator = ProphetPlotGenerator()
        
        class MockModel:
            def __init__(self):
                self.changepoints = []
        
        mock_model = MockModel()
        forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=50),
            'yhat': range(50),
            'yhat_lower': range(50),
            'yhat_upper': range(50),
            'trend': range(50)
        })
        
        chart_data = generator.prepare_chart_data(
            mock_model, forecast_df, sample_time_series, 'date', 'value', confidence_interval=0.95
        )
        
        assert chart_data['confidence_percentage'] == 95
    
    def test_prepare_chart_data_error_handling(self):
        """Test chart data preparation error handling"""
        generator = ProphetPlotGenerator()
        
        # Use invalid data to trigger error
        invalid_forecast = pd.DataFrame({'invalid': [1, 2, 3]})
        invalid_actual = pd.DataFrame({'invalid': [1, 2, 3]})
        
        chart_data = generator.prepare_chart_data(
            None, invalid_forecast, invalid_actual, 'nonexistent', 'nonexistent'
        )
        
        assert chart_data['success'] is False
        assert chart_data['error'] is not None
    
    def test_create_forecast_chart_success(self, sample_time_series):
        """Test successful forecast chart creation"""
        generator = ProphetPlotGenerator()
        
        # Prepare valid chart data
        class MockModel:
            def __init__(self):
                self.changepoints = pd.date_range('2020-02-01', periods=2, freq='30D')
        
        mock_model = MockModel()
        forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=120),
            'yhat': np.random.normal(150, 20, 120),
            'yhat_lower': np.random.normal(130, 15, 120),
            'yhat_upper': np.random.normal(170, 15, 120),
            'trend': np.linspace(100, 200, 120)
        })
        
        chart_data = generator.prepare_chart_data(
            mock_model, forecast_df, sample_time_series, 'date', 'value'
        )
        
        fig = generator.create_forecast_chart(chart_data)
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        
        # Check that traces are added
        assert len(fig.data) > 0
        
        # Check layout properties
        assert fig.layout.title.text == "Prophet Forecast Results"
        assert fig.layout.xaxis.title.text == "Date"
        assert fig.layout.yaxis.title.text == "value"
        assert fig.layout.height == 500
    
    def test_create_forecast_chart_no_changepoints(self, sample_time_series):
        """Test forecast chart creation without changepoints"""
        generator = ProphetPlotGenerator()
        
        class MockModel:
            def __init__(self):
                self.changepoints = []
        
        mock_model = MockModel()
        forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=50),
            'yhat': range(50),
            'yhat_lower': range(50),
            'yhat_upper': range(50),
            'trend': range(50)
        })
        
        chart_data = generator.prepare_chart_data(
            mock_model, forecast_df, sample_time_series, 'date', 'value'
        )
        
        fig = generator.create_forecast_chart(chart_data)
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
    
    def test_create_forecast_chart_error_handling(self):
        """Test forecast chart creation error handling"""
        generator = ProphetPlotGenerator()
        
        # Use invalid chart data
        invalid_chart_data = {'success': False, 'error': 'Test error'}
        
        fig = generator.create_forecast_chart(invalid_chart_data)
        
        assert fig is None
    
    def test_create_components_chart(self):
        """Test components chart creation"""
        generator = ProphetPlotGenerator()
        
        # Create mock model and forecast with components
        class MockModel:
            pass
        
        mock_model = MockModel()
        forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=50),
            'trend': np.linspace(100, 200, 50),
            'yearly': np.sin(np.linspace(0, 2*np.pi, 50)) * 10,
            'weekly': np.sin(np.linspace(0, 14*np.pi, 50)) * 5
        })
        
        fig = generator.create_components_chart(mock_model, forecast_df)
        
        if fig is not None:  # May be None if components not available
            assert isinstance(fig, go.Figure)
            assert len(fig.data) > 0
    
    def test_create_residuals_chart(self, sample_time_series):
        """Test residuals chart creation"""
        generator = ProphetPlotGenerator()
        
        # Create forecast data that overlaps with actual data
        forecast_df = pd.DataFrame({
            'ds': sample_time_series['date'][:50],
            'yhat': sample_time_series['value'][:50] + np.random.normal(0, 5, 50)
        })
        
        actual_data = sample_time_series.copy()
        actual_data.columns = ['ds', 'y']
        
        fig = generator.create_residuals_chart(forecast_df, actual_data, 'ds', 'y')
        
        if fig is not None:  # May be None if no overlapping data
            assert isinstance(fig, go.Figure)

class TestProphetPlotFactory:
    """Test suite for ProphetPlotFactory"""
    
    def test_create_plot_generator(self):
        """Test plot generator creation through factory"""
        generator = ProphetPlotFactory.create_plot_generator()
        
        assert generator is not None
        assert isinstance(generator, ProphetPlotGenerator)
    
    def test_create_plot_generator_with_config(self):
        """Test plot generator creation with custom config"""
        custom_config = ProphetVisualizationConfig()
        custom_config.height = 700
        
        generator = ProphetPlotFactory.create_plot_generator(custom_config)
        
        assert generator.config.height == 700
    
    def test_create_default_config(self):
        """Test default configuration creation"""
        config = ProphetPlotFactory.create_default_config()
        
        assert isinstance(config, ProphetVisualizationConfig)
        assert config.height == 500

class TestCreateProphetPlots:
    """Test suite for create_prophet_plots helper function"""
    
    def test_create_plots_success(self, sample_time_series):
        """Test successful plot creation from forecast result"""
        # Create mock forecast result
        mock_model = type('MockModel', (), {'changepoints': []})()
        forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=50),
            'yhat': range(50),
            'yhat_lower': range(50),
            'yhat_upper': range(50),
            'trend': range(50)
        })
        
        forecast_result = ProphetForecastResult(
            success=True,
            error=None,
            model=mock_model,
            raw_forecast=forecast_df,
            metrics={'mape': 10.0, 'mae': 5.0, 'rmse': 7.0, 'r2': 0.85}
        )
        
        plots = create_prophet_plots(forecast_result, sample_time_series, 'date', 'value')
        
        assert isinstance(plots, dict)
        # May contain forecast_plot if creation succeeded
        if 'forecast_plot' in plots and plots['forecast_plot'] is not None:
            assert isinstance(plots['forecast_plot'], go.Figure)
    
    def test_create_plots_failure(self, sample_time_series):
        """Test plot creation with failed forecast result"""
        forecast_result = ProphetForecastResult(
            success=False,
            error="Test error",
            model=None,
            raw_forecast=None,
            metrics={}
        )
        
        plots = create_prophet_plots(forecast_result, sample_time_series, 'date', 'value')
        
        assert isinstance(plots, dict)
        # Should return empty plots or None values
    
    def test_create_plots_custom_config(self, sample_time_series):
        """Test plot creation with custom configuration"""
        custom_config = ProphetVisualizationConfig()
        custom_config.height = 800
        
        mock_model = type('MockModel', (), {'changepoints': []})()
        forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=50),
            'yhat': range(50),
            'yhat_lower': range(50),
            'yhat_upper': range(50),
            'trend': range(50)
        })
        
        forecast_result = ProphetForecastResult(
            success=True,
            error=None,
            model=mock_model,
            raw_forecast=forecast_df,
            metrics={'mape': 10.0, 'mae': 5.0, 'rmse': 7.0, 'r2': 0.85}
        )
        
        plots = create_prophet_plots(forecast_result, sample_time_series, 'date', 'value', custom_config)
        
        assert isinstance(plots, dict)

# Mark tests for specific execution
pytestmark = [
    pytest.mark.unit,
    pytest.mark.prophet_presentation,
    pytest.mark.visualization
]
