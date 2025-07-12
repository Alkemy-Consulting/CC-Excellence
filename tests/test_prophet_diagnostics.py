"""
Test suite for Prophet Diagnostics Module
Enterprise-level testing for extended diagnostic capabilities
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from modules.prophet_diagnostics import (
    ProphetDiagnosticConfig,
    ProphetDiagnosticAnalyzer, 
    ProphetDiagnosticPlots,
    create_diagnostic_analyzer,
    create_diagnostic_plots
)
from modules.prophet_core import ProphetForecastResult

class TestProphetDiagnosticConfig:
    """Test diagnostic configuration"""
    
    def test_config_initialization(self):
        config = ProphetDiagnosticConfig()
        assert config.plot_height == 400
        assert config.subplot_spacing == 0.08
        assert config.font_size == 12
        assert config.line_width == 2
        assert 'primary' in config.colors
        assert 'secondary' in config.colors

class TestProphetDiagnosticAnalyzer:
    """Test diagnostic analyzer functionality"""
    
    @pytest.fixture
    def sample_forecast_result(self):
        """Create sample forecast result for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        forecast_data = pd.DataFrame({
            'ds': dates,
            'yhat': np.random.normal(100, 10, 100),
            'yhat_lower': np.random.normal(90, 8, 100),
            'yhat_upper': np.random.normal(110, 8, 100),
            'trend': np.linspace(95, 105, 100),
            'yearly': np.sin(np.arange(100) * 2 * np.pi / 365) * 5,
            'weekly': np.sin(np.arange(100) * 2 * np.pi / 7) * 2
        })
        
        mock_model = Mock()
        mock_model.changepoints = pd.date_range('2023-01-10', periods=5, freq='10D')
        
        return ProphetForecastResult(
            success=True,
            error=None,
            model=mock_model,
            raw_forecast=forecast_data,
            metrics={'mae': 5.0, 'mape': 10.0, 'rmse': 7.0}
        )
    
    @pytest.fixture
    def sample_actual_data(self):
        """Create sample actual data for testing"""
        dates = pd.date_range('2023-01-01', periods=80, freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, 80)
        })
    
    def test_analyzer_initialization(self):
        analyzer = ProphetDiagnosticAnalyzer()
        assert analyzer.config is not None
        assert isinstance(analyzer.config, ProphetDiagnosticConfig)
    
    def test_analyzer_with_custom_config(self):
        custom_config = ProphetDiagnosticConfig()
        custom_config.plot_height = 500
        analyzer = ProphetDiagnosticAnalyzer(custom_config)
        assert analyzer.config.plot_height == 500
    
    def test_analyze_forecast_quality(self, sample_forecast_result, sample_actual_data):
        analyzer = ProphetDiagnosticAnalyzer()
        analysis = analyzer.analyze_forecast_quality(
            sample_forecast_result, sample_actual_data, 'date', 'value'
        )
        
        assert 'forecast_coverage' in analysis
        assert 'residual_analysis' in analysis
        assert 'trend_analysis' in analysis
        assert 'seasonality_analysis' in analysis
        assert 'uncertainty_analysis' in analysis
        assert 'changepoint_analysis' in analysis
        assert 'quality_score' in analysis
        assert isinstance(analysis['quality_score'], (int, float))
        assert 0 <= analysis['quality_score'] <= 100
    
    def test_analyze_forecast_coverage(self, sample_forecast_result, sample_actual_data):
        analyzer = ProphetDiagnosticAnalyzer()
        coverage = analyzer._analyze_forecast_coverage(
            sample_forecast_result, sample_actual_data, 'date'
        )
        
        assert 'coverage_ratio' in coverage
        assert 'forecast_start' in coverage
        assert 'forecast_end' in coverage
        assert 'actual_start' in coverage
        assert 'actual_end' in coverage
        assert 0 <= coverage['coverage_ratio'] <= 1
    
    def test_analyze_residuals(self, sample_forecast_result, sample_actual_data):
        analyzer = ProphetDiagnosticAnalyzer()
        residuals = analyzer._analyze_residuals(
            sample_forecast_result, sample_actual_data, 'date', 'value'
        )
        
        if 'error' not in residuals:
            assert 'residuals' in residuals
            assert 'mean_residual' in residuals
            assert 'std_residual' in residuals
            assert 'skewness' in residuals
            assert 'kurtosis' in residuals
            assert 'normality_p_value' in residuals
            assert 'is_normally_distributed' in residuals
            assert isinstance(residuals['is_normally_distributed'], bool)
    
    def test_analyze_trend_quality(self, sample_forecast_result):
        analyzer = ProphetDiagnosticAnalyzer()
        trend = analyzer._analyze_trend_quality(sample_forecast_result)
        
        assert 'trend_slope' in trend
        assert 'trend_volatility' in trend
        assert 'trend_range' in trend
        assert 'trend_direction' in trend
        assert trend['trend_direction'] in ['increasing', 'decreasing', 'stable']
        assert 'significant_changes' in trend
    
    def test_analyze_seasonality_quality(self, sample_forecast_result):
        analyzer = ProphetDiagnosticAnalyzer()
        seasonality = analyzer._analyze_seasonality_quality(sample_forecast_result)
        
        assert 'yearly' in seasonality
        assert 'weekly' in seasonality
        for component in ['yearly', 'weekly']:
            assert 'amplitude' in seasonality[component]
            assert 'std' in seasonality[component]
            assert 'strength' in seasonality[component]
    
    def test_analyze_uncertainty_quality(self, sample_forecast_result):
        analyzer = ProphetDiagnosticAnalyzer()
        uncertainty = analyzer._analyze_uncertainty_quality(sample_forecast_result)
        
        assert 'mean_interval_width' in uncertainty
        assert 'std_interval_width' in uncertainty
        assert 'mean_relative_width' in uncertainty
        assert 'max_interval_width' in uncertainty
        assert 'min_interval_width' in uncertainty
    
    def test_analyze_changepoints(self, sample_forecast_result, sample_actual_data):
        analyzer = ProphetDiagnosticAnalyzer()
        changepoints = analyzer._analyze_changepoints(
            sample_forecast_result, sample_actual_data, 'date', 'value'
        )
        
        assert 'changepoints_count' in changepoints
        assert 'valid_changepoints_count' in changepoints
        assert changepoints['changepoints_count'] >= 0
        assert changepoints['valid_changepoints_count'] >= 0
    
    def test_calculate_quality_score(self):
        analyzer = ProphetDiagnosticAnalyzer()
        
        # Test with complete analysis
        analysis = {
            'forecast_coverage': {'coverage_ratio': 0.8},
            'residual_analysis': {
                'is_normally_distributed': True,
                'has_autocorrelation': False,
                'std_residual': 0.5
            },
            'trend_analysis': {'significant_changes': 2},
            'uncertainty_analysis': {'mean_relative_width': 0.1}
        }
        
        score = analyzer._calculate_quality_score(analysis)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100
    
    def test_ljung_box_test(self):
        analyzer = ProphetDiagnosticAnalyzer()
        residuals = pd.Series(np.random.normal(0, 1, 100))
        
        stat, p_value = analyzer._ljung_box_test(residuals)
        assert isinstance(stat, (int, float))
        assert isinstance(p_value, (int, float))
        assert 0 <= p_value <= 1
    
    def test_durbin_watson_test(self):
        analyzer = ProphetDiagnosticAnalyzer()
        residuals = pd.Series(np.random.normal(0, 1, 100))
        
        dw_stat = analyzer._durbin_watson_test(residuals)
        assert isinstance(dw_stat, (int, float))
        assert 0 <= dw_stat <= 4  # DW statistic is typically between 0 and 4
    
    def test_error_handling_no_forecast(self):
        analyzer = ProphetDiagnosticAnalyzer()
        failed_result = ProphetForecastResult(
            success=False,
            error="Test error",
            model=None,
            raw_forecast=None,
            metrics={}
        )
        
        sample_data = pd.DataFrame({'date': [datetime.now()], 'value': [1]})
        analysis = analyzer.analyze_forecast_quality(failed_result, sample_data, 'date', 'value')
        
        # Should handle gracefully
        assert 'quality_score' in analysis

class TestProphetDiagnosticPlots:
    """Test diagnostic plots functionality"""
    
    @pytest.fixture
    def sample_forecast_result(self):
        """Create sample forecast result for testing"""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        forecast_data = pd.DataFrame({
            'ds': dates,
            'yhat': np.random.normal(100, 10, 100),
            'yhat_lower': np.random.normal(90, 8, 100),
            'yhat_upper': np.random.normal(110, 8, 100),
            'trend': np.linspace(95, 105, 100),
            'yearly': np.sin(np.arange(100) * 2 * np.pi / 365) * 5,
            'weekly': np.sin(np.arange(100) * 2 * np.pi / 7) * 2
        })
        
        mock_model = Mock()
        mock_model.changepoints = pd.date_range('2023-01-10', periods=5, freq='10D')
        
        return ProphetForecastResult(
            success=True,
            error=None,
            model=mock_model,
            raw_forecast=forecast_data,
            metrics={'mae': 5.0, 'mape': 10.0, 'rmse': 7.0}
        )
    
    @pytest.fixture
    def sample_actual_data(self):
        """Create sample actual data for testing"""
        dates = pd.date_range('2023-01-01', periods=80, freq='D')
        return pd.DataFrame({
            'date': dates,
            'value': np.random.normal(100, 10, 80)
        })
    
    def test_plots_initialization(self):
        plots = ProphetDiagnosticPlots()
        assert plots.config is not None
        assert plots.analyzer is not None
        assert isinstance(plots.config, ProphetDiagnosticConfig)
        assert isinstance(plots.analyzer, ProphetDiagnosticAnalyzer)
    
    def test_comprehensive_diagnostic_report(self, sample_forecast_result, sample_actual_data):
        plots = ProphetDiagnosticPlots()
        report = plots.create_comprehensive_diagnostic_report(
            sample_forecast_result, sample_actual_data, 'date', 'value'
        )
        
        expected_plots = [
            'residual_analysis', 'trend_decomposition', 'seasonality_analysis',
            'uncertainty_analysis', 'quality_dashboard', 'forecast_validation'
        ]
        
        for plot_name in expected_plots:
            assert plot_name in report
            assert isinstance(report[plot_name], go.Figure)
    
    def test_residual_analysis_plot(self, sample_forecast_result, sample_actual_data):
        plots = ProphetDiagnosticPlots()
        
        # Create residual analysis data
        residual_analysis = {
            'residuals': pd.Series(np.random.normal(0, 1, 50)),
            'dates': pd.date_range('2023-01-01', periods=50),
            'mean_residual': 0.1,
            'std_residual': 1.0,
            'is_normally_distributed': True,
            'has_autocorrelation': False
        }
        
        fig = plots.create_residual_analysis_plot(residual_analysis)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_trend_decomposition_plot(self, sample_forecast_result):
        plots = ProphetDiagnosticPlots()
        fig = plots.create_trend_decomposition_plot(sample_forecast_result)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_seasonality_analysis_plot(self, sample_forecast_result):
        plots = ProphetDiagnosticPlots()
        fig = plots.create_seasonality_analysis_plot(sample_forecast_result)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_uncertainty_analysis_plot(self, sample_forecast_result):
        plots = ProphetDiagnosticPlots()
        fig = plots.create_uncertainty_analysis_plot(sample_forecast_result)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_quality_dashboard(self):
        plots = ProphetDiagnosticPlots()
        
        # Create sample analysis data
        analysis = {
            'quality_score': 75,
            'forecast_coverage': {'coverage_ratio': 0.8},
            'residual_analysis': {
                'is_normally_distributed': True,
                'has_autocorrelation': False,
                'std_residual': 0.5
            },
            'trend_analysis': {
                'trend_slope': 0.1,
                'trend_volatility': 2.0,
                'significant_changes': 2
            }
        }
        
        fig = plots.create_quality_dashboard(analysis)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_forecast_validation_plot(self, sample_forecast_result, sample_actual_data):
        plots = ProphetDiagnosticPlots()
        fig = plots.create_forecast_validation_plot(
            sample_forecast_result, sample_actual_data, 'date', 'value'
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_error_handling_no_data(self):
        plots = ProphetDiagnosticPlots()
        failed_result = ProphetForecastResult(
            success=False,
            error="Test error",
            model=None,
            raw_forecast=None,
            metrics={}
        )
        
        fig = plots.create_trend_decomposition_plot(failed_result)
        assert isinstance(fig, go.Figure)
        # Should create figure with error message

class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_diagnostic_analyzer(self):
        analyzer = create_diagnostic_analyzer()
        assert isinstance(analyzer, ProphetDiagnosticAnalyzer)
    
    def test_create_diagnostic_analyzer_with_config(self):
        config = ProphetDiagnosticConfig()
        analyzer = create_diagnostic_analyzer(config)
        assert isinstance(analyzer, ProphetDiagnosticAnalyzer)
        assert analyzer.config == config
    
    def test_create_diagnostic_plots(self):
        plots = create_diagnostic_plots()
        assert isinstance(plots, ProphetDiagnosticPlots)
    
    def test_create_diagnostic_plots_with_config(self):
        config = ProphetDiagnosticConfig()
        plots = create_diagnostic_plots(config)
        assert isinstance(plots, ProphetDiagnosticPlots)
        assert plots.config == config

@pytest.mark.integration
class TestDiagnosticIntegration:
    """Integration tests for diagnostic module"""
    
    def test_full_diagnostic_workflow(self):
        """Test complete diagnostic workflow"""
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        forecast_data = pd.DataFrame({
            'ds': dates,
            'yhat': np.random.normal(100, 10, 100),
            'yhat_lower': np.random.normal(90, 8, 100),
            'yhat_upper': np.random.normal(110, 8, 100),
            'trend': np.linspace(95, 105, 100),
            'yearly': np.sin(np.arange(100) * 2 * np.pi / 365) * 5
        })
        
        mock_model = Mock()
        mock_model.changepoints = pd.date_range('2023-01-10', periods=3, freq='15D')
        
        forecast_result = ProphetForecastResult(
            success=True,
            error=None,
            model=mock_model,
            raw_forecast=forecast_data,
            metrics={'mae': 5.0, 'mape': 10.0, 'rmse': 7.0}
        )
        
        actual_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=80, freq='D'),
            'value': np.random.normal(100, 10, 80)
        })
        
        # Test complete workflow
        analyzer = create_diagnostic_analyzer()
        plots = create_diagnostic_plots()
        
        # Analyze quality
        analysis = analyzer.analyze_forecast_quality(forecast_result, actual_data, 'date', 'value')
        assert 'quality_score' in analysis
        
        # Create diagnostic report
        report = plots.create_comprehensive_diagnostic_report(forecast_result, actual_data, 'date', 'value')
        assert len(report) > 0
        
        # Verify all plots are created
        for plot_name, figure in report.items():
            assert isinstance(figure, go.Figure)
            if plot_name != 'error':  # Skip error plots
                assert len(figure.data) > 0

@pytest.mark.performance
class TestDiagnosticPerformance:
    """Performance tests for diagnostic module"""
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset"""
        # Create large dataset
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        forecast_data = pd.DataFrame({
            'ds': dates,
            'yhat': np.random.normal(100, 10, 1000),
            'yhat_lower': np.random.normal(90, 8, 1000),
            'yhat_upper': np.random.normal(110, 8, 1000),
            'trend': np.linspace(95, 105, 1000)
        })
        
        mock_model = Mock()
        mock_model.changepoints = pd.date_range('2020-01-10', periods=10, freq='30D')
        
        forecast_result = ProphetForecastResult(
            success=True,
            error=None,
            model=mock_model,
            raw_forecast=forecast_data,
            metrics={'mae': 5.0, 'mape': 10.0, 'rmse': 7.0}
        )
        
        actual_data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=800, freq='D'),
            'value': np.random.normal(100, 10, 800)
        })
        
        analyzer = create_diagnostic_analyzer()
        
        import time
        start_time = time.time()
        analysis = analyzer.analyze_forecast_quality(forecast_result, actual_data, 'date', 'value')
        end_time = time.time()
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert end_time - start_time < 10  # seconds
        assert 'quality_score' in analysis
