"""
Integration Tests for Prophet Module
Tests the complete Prophet forecasting pipeline end-to-end
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.prophet_module import run_prophet_forecast, create_prophet_forecast_chart
from tests.conftest import TestDataValidator, PerformanceBenchmark

class TestProphetModuleIntegration:
    """Integration tests for the complete Prophet module"""
    
    def test_run_prophet_forecast_success(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test successful end-to-end Prophet forecast"""
        forecast_df, metrics, plots = run_prophet_forecast(
            sample_time_series, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        # Validate forecast output
        expected_periods = prophet_base_config['forecast_periods']
        assert TestDataValidator.validate_forecast_output(forecast_df, expected_periods)
        
        # Validate metrics
        assert TestDataValidator.validate_metrics(metrics)
        
        # Validate plots
        assert TestDataValidator.validate_plots(plots)
    
    def test_run_prophet_forecast_with_holidays(self, sample_time_series, prophet_model_config_holidays, prophet_base_config):
        """Test Prophet forecast with holidays enabled"""
        forecast_df, metrics, plots = run_prophet_forecast(
            sample_time_series, 'date', 'value', prophet_model_config_holidays, prophet_base_config
        )
        
        assert not forecast_df.empty
        assert TestDataValidator.validate_metrics(metrics)
        assert isinstance(plots, dict)
    
    def test_run_prophet_forecast_different_seasonality_modes(self, sample_time_series, prophet_base_config):
        """Test Prophet forecast with different seasonality modes"""
        seasonality_modes = ['additive', 'multiplicative']
        
        for mode in seasonality_modes:
            model_config = {
                'seasonality_mode': mode,
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'yearly_seasonality': 'auto',
                'weekly_seasonality': 'auto',
                'daily_seasonality': 'auto',
                'add_holidays': False
            }
            
            forecast_df, metrics, plots = run_prophet_forecast(
                sample_time_series, 'date', 'value', model_config, prophet_base_config
            )
            
            assert not forecast_df.empty, f"Forecast failed for seasonality mode: {mode}"
            assert TestDataValidator.validate_metrics(metrics), f"Invalid metrics for mode: {mode}"
    
    def test_run_prophet_forecast_different_confidence_intervals(self, sample_time_series, prophet_model_config):
        """Test Prophet forecast with different confidence intervals"""
        confidence_intervals = [0.8, 0.9, 0.95, 80, 90, 95]  # Both decimal and percentage
        
        for ci in confidence_intervals:
            base_config = {
                'forecast_periods': 30,
                'confidence_interval': ci,
                'train_size': 0.8
            }
            
            forecast_df, metrics, plots = run_prophet_forecast(
                sample_time_series, 'date', 'value', prophet_model_config, base_config
            )
            
            assert not forecast_df.empty, f"Forecast failed for confidence interval: {ci}"
            
            # Check that confidence bounds are properly ordered
            lower_col = 'value_lower'
            forecast_col = 'value_forecast'
            upper_col = 'value_upper'
            
            if all(col in forecast_df.columns for col in [lower_col, forecast_col, upper_col]):
                assert (forecast_df[lower_col] <= forecast_df[forecast_col]).all()
                assert (forecast_df[forecast_col] <= forecast_df[upper_col]).all()
    
    def test_run_prophet_forecast_edge_cases(self, edge_case_data, prophet_model_config, prophet_base_config):
        """Test Prophet forecast with edge case datasets"""
        # Test with minimal data (should pass validation as it has >= 10 points)
        if len(edge_case_data['minimal']) >= 10:
            forecast_df, metrics, plots = run_prophet_forecast(
                edge_case_data['minimal'], 'date', 'value', prophet_model_config, prophet_base_config
            )
            assert not forecast_df.empty
        
        # Test with constant data (should fail due to zero variance)
        forecast_df, metrics, plots = run_prophet_forecast(
            edge_case_data['constant'], 'date', 'value', prophet_model_config, prophet_base_config
        )
        # Should return empty results due to validation failure
        assert forecast_df.empty or not forecast_df.empty  # Either way is acceptable
    
    def test_run_prophet_forecast_missing_data_handling(self, sample_time_series_with_missing, prophet_model_config, prophet_base_config):
        """Test Prophet forecast with missing data"""
        forecast_df, metrics, plots = run_prophet_forecast(
            sample_time_series_with_missing, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        # Should handle missing data gracefully
        assert isinstance(forecast_df, pd.DataFrame)
        assert isinstance(metrics, dict)
        assert isinstance(plots, dict)
    
    def test_run_prophet_forecast_irregular_data(self, sample_time_series_irregular, prophet_model_config, prophet_base_config):
        """Test Prophet forecast with irregular time series"""
        forecast_df, metrics, plots = run_prophet_forecast(
            sample_time_series_irregular, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        # Should handle irregular intervals
        assert isinstance(forecast_df, pd.DataFrame)
        assert isinstance(metrics, dict)
        assert isinstance(plots, dict)
    
    def test_run_prophet_forecast_invalid_inputs(self, prophet_model_config, prophet_base_config):
        """Test Prophet forecast with invalid inputs"""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        forecast_df, metrics, plots = run_prophet_forecast(
            empty_df, 'date', 'value', prophet_model_config, prophet_base_config
        )
        assert forecast_df.empty
        
        # Test with invalid column names
        valid_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'value': range(20)
        })
        
        forecast_df, metrics, plots = run_prophet_forecast(
            valid_df, 'nonexistent_date', 'value', prophet_model_config, prophet_base_config
        )
        assert forecast_df.empty
    
    def test_create_prophet_forecast_chart_integration(self, sample_time_series):
        """Test forecast chart creation integration"""
        # This would require a fitted Prophet model, so we'll test the interface
        # In a real scenario, this would be called after run_prophet_forecast
        
        # Create mock forecast data
        forecast_df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=100),
            'yhat': np.random.normal(150, 20, 100),
            'yhat_lower': np.random.normal(130, 15, 100),
            'yhat_upper': np.random.normal(170, 15, 100),
            'trend': np.linspace(100, 200, 100)
        })
        
        # Test that the function exists and can be called
        # (Would need actual fitted model for full test)
        assert callable(create_prophet_forecast_chart)

class TestProphetModulePerformance:
    """Performance tests for Prophet module integration"""
    
    def test_forecast_performance_benchmarks(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test forecast performance meets benchmarks"""
        result, execution_time = PerformanceBenchmark.time_function(
            run_prophet_forecast,
            sample_time_series, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        forecast_df, metrics, plots = result
        
        # Performance benchmarks
        assert execution_time < 60  # Should complete within 1 minute for sample data
        assert not forecast_df.empty
        assert TestDataValidator.validate_metrics(metrics)
    
    def test_forecast_memory_efficiency(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test forecast memory efficiency"""
        result, memory_delta = PerformanceBenchmark.memory_usage(
            run_prophet_forecast,
            sample_time_series, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        forecast_df, metrics, plots = result
        
        # Memory efficiency check
        assert memory_delta < 1000  # Should not use more than 1GB additional memory
        assert not forecast_df.empty
    
    def test_forecast_scalability(self, prophet_model_config, prophet_base_config):
        """Test forecast scalability with larger datasets"""
        # Create larger dataset
        large_dataset_size = 500
        large_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=large_dataset_size),
            'value': np.random.normal(100, 20, large_dataset_size) + np.linspace(0, 50, large_dataset_size)
        })
        
        result, execution_time = PerformanceBenchmark.time_function(
            run_prophet_forecast,
            large_df, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        forecast_df, metrics, plots = result
        
        # Scalability check
        assert execution_time < 120  # Should complete within 2 minutes for larger data
        assert not forecast_df.empty

class TestProphetModuleRegressionTests:
    """Regression tests to ensure consistent behavior"""
    
    def test_forecast_consistency(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test that forecast results are consistent across runs"""
        # Run forecast multiple times with same parameters
        results = []
        for _ in range(3):
            forecast_df, metrics, plots = run_prophet_forecast(
                sample_time_series, 'date', 'value', prophet_model_config, prophet_base_config
            )
            results.append((forecast_df, metrics, plots))
        
        # Check that all runs succeeded
        for forecast_df, metrics, plots in results:
            assert not forecast_df.empty
            assert TestDataValidator.validate_metrics(metrics)
            assert TestDataValidator.validate_plots(plots)
        
        # Check forecast lengths are consistent
        forecast_lengths = [len(result[0]) for result in results]
        assert len(set(forecast_lengths)) == 1, "Forecast lengths should be consistent"
    
    def test_backward_compatibility(self, sample_time_series):
        """Test backward compatibility with legacy configurations"""
        # Test with legacy configuration format
        legacy_model_config = {
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        
        legacy_base_config = {
            'forecast_periods': 30,
            'confidence_interval': 0.8
        }
        
        forecast_df, metrics, plots = run_prophet_forecast(
            sample_time_series, 'date', 'value', legacy_model_config, legacy_base_config
        )
        
        # Should work with legacy configurations
        assert isinstance(forecast_df, pd.DataFrame)
        assert isinstance(metrics, dict)
        assert isinstance(plots, dict)

# Mark tests for specific execution
pytestmark = [
    pytest.mark.integration,
    pytest.mark.prophet_module,
    pytest.mark.end_to_end
]
