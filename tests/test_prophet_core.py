"""
Unit Tests for Prophet Core Business Logic
Tests the enterprise business logic layer in isolation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.prophet_core import ProphetForecaster, ProphetForecastResult
from tests.conftest import TestDataValidator, PerformanceBenchmark

class TestProphetForecaster:
    """Test suite for ProphetForecaster core business logic"""
    
    def test_forecaster_initialization(self):
        """Test ProphetForecaster initialization"""
        forecaster = ProphetForecaster()
        assert forecaster is not None
        assert hasattr(forecaster, 'run_forecast_core')
        assert hasattr(forecaster, 'validate_inputs')
    
    def test_input_validation_valid_data(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test input validation with valid data"""
        forecaster = ProphetForecaster()
        
        # Should not raise exception and return (True, None)
        is_valid, error_msg = forecaster.validate_inputs(
            sample_time_series, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        assert is_valid is True
        assert error_msg is None
    
    def test_input_validation_missing_columns(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test input validation with missing columns"""
        forecaster = ProphetForecaster()
        
        is_valid, error_msg = forecaster.validate_inputs(
            sample_time_series, 'nonexistent_date', 'value', prophet_model_config, prophet_base_config
        )
        assert is_valid is False
        assert "not found" in error_msg.lower()
        
        is_valid, error_msg = forecaster.validate_inputs(
            sample_time_series, 'date', 'nonexistent_value', prophet_model_config, prophet_base_config
        )
        assert is_valid is False
        assert "not found" in error_msg.lower()
    
    def test_input_validation_insufficient_data(self, prophet_model_config, prophet_base_config):
        """Test input validation with insufficient data"""
        forecaster = ProphetForecaster()
        
        # Create minimal dataset (less than 10 points)
        small_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'value': range(5)
        })
        
        is_valid, error_msg = forecaster.validate_inputs(
            small_df, 'date', 'value', prophet_model_config, prophet_base_config
        )
        assert is_valid is False
        assert "insufficient data" in error_msg.lower()
    
    def test_input_validation_too_many_missing_values(self, prophet_model_config, prophet_base_config):
        """Test input validation with too many missing values"""
        forecaster = ProphetForecaster()
        
        # Create dataset with >30% missing values
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'value': [np.nan] * 8 + list(range(12))  # 40% missing
        })
        
        is_valid, error_msg = forecaster.validate_inputs(
            df, 'date', 'value', prophet_model_config, prophet_base_config
        )
        assert is_valid is False
        assert "missing values" in error_msg.lower()
    
    def test_input_validation_non_numeric_values(self, prophet_model_config, prophet_base_config):
        """Test input validation with non-numeric target values"""
        forecaster = ProphetForecaster()
        
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'value': ['text'] * 20
        })
        
        is_valid, error_msg = forecaster.validate_inputs(
            df, 'date', 'value', prophet_model_config, prophet_base_config
        )
        assert is_valid is False
        assert "non-numeric" in error_msg.lower()
    
    def test_input_validation_zero_variance(self, prophet_model_config, prophet_base_config):
        """Test input validation with zero variance data"""
        forecaster = ProphetForecaster()
        
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=20),
            'value': [100] * 20  # All same value
        })
        
        is_valid, error_msg = forecaster.validate_inputs(
            df, 'date', 'value', prophet_model_config, prophet_base_config
        )
        assert is_valid is False
        assert "zero variance" in error_msg.lower()
    
    def test_optimize_dataframe(self, sample_time_series):
        """Test DataFrame optimization"""
        forecaster = ProphetForecaster()
        
        # Add extra columns to test optimization
        test_df = sample_time_series.copy()
        test_df['extra_col'] = range(len(test_df))
        
        optimized_df = forecaster.optimize_dataframe(test_df)
        
        # Should maintain data integrity
        assert len(optimized_df) == len(test_df)
        assert optimized_df.dtypes['date'] == 'datetime64[ns]'
        assert pd.api.types.is_numeric_dtype(optimized_df['value'])
    
    def test_prepare_data(self, sample_time_series):
        """Test data preparation for Prophet"""
        forecaster = ProphetForecaster()
        
        prepared_df = forecaster.prepare_data(sample_time_series, 'date', 'value')
        
        # Check Prophet format
        assert list(prepared_df.columns) == ['ds', 'y']
        assert len(prepared_df) <= len(sample_time_series)  # May be less due to dropna
        assert prepared_df['ds'].dtype == 'datetime64[ns]'
        assert pd.api.types.is_numeric_dtype(prepared_df['y'])
    
    def test_split_data(self, sample_time_series):
        """Test data splitting functionality"""
        forecaster = ProphetForecaster()
        
        prepared_df = forecaster.prepare_data(sample_time_series, 'date', 'value')
        train_df, test_df = forecaster.split_data(prepared_df, train_size=0.8)
        
        # Check split logic
        expected_split = int(len(prepared_df) * 0.8)
        assert len(test_df) == len(prepared_df) - expected_split
        assert len(train_df) == len(prepared_df)  # For forecasting, use all data for training
    
    def test_create_model_basic(self, prophet_model_config):
        """Test model creation with basic configuration"""
        forecaster = ProphetForecaster()
        
        model = forecaster.create_model(prophet_model_config)
        
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'make_future_dataframe')
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        forecaster = ProphetForecaster()
        
        # Create sample actual and predicted values
        actual = np.array([100, 110, 120, 130, 140])
        predicted = np.array([98, 112, 118, 132, 142])
        
        metrics = forecaster.calculate_metrics(actual, predicted)
        
        # Validate metrics structure
        assert TestDataValidator.validate_metrics(metrics)
        
        # Check specific calculations
        assert metrics['mae'] > 0
        assert metrics['rmse'] > 0
        assert 0 <= metrics['mape'] <= 100
        assert -1 <= metrics['r2'] <= 1
    
    def test_calculate_metrics_edge_cases(self):
        """Test metrics calculation with edge cases"""
        forecaster = ProphetForecaster()
        
        # Perfect predictions
        actual = np.array([100, 110, 120])
        predicted = np.array([100, 110, 120])
        metrics = forecaster.calculate_metrics(actual, predicted)
        
        assert metrics['mae'] == 0
        assert metrics['rmse'] == 0
        assert metrics['mape'] == 0
        assert metrics['r2'] == 1
        
        # Zero values (MAPE edge case)
        actual_with_zeros = np.array([0, 0, 100])
        predicted_with_zeros = np.array([1, 1, 100])
        metrics_zeros = forecaster.calculate_metrics(actual_with_zeros, predicted_with_zeros)
        
        assert TestDataValidator.validate_metrics(metrics_zeros)
    
    def test_run_forecast_core_success(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test successful end-to-end forecast execution"""
        forecaster = ProphetForecaster()
        
        result = forecaster.run_forecast_core(
            sample_time_series, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        # Validate result structure
        assert isinstance(result, ProphetForecastResult)
        assert result.success is True
        assert result.error is None
        assert result.model is not None
        assert result.raw_forecast is not None
        assert TestDataValidator.validate_metrics(result.metrics)
        
        # Validate forecast data
        expected_periods = prophet_base_config['forecast_periods']
        assert len(result.raw_forecast) >= expected_periods
        
        # Check forecast columns
        required_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
        assert all(col in result.raw_forecast.columns for col in required_cols)
    
    def test_run_forecast_core_with_holidays(self, sample_time_series, prophet_model_config_holidays, prophet_base_config):
        """Test forecast with holidays enabled"""
        forecaster = ProphetForecaster()
        
        result = forecaster.run_forecast_core(
            sample_time_series, 'date', 'value', prophet_model_config_holidays, prophet_base_config
        )
        
        assert result.success is True
        assert result.model is not None
    
    def test_run_forecast_core_failure_handling(self, prophet_model_config, prophet_base_config):
        """Test error handling in forecast execution"""
        forecaster = ProphetForecaster()
        
        # Use invalid data to trigger failure
        invalid_df = pd.DataFrame({
            'date': ['invalid_date'] * 20,
            'value': range(20)
        })
        
        result = forecaster.run_forecast_core(
            invalid_df, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        assert result.success is False
        assert result.error is not None
        assert result.model is None
        assert result.raw_forecast is None

class TestProphetForecastResult:
    """Test suite for ProphetForecastResult data class"""
    
    def test_forecast_result_creation_success(self):
        """Test successful forecast result creation"""
        result = ProphetForecastResult(
            success=True,
            error=None,
            model="mock_model",
            raw_forecast=pd.DataFrame(),
            metrics={'mape': 10.0, 'mae': 5.0, 'rmse': 7.0, 'r2': 0.85}
        )
        
        assert result.success is True
        assert result.error is None
        assert result.model == "mock_model"
        assert isinstance(result.raw_forecast, pd.DataFrame)
        assert isinstance(result.metrics, dict)
    
    def test_forecast_result_creation_failure(self):
        """Test failure forecast result creation"""
        result = ProphetForecastResult(
            success=False,
            error="Test error message",
            model=None,
            raw_forecast=None,
            metrics={}
        )
        
        assert result.success is False
        assert result.error == "Test error message"
        assert result.model is None
        assert result.raw_forecast is None
        assert result.metrics == {}

class TestProphetPerformance:
    """Performance tests for Prophet forecaster"""
    
    def test_forecast_performance_small_dataset(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test forecast performance with small dataset"""
        forecaster = ProphetForecaster()
        
        result, execution_time = PerformanceBenchmark.time_function(
            forecaster.run_forecast_core,
            sample_time_series, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        # Performance thresholds for small dataset
        assert execution_time < 30  # Should complete within 30 seconds
        assert result.success is True
    
    def test_forecast_memory_usage(self, sample_time_series, prophet_model_config, prophet_base_config):
        """Test forecast memory usage"""
        forecaster = ProphetForecaster()
        
        result, memory_delta = PerformanceBenchmark.memory_usage(
            forecaster.run_forecast_core,
            sample_time_series, 'date', 'value', prophet_model_config, prophet_base_config
        )
        
        # Memory usage should be reasonable (less than 500MB increase)
        assert memory_delta < 500
        assert result.success is True

# Mark tests for specific execution
pytestmark = [
    pytest.mark.unit,
    pytest.mark.prophet_core,
    pytest.mark.business_logic
]
