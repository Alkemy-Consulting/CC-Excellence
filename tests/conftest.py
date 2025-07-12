"""
Test Configuration and Fixtures
Enterprise testing setup for CC-Excellence forecasting modules
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Test Data Generation Constants
TEST_DATA_POINTS = 100
TEST_FORECAST_PERIODS = 30
SEED = 42

@pytest.fixture
def sample_time_series():
    """Generate synthetic time series data for testing"""
    np.random.seed(SEED)
    
    # Create date range
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(TEST_DATA_POINTS)]
    
    # Generate synthetic data with trend, seasonality, and noise
    trend = np.linspace(100, 200, TEST_DATA_POINTS)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(TEST_DATA_POINTS) / 365.25)
    noise = np.random.normal(0, 5, TEST_DATA_POINTS)
    
    values = trend + seasonality + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })

@pytest.fixture
def sample_time_series_with_missing():
    """Generate time series with missing values for robustness testing"""
    df = sample_time_series()
    
    # Introduce random missing values (5% of data)
    missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
    df.loc[missing_indices, 'value'] = np.nan
    
    return df

@pytest.fixture
def sample_time_series_irregular():
    """Generate irregular time series for edge case testing"""
    np.random.seed(SEED)
    
    # Create irregular date intervals
    base_date = datetime(2020, 1, 1)
    irregular_intervals = np.cumsum(np.random.randint(1, 5, 50))  # 1-4 day gaps
    dates = [base_date + timedelta(days=int(x)) for x in irregular_intervals]
    
    values = np.random.normal(100, 20, 50)
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })

@pytest.fixture
def prophet_model_config():
    """Standard Prophet model configuration for testing"""
    return {
        'seasonality_mode': 'additive',
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'add_holidays': False
    }

@pytest.fixture
def prophet_base_config():
    """Standard Prophet base configuration for testing"""
    return {
        'forecast_periods': TEST_FORECAST_PERIODS,
        'confidence_interval': 0.8,
        'train_size': 0.8
    }

@pytest.fixture
def prophet_model_config_holidays(prophet_model_config):
    """Prophet configuration with holidays enabled"""
    config = prophet_model_config.copy()
    config.update({
        'add_holidays': True,
        'holidays_country': 'US'
    })
    return config

@pytest.fixture
def edge_case_data():
    """Edge case datasets for robustness testing"""
    return {
        'minimal': pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'value': range(10)
        }),
        'constant': pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'value': [100] * 50
        }),
        'extreme_values': pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'value': [1e6, -1e6] * 25
        }),
        'single_point': pd.DataFrame({
            'date': [datetime(2023, 1, 1)],
            'value': [100]
        })
    }

class TestDataValidator:
    """Utility class for validating test data"""
    
    @staticmethod
    def validate_forecast_output(forecast_df: pd.DataFrame, expected_periods: int) -> bool:
        """Validate forecast output structure and content"""
        if forecast_df.empty:
            return False
            
        # Check required columns
        required_cols = ['date', 'value_forecast', 'value_lower', 'value_upper']
        if not all(col in forecast_df.columns for col in required_cols):
            return False
            
        # Check forecast length
        if len(forecast_df) < expected_periods:
            return False
            
        # Check for valid numeric values
        numeric_cols = ['value_forecast', 'value_lower', 'value_upper']
        for col in numeric_cols:
            if forecast_df[col].isna().any() or not np.isfinite(forecast_df[col]).all():
                return False
                
        # Check confidence interval logic (lower <= forecast <= upper)
        valid_intervals = (
            (forecast_df['value_lower'] <= forecast_df['value_forecast']).all() and
            (forecast_df['value_forecast'] <= forecast_df['value_upper']).all()
        )
        
        return valid_intervals
    
    @staticmethod
    def validate_metrics(metrics: Dict) -> bool:
        """Validate metrics structure and values"""
        required_metrics = ['mape', 'mae', 'rmse', 'r2']
        
        if not all(metric in metrics for metric in required_metrics):
            return False
            
        # Check for valid numeric values
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                return False
                
        # Check reasonable ranges
        if metrics['mape'] < 0 or metrics['mae'] < 0 or metrics['rmse'] < 0:
            return False
            
        if metrics['r2'] < -1 or metrics['r2'] > 1:
            return False
            
        return True
    
    @staticmethod
    def validate_plots(plots: Dict) -> bool:
        """Validate plots structure"""
        if not isinstance(plots, dict):
            return False
            
        # Check for expected plot types
        expected_plots = ['forecast_plot']
        for plot_name in expected_plots:
            if plot_name in plots and plots[plot_name] is None:
                return False
                
        return True

# Performance benchmarking utilities
class PerformanceBenchmark:
    """Performance benchmarking utilities for testing"""
    
    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time function execution"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    @staticmethod
    def memory_usage(func, *args, **kwargs):
        """Measure memory usage of function"""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            # Get initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Get final memory
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_delta = final_memory - initial_memory
            
            return result, memory_delta
        except ImportError:
            # psutil not available, return 0 as fallback
            result = func(*args, **kwargs)
            return result, 0

# Test markers for pytest
pytestmark = [
    pytest.mark.forecasting,
    pytest.mark.prophet,
    pytest.mark.enterprise
]
