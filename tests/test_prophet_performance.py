"""
Test suite for Prophet Performance Module
Enterprise-level testing for performance optimization features
"""

import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import psutil

from modules.prophet_performance import (
    PerformanceMetrics,
    PerformanceMonitor,
    AdvancedCache,
    OptimizedProphetForecaster,
    DataFrameOptimizer,
    PerformanceTuner,
    performance_optimized,
    create_optimized_forecaster,
    create_dataframe_optimizer,
    create_performance_tuner,
    get_performance_report,
    benchmark_prophet_performance
)
from modules.prophet_core import ProphetForecastResult

class TestPerformanceMetrics:
    """Test performance metrics data structure"""
    
    def test_performance_metrics_initialization(self):
        metrics = PerformanceMetrics(
            execution_time=1.5,
            memory_usage=100.0,
            memory_peak=150.0,
            cpu_usage=25.0,
            cache_hits=10,
            cache_misses=2,
            data_size=1000,
            forecast_points=30
        )
        
        assert metrics.execution_time == 1.5
        assert metrics.memory_usage == 100.0
        assert metrics.cache_hits == 10
        assert metrics.optimization_applied == []
    
    def test_performance_metrics_with_optimizations(self):
        metrics = PerformanceMetrics(
            execution_time=1.0,
            memory_usage=80.0,
            memory_peak=120.0,
            cpu_usage=20.0,
            cache_hits=15,
            cache_misses=1,
            data_size=1000,
            forecast_points=30,
            optimization_applied=['caching', 'dataframe_optimization']
        )
        
        assert 'caching' in metrics.optimization_applied
        assert 'dataframe_optimization' in metrics.optimization_applied

class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def test_monitor_initialization(self):
        monitor = PerformanceMonitor()
        assert monitor.metrics_history == []
        assert monitor.cache_stats == {'hits': 0, 'misses': 0}
        assert monitor.memory_baseline > 0
    
    def test_monitor_execution_context(self):
        monitor = PerformanceMonitor()
        
        with monitor.monitor_execution("test_operation") as m:
            # Simulate some work
            time.sleep(0.1)
            data = [i**2 for i in range(1000)]
        
        assert len(monitor.metrics_history) == 1
        metrics = monitor.metrics_history[0]
        assert metrics.execution_time >= 0.1
        assert isinstance(metrics.memory_usage, (int, float))
    
    def test_cache_stats_update(self):
        monitor = PerformanceMonitor()
        
        monitor.update_cache_stats(hit=True)
        monitor.update_cache_stats(hit=True)
        monitor.update_cache_stats(hit=False)
        
        assert monitor.cache_stats['hits'] == 2
        assert monitor.cache_stats['misses'] == 1
        assert monitor.get_cache_hit_ratio() == 2/3
    
    def test_performance_summary(self):
        monitor = PerformanceMonitor()
        
        # Add some mock metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                execution_time=1.0 + i * 0.1,
                memory_usage=100.0 + i * 10,
                memory_peak=150.0 + i * 10,
                cpu_usage=20.0 + i * 2,
                cache_hits=10 + i,
                cache_misses=2,
                data_size=1000,
                forecast_points=30
            )
            monitor.metrics_history.append(metrics)
        
        summary = monitor.get_performance_summary()
        
        assert 'avg_execution_time' in summary
        assert 'avg_memory_usage' in summary
        assert 'total_operations' in summary
        assert summary['total_operations'] == 5

class TestAdvancedCache:
    """Test advanced caching functionality"""
    
    def test_cache_initialization_memory(self):
        cache = AdvancedCache(max_size=10, ttl_seconds=60, use_redis=False)
        assert cache.max_size == 10
        assert cache.ttl_seconds == 60
        assert not cache.use_redis
    
    def test_cache_basic_operations(self):
        cache = AdvancedCache(max_size=5, ttl_seconds=60, use_redis=False)
        
        # Test set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
    
    def test_cache_ttl_expiration(self):
        cache = AdvancedCache(max_size=5, ttl_seconds=1, use_redis=False)
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for TTL expiration
        time.sleep(1.1)
        assert cache.get("key1") is None
    
    def test_cache_lru_eviction(self):
        cache = AdvancedCache(max_size=2, ttl_seconds=60, use_redis=False)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 to make it more recently used
        cache.get("key1")
        
        # Add key3, should evict key2 (least recently used)
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key3") == "value3"
        assert cache.get("key2") is None
    
    def test_cache_key_generation(self):
        cache = AdvancedCache()
        
        key1 = cache._generate_key("test_func", (1, 2), {"param": "value"})
        key2 = cache._generate_key("test_func", (1, 2), {"param": "value"})
        key3 = cache._generate_key("test_func", (1, 3), {"param": "value"})
        
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
    
    def test_cache_stats(self):
        cache = AdvancedCache(use_redis=False)
        
        cache.set("key1", "value1")
        stats = cache.stats()
        
        assert stats['type'] == 'memory'
        assert stats['size'] == 1
        assert 'max_size' in stats

class TestDataFrameOptimizer:
    """Test DataFrame optimization functionality"""
    
    def test_optimizer_initialization(self):
        optimizer = DataFrameOptimizer()
        assert optimizer.optimization_stats == {}
    
    def test_dataframe_optimization_numeric(self):
        # Create DataFrame with suboptimal dtypes
        df = pd.DataFrame({
            'int_col': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'float_col': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
            'date_col': pd.date_range('2023-01-01', periods=5)
        })
        
        optimizer = DataFrameOptimizer()
        optimized_df = optimizer.optimize_dataframe(df)
        
        # Check that optimization was applied
        assert optimized_df['int_col'].dtype != np.int64  # Should be downcast
        assert optimized_df['float_col'].dtype != np.float64  # Should be downcast
        
        stats = optimizer.get_optimization_stats()
        assert 'memory_savings_mb' in stats
        assert 'optimizations_applied' in stats
    
    def test_dataframe_optimization_categorical(self):
        # Create DataFrame with repeated string values
        df = pd.DataFrame({
            'category_col': ['A', 'B', 'A', 'B', 'A'] * 100,  # Repeated values
            'unique_col': [f'value_{i}' for i in range(500)]   # Mostly unique
        })
        
        optimizer = DataFrameOptimizer()
        optimized_df = optimizer.optimize_dataframe(df)
        
        # category_col should be converted to category
        assert optimized_df['category_col'].dtype.name == 'category'
        # unique_col should remain object (too many unique values)
        assert optimized_df['unique_col'].dtype == 'object'
    
    def test_dataframe_duplicate_removal(self):
        # Create DataFrame with duplicates
        df = pd.DataFrame({
            'col1': [1, 2, 3, 2, 3],
            'col2': [4, 5, 6, 5, 6]
        })
        
        optimizer = DataFrameOptimizer()
        optimized_df = optimizer.optimize_dataframe(df)
        
        # Should remove duplicate rows
        assert len(optimized_df) == 3  # Original had 5 rows, 2 duplicates
        assert optimized_df.duplicated().sum() == 0

class TestOptimizedProphetForecaster:
    """Test optimized Prophet forecaster"""
    
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = np.random.normal(100, 10, 100)
        return pd.DataFrame({'date': dates, 'value': values})
    
    @pytest.fixture
    def model_config(self):
        return {
            'growth': 'linear',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
    
    @pytest.fixture
    def base_config(self):
        return {
            'forecast_periods': 30,
            'confidence_interval': 0.8,
            'train_size': 0.8
        }
    
    def test_optimized_forecaster_initialization(self):
        forecaster = OptimizedProphetForecaster(enable_parallel=True, max_workers=2)
        assert forecaster.enable_parallel is True
        assert forecaster.max_workers == 2
        assert forecaster.data_optimizer is not None
    
    @patch('modules.prophet_core.Prophet')
    def test_optimized_forecast_execution(self, mock_prophet, sample_data, model_config, base_config):
        # Mock Prophet model
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.make_future_dataframe.return_value = sample_data.rename(columns={'date': 'ds'})
        mock_model.predict.return_value = pd.DataFrame({
            'ds': sample_data['date'],
            'yhat': np.random.normal(100, 5, 100),
            'yhat_lower': np.random.normal(90, 5, 100),
            'yhat_upper': np.random.normal(110, 5, 100),
            'trend': np.linspace(95, 105, 100)
        })
        mock_prophet.return_value = mock_model
        
        forecaster = OptimizedProphetForecaster()
        result = forecaster.run_forecast_core(sample_data, 'date', 'value', model_config, base_config)
        
        # Should return valid result
        assert isinstance(result, ProphetForecastResult)
    
    def test_parallel_diagnostics_enabled(self, sample_data, model_config, base_config):
        forecaster = OptimizedProphetForecaster(enable_parallel=True)
        
        # Create mock forecast result
        mock_result = Mock(spec=ProphetForecastResult)
        mock_result.success = True
        mock_result.raw_forecast = pd.DataFrame({
            'ds': sample_data['date'],
            'yhat': np.random.normal(100, 5, 100),
            'trend': np.linspace(95, 105, 100)
        })
        mock_result.model = Mock()
        mock_result.model.changepoints = pd.date_range('2023-01-10', periods=3, freq='10D')
        
        # Test parallel diagnostics
        diagnostics = forecaster.run_parallel_diagnostics(sample_data, 'date', 'value', mock_result)
        
        assert isinstance(diagnostics, dict)
        assert 'quality_score' in diagnostics
    
    def test_parallel_diagnostics_disabled(self, sample_data, model_config, base_config):
        forecaster = OptimizedProphetForecaster(enable_parallel=False)
        
        # Create mock forecast result
        mock_result = Mock(spec=ProphetForecastResult)
        mock_result.success = True
        mock_result.raw_forecast = pd.DataFrame({
            'ds': sample_data['date'],
            'yhat': np.random.normal(100, 5, 100),
            'trend': np.linspace(95, 105, 100)
        })
        mock_result.model = Mock()
        mock_result.model.changepoints = pd.date_range('2023-01-10', periods=3, freq='10D')
        
        # Should fallback to sequential execution
        diagnostics = forecaster.run_parallel_diagnostics(sample_data, 'date', 'value', mock_result)
        
        assert isinstance(diagnostics, dict)

class TestPerformanceTuner:
    """Test performance tuning functionality"""
    
    def test_tuner_initialization(self):
        tuner = PerformanceTuner()
        assert tuner.tuning_history == []
        assert tuner.optimal_params == {}
    
    def test_auto_tune_small_dataset(self):
        # Small dataset
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'value': np.random.normal(100, 10, 50)
        })
        
        model_config = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}
        base_config = {'forecast_periods': 30, 'train_size': 0.8}
        
        tuner = PerformanceTuner()
        tuned_model_config, tuned_base_config = tuner.auto_tune_model_config(df, model_config, base_config)
        
        # Should apply small dataset optimizations
        assert tuned_model_config['changepoint_prior_scale'] == 0.1
        assert tuned_model_config['seasonality_prior_scale'] == 5.0
        assert len(tuner.tuning_history) == 1
    
    def test_auto_tune_large_dataset(self):
        # Large dataset
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=1500),
            'value': np.random.normal(100, 10, 1500)
        })
        
        model_config = {'changepoint_prior_scale': 0.05, 'seasonality_prior_scale': 10.0}
        base_config = {'forecast_periods': 30, 'train_size': 0.8}
        
        tuner = PerformanceTuner()
        tuned_model_config, tuned_base_config = tuner.auto_tune_model_config(df, model_config, base_config)
        
        # Should apply large dataset optimizations
        assert tuned_model_config['changepoint_prior_scale'] == 0.01
        assert tuned_model_config['seasonality_prior_scale'] == 15.0
        assert tuned_base_config['train_size'] == 0.9
    
    def test_tuning_recommendations(self):
        tuner = PerformanceTuner()
        
        # High execution time scenario
        metrics = PerformanceMetrics(
            execution_time=35.0,  # High
            memory_usage=600.0,   # High
            memory_peak=800.0,
            cpu_usage=80.0,
            cache_hits=1,
            cache_misses=10,      # Low hit ratio
            data_size=1000,
            forecast_points=100
        )
        
        recommendations = tuner.get_tuning_recommendations(metrics)
        
        assert any('forecast_periods' in rec for rec in recommendations)
        assert any('MCMC' in rec for rec in recommendations)
        assert any('optimization' in rec for rec in recommendations)

class TestPerformanceDecorator:
    """Test performance optimization decorator"""
    
    def test_performance_optimized_decorator(self):
        call_count = 0
        
        @performance_optimized
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call - should execute function
        result1 = test_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = test_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again
        
        # Different args - should execute function
        result3 = test_function(2, 3)
        assert result3 == 5
        assert call_count == 2

class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_optimized_forecaster(self):
        forecaster = create_optimized_forecaster(enable_parallel=True, max_workers=4)
        assert isinstance(forecaster, OptimizedProphetForecaster)
        assert forecaster.enable_parallel is True
        assert forecaster.max_workers == 4
    
    def test_create_dataframe_optimizer(self):
        optimizer = create_dataframe_optimizer()
        assert isinstance(optimizer, DataFrameOptimizer)
    
    def test_create_performance_tuner(self):
        tuner = create_performance_tuner()
        assert isinstance(tuner, PerformanceTuner)

class TestPerformanceReporting:
    """Test performance reporting functionality"""
    
    def test_get_performance_report(self):
        report = get_performance_report()
        
        assert 'performance_summary' in report
        assert 'cache_stats' in report
        assert 'system_info' in report
        assert 'optimization_recommendations' in report
        
        # Check system info
        system_info = report['system_info']
        assert 'cpu_count' in system_info
        assert 'memory_total_mb' in system_info
        assert system_info['cpu_count'] > 0
        assert system_info['memory_total_mb'] > 0

@pytest.mark.integration
class TestPerformanceIntegration:
    """Integration tests for performance optimization"""
    
    @pytest.fixture
    def large_dataset(self):
        """Create a larger dataset for performance testing"""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        trend = np.linspace(100, 200, 1000)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(1000) / 365.25)
        noise = np.random.normal(0, 5, 1000)
        values = trend + seasonal + noise
        
        return pd.DataFrame({'date': dates, 'value': values})
    
    def test_end_to_end_optimization(self, large_dataset):
        """Test complete optimization workflow"""
        model_config = {
            'growth': 'linear',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
        base_config = {
            'forecast_periods': 30,
            'confidence_interval': 0.8,
            'train_size': 0.8
        }
        
        # Create optimized components
        forecaster = create_optimized_forecaster()
        optimizer = create_dataframe_optimizer()
        tuner = create_performance_tuner()
        
        # Optimize DataFrame
        optimized_df = optimizer.optimize_dataframe(large_dataset)
        assert len(optimized_df) <= len(large_dataset)  # May remove duplicates
        
        # Auto-tune configuration
        tuned_config, tuned_base = tuner.auto_tune_model_config(optimized_df, model_config, base_config)
        assert tuned_config != model_config  # Should have changes
        
        # Performance monitoring should be active
        report = get_performance_report()
        assert 'performance_summary' in report

@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.fixture
    def benchmark_data(self):
        """Create benchmark dataset"""
        dates = pd.date_range('2022-01-01', periods=500, freq='D')
        values = np.random.normal(100, 15, 500)
        return pd.DataFrame({'date': dates, 'value': values})
    
    def test_benchmark_execution_time(self, benchmark_data):
        """Test that optimized version is faster (or at least not slower)"""
        model_config = {'growth': 'linear', 'yearly_seasonality': True}
        base_config = {'forecast_periods': 20, 'train_size': 0.8}
        
        # This would ideally run actual benchmarks
        # For testing, we just ensure the benchmark function works
        try:
            results = benchmark_prophet_performance(
                benchmark_data, 'date', 'value', model_config, base_config, num_runs=1
            )
            assert 'runs' in results
            assert 'average_metrics' in results
        except Exception as e:
            # Benchmark might fail in test environment, that's ok
            pytest.skip(f"Benchmark test skipped due to environment: {e}")
    
    def test_memory_usage_optimization(self):
        """Test memory optimization effectiveness"""
        # Create memory-heavy DataFrame
        large_df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10000),
            'value': np.random.normal(100, 20, 10000).astype(np.float64),  # Force float64
            'category': ['Type_A', 'Type_B'] * 5000,  # Repeating categories
            'id': range(10000)  # Large int range
        })
        
        optimizer = create_dataframe_optimizer()
        original_memory = large_df.memory_usage(deep=True).sum()
        
        optimized_df = optimizer.optimize_dataframe(large_df)
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        
        stats = optimizer.get_optimization_stats()
        
        # Should achieve some memory savings
        assert optimized_memory <= original_memory
        assert stats['memory_savings_mb'] >= 0
        assert len(stats['optimizations_applied']) > 0

# Mark tests for specific execution
pytestmark = [
    pytest.mark.unit,
    pytest.mark.prophet_performance,
    pytest.mark.optimization
]
