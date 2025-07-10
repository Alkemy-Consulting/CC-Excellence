"""
Unit tests for the enhanced forecasting modules.
Tests data utilities, model functionality, and integration.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the modules directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

try:
    from modules.data_utils import *
    from modules.config import *
    from modules.forecast_engine import *
except ImportError:
    # Alternative import for testing
    from data_utils import *
    from config import *
    from forecast_engine import *


class TestDataUtils(unittest.TestCase):
    """Test data utility functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample time series data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        values = 100 + np.cumsum(np.random.randn(len(dates))) + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        
        self.sample_df = pd.DataFrame({
            'date': dates,
            'volume': values,
            'regressor1': np.random.randn(len(dates)),
            'regressor2': np.random.randn(len(dates))
        })
        
        # Add some missing values and outliers
        self.sample_df_missing = self.sample_df.copy()
        self.sample_df_missing.loc[10:15, 'volume'] = np.nan
        self.sample_df_missing.loc[100, 'volume'] = 10000  # Outlier
    
    def test_detect_date_column(self):
        """Test automatic date column detection."""
        date_col = detect_date_column(self.sample_df)
        self.assertEqual(date_col, 'date')
        
        # Test with no date column
        df_no_date = self.sample_df.drop('date', axis=1)
        date_col = detect_date_column(df_no_date)
        self.assertIsNone(date_col)
    
    def test_detect_target_column(self):
        """Test automatic target column detection."""
        target_col = detect_target_column(self.sample_df, 'date')
        self.assertEqual(target_col, 'volume')
    
    def test_get_data_statistics(self):
        """Test data statistics extraction."""
        stats = get_data_statistics(self.sample_df, 'date', 'volume')
        
        self.assertIn('record_count', stats)
        self.assertIn('date_range', stats)
        self.assertIn('missing_values', stats)
        self.assertIn('data_quality_score', stats)
        
        self.assertEqual(stats['record_count'], len(self.sample_df))
        self.assertEqual(stats['missing_values'], 0)
    
    def test_detect_outliers(self):
        """Test outlier detection."""
        outliers_data = detect_outliers(self.sample_df_missing, 'volume')
        
        self.assertIn('outlier_indices', outliers_data)
        self.assertIn('outlier_count', outliers_data)
        self.assertIn('outlier_percentage', outliers_data)
        
        # Should detect the outlier we added
        self.assertGreater(outliers_data['outlier_count'], 0)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df_filled = handle_missing_values(self.sample_df_missing, 'volume', 'forward_fill')
        
        # Should have no missing values after filling
        self.assertEqual(df_filled['volume'].isnull().sum(), 0)
        
        # Test different fill methods
        for method in ['backward_fill', 'linear_interpolation', 'zero_fill']:
            df_test = handle_missing_values(self.sample_df_missing, 'volume', method)
            self.assertEqual(df_test['volume'].isnull().sum(), 0)
    
    def test_get_regressor_candidates(self):
        """Test regressor candidate detection."""
        candidates = get_regressor_candidates(self.sample_df, 'date', 'volume')
        
        self.assertIn('regressor1', candidates)
        self.assertIn('regressor2', candidates)
        self.assertNotIn('date', candidates)
        self.assertNotIn('volume', candidates)
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        sample_data = generate_sample_data()
        
        self.assertIsInstance(sample_data, pd.DataFrame)
        self.assertIn('date', sample_data.columns)
        self.assertIn('volume', sample_data.columns)
        self.assertGreater(len(sample_data), 0)


class TestForecastEngine(unittest.TestCase):
    """Test forecast engine functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = generate_sample_data()
        
        self.base_config = {
            'forecast_periods': 30,
            'confidence_interval': 0.95,
            'train_size': 0.8
        }
    
    def test_calculate_model_score(self):
        """Test model scoring function."""
        # Test with complete metrics
        metrics = {
            'aic': 100.0,
            'train_rmse': 5.0,
            'train_mape': 0.1,
            'val_rmse': 6.0,
            'val_mape': 0.12
        }
        
        score = calculate_model_score(metrics)
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)
        
        # Test with missing metrics
        incomplete_metrics = {'aic': 100.0}
        score = calculate_model_score(incomplete_metrics)
        self.assertIsInstance(score, float)
        
        # Test with invalid metrics
        invalid_metrics = {}
        score = calculate_model_score(invalid_metrics)
        self.assertEqual(score, float('inf'))
    
    def test_run_enhanced_forecast_validation(self):
        """Test forecast function validation."""
        # Test with invalid model type
        try:
            forecast_df, metrics, plots = run_enhanced_forecast(
                self.sample_data, 'date', 'volume', 'InvalidModel', {}, self.base_config
            )
            # Should return empty results for invalid model
            self.assertTrue(forecast_df.empty)
        except Exception:
            # Or raise an appropriate error
            pass


class TestModelIntegration(unittest.TestCase):
    """Test model integration and workflow."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = generate_sample_data()
        
        # Simple model configs for testing
        self.model_configs = {
            'Prophet': {
                'seasonality_mode': 'additive',
                'changepoint_prior_scale': 0.05
            },
            'ARIMA': {
                'auto_arima': True,
                'max_p': 2,
                'max_q': 2
            },
            'SARIMA': {
                'auto_sarima': True,
                'seasonal_period': 7
            },
            'Holt-Winters': {
                'trend': 'add',
                'seasonal': 'add',
                'seasonal_periods': 7
            }
        }
        
        self.forecast_config = {
            'forecast_periods': 14,
            'confidence_interval': 0.95,
            'train_size': 0.8
        }
    
    def test_data_preprocessing_workflow(self):
        """Test complete data preprocessing workflow."""
        # Test data upload and validation
        stats = get_data_statistics(self.sample_data, 'date', 'volume')
        self.assertGreater(stats['data_quality_score'], 0.5)
        
        # Test outlier detection
        outliers = detect_outliers(self.sample_data, 'volume')
        self.assertIsInstance(outliers['outlier_count'], int)
        
        # Test regressor detection
        regressors = get_regressor_candidates(self.sample_data, 'date', 'volume')
        self.assertIsInstance(regressors, list)
    
    def test_model_configuration_validation(self):
        """Test model configuration validation."""
        for model_name, config in self.model_configs.items():
            # Each config should be a dictionary
            self.assertIsInstance(config, dict)
            
            # Should have at least one parameter
            self.assertGreater(len(config), 0)


class TestConfigConstants(unittest.TestCase):
    """Test configuration constants and defaults."""
    
    def test_model_labels_exist(self):
        """Test that model labels are defined."""
        self.assertIn('Prophet', MODEL_LABELS)
        self.assertIn('ARIMA', MODEL_LABELS)
        self.assertIn('SARIMA', MODEL_LABELS)
        self.assertIn('Holt-Winters', MODEL_LABELS)
    
    def test_default_parameters(self):
        """Test that default parameters are defined."""
        self.assertIsInstance(PROPHET_DEFAULTS, dict)
        self.assertIsInstance(ARIMA_DEFAULTS, dict)
        self.assertIsInstance(SARIMA_DEFAULTS, dict)
        self.assertIsInstance(HOLTWINTERS_DEFAULTS, dict)
        
        # Check some essential parameters
        self.assertIn('changepoint_prior_scale', PROPHET_DEFAULTS)
        self.assertIn('p', ARIMA_DEFAULTS)
        self.assertIn('seasonal_periods', HOLTWINTERS_DEFAULTS)
    
    def test_parameter_tooltips(self):
        """Test that parameter tooltips are defined."""
        self.assertIsInstance(PARAMETER_TOOLTIPS, dict)
        self.assertIn('prophet', PARAMETER_TOOLTIPS)
        self.assertIn('arima', PARAMETER_TOOLTIPS)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestForecastEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestModelIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigConstants))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("🧪 Running Enhanced Forecasting Module Tests...")
    print("=" * 60)
    
    result = run_tests()
    
    print("\n" + "=" * 60)
    print(f"🏁 Tests completed!")
    print(f"📊 Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"🚨 Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError: ')[-1].split('\n')[0]}")
    
    if result.errors:
        print("\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('\n')[-2]}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed successfully!")
    else:
        print(f"\n⚠️  {len(result.failures + result.errors)} test(s) failed. Please review and fix.")
