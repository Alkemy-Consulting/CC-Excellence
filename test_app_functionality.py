#!/usr/bin/env python3
"""
Comprehensive test script to validate CC-Excellence app functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append('/workspaces/CC-Excellence')

def test_imports():
    """Test all module imports"""
    print("🔍 Testing module imports...")
    
    try:
        # Core libraries
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Core libraries imported")
        
        # Project modules
        from modules.config import *
        print("✅ Config module imported")
        
        from modules.data_utils import *
        print("✅ Data utils module imported")
        
        from modules.ui_components import *
        print("✅ UI components module imported")
        
        from modules.forecast_engine import *
        print("✅ Forecast engine module imported")
        
        # Enhanced model modules
        from modules.prophet_enhanced import ProphetEnhanced
        from modules.arima_enhanced import ARIMAEnhanced
        from modules.holtwinters_enhanced import HoltWintersEnhanced
        from modules.sarima_enhanced import SARIMAEnhanced
        print("✅ Enhanced model modules imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_generation():
    """Test data generation and processing"""
    print("\n📊 Testing data generation...")
    
    try:
        from modules.data_utils import generate_sample_data
        
        # Generate sample data
        df = generate_sample_data()
        print(f"✅ Sample data generated: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_model_initialization():
    """Test model initialization"""
    print("\n🤖 Testing model initialization...")
    
    try:
        from modules.prophet_enhanced import ProphetEnhanced
        from modules.arima_enhanced import ARIMAEnhanced
        from modules.holtwinters_enhanced import HoltWintersEnhanced
        from modules.sarima_enhanced import SARIMAEnhanced
        
        # Test Prophet
        prophet_model = ProphetEnhanced()
        print("✅ Prophet model initialized")
        
        # Test ARIMA
        arima_model = ARIMAEnhanced()
        print("✅ ARIMA model initialized")
        
        # Test Holt-Winters
        hw_model = HoltWintersEnhanced()
        print("✅ Holt-Winters model initialized")
        
        # Test SARIMA
        sarima_model = SARIMAEnhanced()
        print("✅ SARIMA model initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forecast_functionality(df):
    """Test basic forecasting functionality"""
    print("\n🔮 Testing forecast functionality...")
    
    if df is None:
        print("❌ No data available for testing")
        return False
    
    try:
        from modules.forecast_engine import run_enhanced_forecast
        from modules.config import DEFAULT_MODEL_CONFIGS
        
        # Prepare data
        df_prepared = df.copy()
        df_prepared['ds'] = df_prepared['date']
        df_prepared['y'] = df_prepared['value']
        
        # Test Prophet forecast
        prophet_config = DEFAULT_MODEL_CONFIGS['Prophet'].copy()
        prophet_config['periods'] = 30  # 30 days forecast
        
        result = run_enhanced_forecast(
            df=df_prepared,
            model_type='Prophet',
            config=prophet_config
        )
        
        if result and 'forecast' in result:
            print("✅ Prophet forecast successful")
            print(f"   Forecast shape: {result['forecast'].shape}")
        else:
            print("❌ Prophet forecast failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Forecast error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing():
    """Test data processing utilities"""
    print("\n🛠️ Testing data processing utilities...")
    
    try:
        from modules.data_utils import (
            detect_date_columns, 
            detect_numeric_columns,
            calculate_data_statistics,
            handle_missing_values,
            detect_outliers
        )
        
        # Create test data with issues
        test_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'value': np.random.normal(100, 20, 100),
            'text_col': ['A'] * 100
        })
        
        # Add some missing values and outliers
        test_data.loc[10:15, 'value'] = np.nan
        test_data.loc[50, 'value'] = 1000  # outlier
        
        # Test date column detection
        date_cols = detect_date_columns(test_data)
        print(f"✅ Date columns detected: {date_cols}")
        
        # Test numeric column detection
        numeric_cols = detect_numeric_columns(test_data)
        print(f"✅ Numeric columns detected: {numeric_cols}")
        
        # Test statistics calculation
        stats = calculate_data_statistics(test_data, 'date', 'value')
        print(f"✅ Statistics calculated: {len(stats)} metrics")
        
        # Test missing value handling
        cleaned_data = handle_missing_values(test_data, 'value', 'interpolate')
        missing_after = cleaned_data['value'].isna().sum()
        print(f"✅ Missing values handled: {missing_after} remaining")
        
        # Test outlier detection
        outliers = detect_outliers(test_data, 'value')
        print(f"✅ Outliers detected: {len(outliers)} found")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting CC-Excellence App Functionality Test\n")
    print("=" * 60)
    
    # Run all tests
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Data generation
    df = test_data_generation()
    if df is not None:
        tests_passed += 1
    
    # Test 3: Model initialization
    if test_model_initialization():
        tests_passed += 1
    
    # Test 4: Data processing
    if test_data_processing():
        tests_passed += 1
    
    # Test 5: Forecast functionality
    if test_forecast_functionality(df):
        tests_passed += 1
    
    # Final report
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The app is ready to use.")
        return True
    else:
        print(f"⚠️ {total_tests - tests_passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
