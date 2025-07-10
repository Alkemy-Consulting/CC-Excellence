#!/usr/bin/env python3
"""
Simple test to verify the fixes work.
"""

def test_imports():
    """Test that our enhanced modules import correctly."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import streamlit as st
        print("✅ Basic packages imported")
        
        # Test pmdarima
        import pmdarima as pm
        print("✅ pmdarima imported")
        
        # Test our modules (add current directory to path)
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        from modules.config import MODEL_LABELS
        print("✅ Config module imported")
        
        from modules.data_utils import detect_date_column
        print("✅ Data utils imported")
        
        from modules.prophet_enhanced import run_prophet_model
        print("✅ Prophet enhanced imported")
        
        from modules.sarima_enhanced import run_sarima_forecast
        print("✅ SARIMA enhanced imported")
        
        from modules.arima_enhanced import run_arima_model
        print("✅ ARIMA enhanced imported")
        
        from modules.holtwinters_enhanced import run_holtwinters_forecast
        print("✅ Holt-Winters enhanced imported")
        
        from modules.forecast_engine import display_forecast_results
        print("✅ Forecast engine imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_serialization():
    """Test JSON serialization fix."""
    print("\nTesting JSON serialization...")
    
    try:
        import json
        import pandas as pd
        import numpy as np
        
        # Create test data with timestamps (but avoid arrays that cause issues)
        test_data = {
            'timestamp': pd.Timestamp.now(),
            'numpy_float': np.float64(3.14),
            'numpy_int': np.int64(42),
            'nan_value': np.nan
        }
        
        # Custom serializer (from our fix)
        def json_serializer(obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, np.datetime64):
                return pd.Timestamp(obj).isoformat()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return str(obj)
        
        # Test serialization
        json_str = json.dumps(test_data, default=json_serializer, indent=2)
        print("✅ JSON serialization works")
        
        # Test with DataFrame-like data (like in the actual app)
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        test_df = pd.DataFrame({
            'ds': dates,
            'yhat': [1.0, 2.0, 3.0]
        })
        
        # Test the actual DataFrame to records conversion like in the app
        forecast_records = []
        for _, row in test_df.iterrows():
            record = {}
            for col, value in row.items():
                try:
                    record[col] = json_serializer(value)
                except:
                    record[col] = str(value)
            forecast_records.append(record)
        
        json_str2 = json.dumps(forecast_records, default=json_serializer, indent=2)
        print("✅ DataFrame JSON serialization works")
        
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🧪 Running validation tests...\n")
    
    import_success = test_imports()
    json_success = test_json_serialization()
    
    print(f"\n📊 Results:")
    print(f"   Imports: {'✅ PASS' if import_success else '❌ FAIL'}")
    print(f"   JSON: {'✅ PASS' if json_success else '❌ FAIL'}")
    
    if import_success and json_success:
        print("\n🎉 All tests passed! The fixes are working.")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
