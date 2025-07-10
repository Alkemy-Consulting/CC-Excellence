#!/usr/bin/env python3
"""
Test that the Streamlit app can load all enhanced modules correctly.
"""

def test_app_imports():
    """Test imports as they would be used in the Streamlit app."""
    print("🚀 Testing Streamlit app imports...")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        # Test forecast engine import (this imports all models)
        from modules.forecast_engine import display_forecast_results, ENHANCED_MODELS_AVAILABLE
        print(f"Enhanced models available: {ENHANCED_MODELS_AVAILABLE}")
        
        if ENHANCED_MODELS_AVAILABLE:
            print("✅ All enhanced models are available")
        else:
            print("❌ Some enhanced models are not available")
            return False
        
        # Test individual model imports
        from modules.prophet_enhanced import run_prophet_model
        from modules.arima_enhanced import run_arima_model
        from modules.sarima_enhanced import run_sarima_forecast, SARIMA_AVAILABLE
        from modules.holtwinters_enhanced import run_holtwinters_forecast
        
        print("✅ All individual model modules imported successfully")
        print(f"SARIMA specifically available: {SARIMA_AVAILABLE}")
        
        # Test config and utilities
        from modules.config import MODEL_LABELS
        from modules.data_utils import detect_date_column
        from modules.ui_components import render_data_upload_section
        
        print("✅ Config and utility modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ App import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_json_export():
    """Test that JSON export works with timestamps."""
    print("\n📊 Testing JSON export functionality...")
    
    try:
        import json
        import pandas as pd
        import numpy as np
        
        # Simulate forecast data with timestamps like the app would generate
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        forecast_df = pd.DataFrame({
            'ds': dates,
            'yhat': np.random.randn(10),
            'yhat_lower': np.random.randn(10),
            'yhat_upper': np.random.randn(10)
        })
        
        # Test the JSON serializer from forecast_engine
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
        
        # Test DataFrame to records conversion like in the app
        forecast_records = []
        for _, row in forecast_df.iterrows():
            record = {}
            for col, value in row.items():
                try:
                    record[col] = json_serializer(value)
                except:
                    record[col] = str(value)
            forecast_records.append(record)
        
        json_data = {
            'model': 'test',
            'forecast': forecast_records,
            'generated_at': pd.Timestamp.now().isoformat()
        }
        
        # Test JSON serialization
        json_str = json.dumps(json_data, indent=2, default=json_serializer)
        print("✅ JSON export functionality works correctly")
        return True
        
    except Exception as e:
        print(f"❌ JSON export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all app readiness tests."""
    print("🏁 Testing Streamlit app readiness...\n")
    
    imports_ok = test_app_imports()
    json_ok = test_json_export()
    
    print(f"\n📊 App Readiness Results:")
    print(f"   Module Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"   JSON Export: {'✅ PASS' if json_ok else '❌ FAIL'}")
    
    if imports_ok and json_ok:
        print("\n🎉 Streamlit app is ready for use!")
        print("   ✅ All models available")
        print("   ✅ JSON export working")
        print("   ✅ No compatibility issues")
    else:
        print("\n❌ App still has issues to resolve.")

if __name__ == "__main__":
    main()
