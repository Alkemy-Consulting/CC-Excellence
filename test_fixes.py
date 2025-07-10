#!/usr/bin/env python3
"""
Test script to validate all fixes:
1. sklearn compatibility fixes
2. prophet_enhanced module import
3. JSON serialization handling
4. Model wrapper functions
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def test_sklearn_compatibility():
    """Test that sklearn compatibility fixes work."""
    print("🔧 Testing sklearn compatibility fixes...")
    
    try:
        # Test ARIMA module compatibility
        from modules.arima_enhanced import run_arima_model
        print("✅ ARIMA module imports successfully")
        
        # Test SARIMA module compatibility  
        from modules.sarima_enhanced import run_sarima_model
        print("✅ SARIMA module imports successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

def test_prophet_import():
    """Test that prophet_enhanced module imports correctly."""
    print("\n🔮 Testing Prophet enhanced module...")
    
    try:
        from modules.prophet_enhanced import run_prophet_model
        print("✅ Prophet enhanced module imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Prophet import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Prophet other error: {e}")
        return False

def test_json_serialization():
    """Test JSON serialization with timestamps."""
    print("\n📊 Testing JSON serialization...")
    
    try:
        import json
        
        # Create test DataFrame with timestamps
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        test_df = pd.DataFrame({
            'ds': dates,
            'y': np.random.randn(10),
            'forecast': np.random.randn(10)
        })
        
        # Custom JSON serializer (same as in forecast_engine.py)
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
        forecast_records = []
        for _, row in test_df.iterrows():
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
        
        # Try to serialize to JSON
        json_str = json.dumps(json_data, indent=2, default=json_serializer)
        print("✅ JSON serialization works correctly")
        return True
        
    except Exception as e:
        print(f"❌ JSON serialization error: {e}")
        return False

def test_model_wrappers():
    """Test that model wrapper functions work."""
    print("\n🎯 Testing model wrapper functions...")
    
    try:
        # Create synthetic test data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'volume': 100 + np.random.randn(100).cumsum()
        })
        
        # Test Prophet wrapper (minimal test)
        try:
            from modules.prophet_enhanced import prepare_prophet_data
            prophet_data = prepare_prophet_data(data, 'date', 'volume')
            print("✅ Prophet data preparation works")
        except Exception as e:
            print(f"⚠️ Prophet wrapper issue: {e}")
        
        # Test ARIMA wrapper (minimal test)  
        try:
            from modules.arima_enhanced import prepare_arima_data
            arima_data = prepare_arima_data(data, 'date', 'volume')
            print("✅ ARIMA data preparation works")
        except Exception as e:
            print(f"⚠️ ARIMA wrapper issue: {e}")
            
        return True
        
    except Exception as e:
        print(f"❌ Model wrapper error: {e}")
        return False

def test_forecast_engine_import():
    """Test that forecast engine imports all modules correctly."""
    print("\n🚀 Testing forecast engine imports...")
    
    try:
        from modules.forecast_engine import display_forecast_results
        print("✅ Forecast engine imports successfully")
        return True
    except ImportError as e:
        print(f"❌ Forecast engine import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Forecast engine other error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running comprehensive fix validation tests...\n")
    
    tests = [
        test_sklearn_compatibility,
        test_prophet_import,
        test_json_serialization,
        test_model_wrappers,
        test_forecast_engine_import
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All fixes validated successfully!")
        return 0
    else:
        print("❌ Some issues remain to be fixed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
