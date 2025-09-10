#!/usr/bin/env python3
"""
Test SARIMA compatibility fix
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspaces/CC-Excellence')
sys.path.append('/workspaces/CC-Excellence/modules')

def test_sarima_import():
    print("🧪 Testing SARIMA import fix...")
    
    try:
        from modules.sarima_enhanced import run_sarima_forecast, SARIMA_AVAILABLE
        print(f"✅ SARIMA module imported successfully!")
        print(f"   SARIMA available: {SARIMA_AVAILABLE}")
        
        if SARIMA_AVAILABLE:
            print("✅ All SARIMA dependencies are working!")
        else:
            print("⚠️ SARIMA dependencies have issues but import succeeded")
        
        return True
        
    except Exception as e:
        print(f"❌ SARIMA import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_model_imports():
    print("\n🔧 Testing all enhanced model imports...")
    
    success_count = 0
    total_tests = 4
    
    # Test Prophet
    try:
        from modules.prophet_enhanced import run_prophet_model
        print("✅ Prophet enhanced imported")
        success_count += 1
    except Exception as e:
        print(f"❌ Prophet error: {e}")
    
    # Test ARIMA
    try:
        from modules.arima_enhanced import run_arima_model, STATSMODELS_AVAILABLE
        print(f"✅ ARIMA enhanced imported (statsmodels: {STATSMODELS_AVAILABLE})")
        success_count += 1
    except Exception as e:
        print(f"❌ ARIMA error: {e}")
    
    # Test SARIMA
    try:
        from modules.sarima_enhanced import run_sarima_forecast, SARIMA_AVAILABLE
        print(f"✅ SARIMA enhanced imported (available: {SARIMA_AVAILABLE})")
        success_count += 1
    except Exception as e:
        print(f"❌ SARIMA error: {e}")
    
    # Test Holt-Winters
    try:
        from modules.holtwinters_module import run_holtwinters_forecast
        print("✅ Holt-Winters enhanced imported")
        success_count += 1
    except Exception as e:
        print(f"❌ Holt-Winters error: {e}")
    
    return success_count, total_tests

def test_forecast_engine():
    print("\n🚀 Testing forecast engine with fixed imports...")
    
    try:
        from modules.forecast_engine import run_enhanced_forecast, run_auto_select_forecast
        print("✅ Forecast engine imported successfully!")
        
        # Test that wrapper functions are available
        from modules.forecast_engine import run_prophet_forecast, run_arima_forecast
        print("✅ Wrapper functions available!")
        
        return True
        
    except Exception as e:
        print(f"❌ Forecast engine error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing SARIMA Compatibility Fix")
    print("=" * 50)
    
    # Test individual imports
    sarima_ok = test_sarima_import()
    success, total = test_all_model_imports()
    engine_ok = test_forecast_engine()
    
    print("\n" + "=" * 50)
    print(f"📊 Results: {success}/{total} models imported successfully")
    
    if sarima_ok and engine_ok and success == total:
        print("🎉 All compatibility issues fixed! Ready to use.")
    elif sarima_ok and engine_ok:
        print("✅ Core functionality working. Some model issues may exist.")
    else:
        print("⚠️ Some compatibility issues remain.")
        
    print("\nNext step: Try running the Streamlit app!")
