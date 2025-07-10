#!/usr/bin/env python3
"""
Test the actual SARIMA forecasting function to see where the error occurs.
"""

def test_sarima_forecast_function():
    """Test the run_sarima_forecast function directly."""
    print("Testing run_sarima_forecast function...")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        import pandas as pd
        import numpy as np
        
        # Create test data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': 100 + np.random.randn(100).cumsum()
        })
        
        # Create config
        config = {
            'date_column': 'date',
            'target_column': 'value',
            'forecast_periods': 14,
            'confidence_interval': 0.95,
            'auto_tune': True,
            'train_size': 0.8
        }
        
        # Import SARIMA module and check availability
        from modules.sarima_enhanced import run_sarima_forecast, SARIMA_AVAILABLE
        print(f"SARIMA_AVAILABLE at import: {SARIMA_AVAILABLE}")
        
        if not SARIMA_AVAILABLE:
            print("❌ SARIMA_AVAILABLE is False - this is the problem!")
            return False
        
        # Try to run the forecast
        print("Attempting to run SARIMA forecast...")
        forecast_df, metrics, plots = run_sarima_forecast(data, config)
        
        if forecast_df.empty:
            print("❌ Forecast returned empty DataFrame")
            return False
        else:
            print(f"✅ Forecast generated successfully with {len(forecast_df)} periods")
            print(f"   Metrics: {list(metrics.keys())}")
            print(f"   Plots: {list(plots.keys())}")
            return True
            
    except Exception as e:
        print(f"❌ Error in SARIMA forecast test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sarima_available_consistency():
    """Test that SARIMA_AVAILABLE is consistent across imports."""
    print("\nTesting SARIMA_AVAILABLE consistency...")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        # Test multiple imports
        from modules.sarima_enhanced import SARIMA_AVAILABLE as sarima_avail_1
        print(f"First import: SARIMA_AVAILABLE = {sarima_avail_1}")
        
        # Import again
        from modules.sarima_enhanced import SARIMA_AVAILABLE as sarima_avail_2
        print(f"Second import: SARIMA_AVAILABLE = {sarima_avail_2}")
        
        # Test through forecast engine
        from modules.forecast_engine import ENHANCED_MODELS_AVAILABLE
        print(f"Forecast engine: ENHANCED_MODELS_AVAILABLE = {ENHANCED_MODELS_AVAILABLE}")
        
        # Check if they're consistent
        if sarima_avail_1 == sarima_avail_2 == True:
            print("✅ SARIMA_AVAILABLE is consistently True")
            return True
        else:
            print("❌ SARIMA_AVAILABLE is inconsistent or False")
            return False
            
    except Exception as e:
        print(f"❌ Error in consistency test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🔍 Debugging SARIMA availability issue...\n")
    
    consistency_ok = test_sarima_available_consistency()
    forecast_ok = test_sarima_forecast_function()
    
    print(f"\n📊 Results:")
    print(f"   Consistency test: {'✅ PASS' if consistency_ok else '❌ FAIL'}")
    print(f"   Forecast function test: {'✅ PASS' if forecast_ok else '❌ FAIL'}")
    
    if consistency_ok and forecast_ok:
        print("\n🎉 SARIMA is working correctly!")
    else:
        print("\n❌ Found the issue with SARIMA availability.")

if __name__ == "__main__":
    main()
