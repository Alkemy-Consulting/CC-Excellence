#!/usr/bin/env python3
"""
Test ARIMA iloc fix.
"""

def test_arima_iloc_fix():
    """Test that ARIMA function works with the iloc fix."""
    print("🔧 Testing ARIMA iloc fix...")
    
    try:
        import sys
        import os
        import pandas as pd
        import numpy as np
        sys.path.insert(0, os.getcwd())
        
        from modules.arima_enhanced import run_arima_forecast
        
        # Create simple test data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': [100 + i + np.random.normal(0, 2) for i in range(30)]
        })
        
        # Test parameters
        model_config = {
            'p': 1, 'd': 1, 'q': 1,
            'auto_arima': False,
            'max_p': 2, 'max_d': 1, 'max_q': 2
        }
        
        base_config = {
            'forecast_periods': 5
        }
        
        # Test the function call
        forecast_df, metrics, plots = run_arima_forecast(
            data, 'date', 'value', model_config, base_config
        )
        
        print("✅ ARIMA function called successfully")
        print(f"Forecast shape: {forecast_df.shape}")
        print(f"Forecast columns: {list(forecast_df.columns)}")
        print(f"Metrics keys: {list(metrics.keys())}")
        
        # Check if we have the expected columns
        expected_cols = ['date', 'forecast', 'lower_bound', 'upper_bound']
        has_all_cols = all(col in forecast_df.columns for col in expected_cols)
        
        if has_all_cols and not forecast_df.empty:
            print("✅ ARIMA iloc fix successful!")
            return True
        else:
            print("⚠️ ARIMA returned unexpected results")
            return False
        
    except Exception as e:
        print(f"❌ ARIMA iloc fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing ARIMA iloc fix...\n")
    
    arima_ok = test_arima_iloc_fix()
    
    print(f"\n📊 Results:")
    print(f"   ARIMA iloc fix: {'✅ PASS' if arima_ok else '❌ FAIL'}")
    
    if arima_ok:
        print("\n🎉 ARIMA iloc fix validated successfully!")
    else:
        print("\n❌ ARIMA iloc issue remains.")
