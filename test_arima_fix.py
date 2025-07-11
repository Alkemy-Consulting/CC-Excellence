#!/usr/bin/env python3
"""
Test ARIMA function call fix.
"""

def test_arima_function_call():
    """Test that ARIMA function can be called correctly."""
    print("🔧 Testing ARIMA function call...")
    
    try:
        import sys
        import os
        import pandas as pd
        import numpy as np
        sys.path.insert(0, os.getcwd())
        
        from modules.arima_module import run_arima_model
        
        # Create simple test data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': 100 + np.random.randn(50).cumsum()
        })
        
        # Test parameters
        params = {
            'p': 1,
            'd': 1,
            'q': 1
        }
        
        # Test the function call with correct arguments
        result = run_arima_model(
            df=data,
            date_col='date',
            target_col='value',
            horizon=7,
            selected_metrics=['MAE', 'RMSE'],
            params=params,
            return_metrics=True
        )
        
        print("✅ ARIMA function called successfully")
        print(f"Result type: {type(result)}")
        return True
        
    except Exception as e:
        print(f"❌ ARIMA function call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the test."""
    print("🧪 Testing ARIMA function call fix...\n")
    
    arima_ok = test_arima_function_call()
    
    print(f"\n📊 Results:")
    print(f"   ARIMA Call: {'✅ PASS' if arima_ok else '❌ FAIL'}")
    
    if arima_ok:
        print("\n🎉 ARIMA fix validated successfully!")
    else:
        print("\n❌ ARIMA issue remains.")

if __name__ == "__main__":
    main()
