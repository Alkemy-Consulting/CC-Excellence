#!/usr/bin/env python3
"""
Test both SARIMA and HoltWinters fixes.
"""

def test_sarima_module():
    """Test that SARIMA module loads correctly."""
    print("🔧 Testing SARIMA module...")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        from modules.sarima_enhanced import run_sarima_forecast, SARIMA_AVAILABLE
        
        print(f"SARIMA_AVAILABLE: {SARIMA_AVAILABLE}")
        
        if SARIMA_AVAILABLE:
            print("✅ SARIMA module loaded successfully")
            return True
        else:
            print("❌ SARIMA marked as unavailable")
            return False
            
    except Exception as e:
        print(f"❌ SARIMA module failed to load: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_holtwinters_module():
    """Test that HoltWinters module loads correctly."""
    print("\n🔧 Testing HoltWinters module...")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        from modules.holtwinters_enhanced import run_holtwinters_forecast
        print("✅ HoltWinters module loaded successfully")
        
        # Test a simple forecast to see if get_prediction issue is fixed
        import pandas as pd
        import numpy as np
        
        # Create simple test data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'value': 100 + np.random.randn(50).cumsum()
        })
        
        config = {
            'date_column': 'date',
            'target_column': 'value',
            'seasonal_periods': 7,
            'trend': 'additive',
            'seasonal': 'additive',
            'forecast_periods': 7,
            'confidence_interval': 0.95
        }
        
        # This should not raise the get_prediction error anymore
        forecast_df, metrics, plots = run_holtwinters_forecast(data, config)
        
        if not forecast_df.empty:
            print("✅ HoltWinters forecast generated successfully")
            return True
        else:
            print("⚠️ HoltWinters forecast returned empty DataFrame")
            return False
            
    except Exception as e:
        print(f"❌ HoltWinters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing SARIMA and HoltWinters fixes...\n")
    
    sarima_ok = test_sarima_module()
    holtwinters_ok = test_holtwinters_module()
    
    print(f"\n📊 Results:")
    print(f"   SARIMA: {'✅ PASS' if sarima_ok else '❌ FAIL'}")
    print(f"   HoltWinters: {'✅ PASS' if holtwinters_ok else '❌ FAIL'}")
    
    if sarima_ok and holtwinters_ok:
        print("\n🎉 All fixes validated successfully!")
    else:
        print("\n❌ Some issues remain.")

if __name__ == "__main__":
    main()
