#!/usr/bin/env python3
"""
Test the fixed wrapper functions for Prophet and ARIMA
"""

import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspaces/CC-Excellence')

def generate_test_data(days=100):
    """Generate simple test data for forecasting."""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Create a simple trend + seasonality + noise pattern
    trend = np.linspace(100, 150, days)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(days) / 7)  # Weekly pattern
    noise = np.random.normal(0, 5, days)
    
    values = trend + seasonality + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values
    })

def test_prophet_wrapper():
    """Test the Prophet wrapper function."""
    print("🔮 Testing Prophet wrapper function...")
    
    try:
        from modules.forecast_engine import run_prophet_forecast
        
        # Generate test data
        data = generate_test_data(90)
        
        # Test configuration
        config = {
            'date_column': 'date',
            'target_column': 'value',
            'forecast_periods': 14,
            'seasonality_mode': 'additive',
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
        
        # Run Prophet forecast
        forecast_df, metrics, plots = run_prophet_forecast(data, config)
        
        if not forecast_df.empty:
            print(f"✅ Prophet forecast successful!")
            print(f"   Forecast shape: {forecast_df.shape}")
            print(f"   Columns: {list(forecast_df.columns)}")
            print(f"   Metrics: {len(metrics)} available")
            print(f"   Plots: {len(plots)} available")
            
            # Check forecast values
            if 'yhat' in forecast_df.columns:
                forecast_mean = forecast_df['yhat'].mean()
                print(f"   Average forecast value: {forecast_mean:.2f}")
            
            return True
        else:
            print("❌ Prophet forecast returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ Prophet wrapper error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_arima_wrapper():
    """Test the ARIMA wrapper function."""
    print("\n📊 Testing ARIMA wrapper function...")
    
    try:
        from modules.forecast_engine import run_arima_forecast
        
        # Generate test data
        data = generate_test_data(90)
        
        # Test configuration
        config = {
            'date_column': 'date',
            'target_column': 'value',
            'forecast_periods': 14,
            'auto_order': True,
            'p': 1,
            'd': 1,
            'q': 1
        }
        
        # Run ARIMA forecast
        forecast_df, metrics, plots = run_arima_forecast(data, config)
        
        if not forecast_df.empty:
            print(f"✅ ARIMA forecast successful!")
            print(f"   Forecast shape: {forecast_df.shape}")
            print(f"   Columns: {list(forecast_df.columns)}")
            print(f"   Metrics: {len(metrics)} available")
            print(f"   Plots: {len(plots)} available")
            
            # Check metrics
            if 'aic' in metrics:
                print(f"   AIC: {metrics['aic']:.2f}")
            if 'order' in metrics:
                print(f"   ARIMA order: {metrics['order']}")
            
            return True
        else:
            print("❌ ARIMA forecast returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ ARIMA wrapper error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forecast_engine():
    """Test the main forecast engine with fixed wrappers."""
    print("\n🚀 Testing forecast engine with wrapper functions...")
    
    try:
        from modules.forecast_engine import run_enhanced_forecast
        
        # Generate test data
        data = generate_test_data(90)
        
        # Test Prophet through forecast engine
        print("\n   Testing Prophet through forecast engine...")
        prophet_result = run_enhanced_forecast(
            data=data,
            date_column='date',
            target_column='value',
            model_type='Prophet',
            model_config={'seasonality_mode': 'additive'},
            forecast_config={'forecast_periods': 14}
        )
        
        forecast_df, metrics, plots = prophet_result
        if not forecast_df.empty:
            print("   ✅ Prophet via forecast engine: SUCCESS")
        else:
            print("   ❌ Prophet via forecast engine: FAILED")
        
        # Test ARIMA through forecast engine
        print("\n   Testing ARIMA through forecast engine...")
        arima_result = run_enhanced_forecast(
            data=data,
            date_column='date',
            target_column='value',
            model_type='ARIMA',
            model_config={'auto_order': True},
            forecast_config={'forecast_periods': 14}
        )
        
        forecast_df, metrics, plots = arima_result
        if not forecast_df.empty:
            print("   ✅ ARIMA via forecast engine: SUCCESS")
            return True
        else:
            print("   ❌ ARIMA via forecast engine: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Forecast engine error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing Fixed Wrapper Functions")
    print("=" * 50)
    
    # Test individual wrappers
    prophet_ok = test_prophet_wrapper()
    arima_ok = test_arima_wrapper()
    
    # Test through forecast engine
    engine_ok = test_forecast_engine()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Prophet wrapper: {'✅ PASS' if prophet_ok else '❌ FAIL'}")
    print(f"   ARIMA wrapper: {'✅ PASS' if arima_ok else '❌ FAIL'}")
    print(f"   Forecast engine: {'✅ PASS' if engine_ok else '❌ FAIL'}")
    
    if prophet_ok and arima_ok and engine_ok:
        print("\n🎉 All wrapper functions are working!")
        print("The forecast engine should now work properly in Streamlit.")
    else:
        print("\n⚠️ Some issues remain. Check the errors above.")
        
    print("\nNext: Test in Streamlit app with `streamlit run app.py`")
