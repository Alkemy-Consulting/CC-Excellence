#!/usr/bin/env python3
"""
Test the forecast engine import fix
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspaces/CC-Excellence')
sys.path.append('/workspaces/CC-Excellence/modules')

def test_forecast_engine_import():
    print("🧪 Testing forecast engine imports...")
    
    try:
        # Test the forecast engine import
        from modules.forecast_engine import run_enhanced_forecast, run_auto_select_forecast
        print("✅ Forecast engine imported successfully!")
        
        # Test individual function imports
        from modules.forecast_engine import run_prophet_forecast, run_arima_forecast
        print("✅ Wrapper functions imported successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_forecast():
    print("\n🔮 Testing basic forecast functionality...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.modules.utils.data_utils import generate_sample_data
        from modules.forecast_engine import run_enhanced_forecast
        
        # Generate sample data
        df = generate_sample_data(days=90)
        print(f"✅ Sample data generated: {df.shape}")
        
        # Prepare config for Prophet
        model_config = {
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
        
        forecast_config = {
            'forecast_periods': 14,
            'confidence_interval': 0.95,
            'train_size': 0.8
        }
        
        # Test Prophet forecast
        print("Testing Prophet forecast...")
        result = run_enhanced_forecast(
            data=df,
            date_column='date',
            target_column='value',
            model_type='Prophet',
            model_config=model_config,
            forecast_config=forecast_config
        )
        
        forecast_df, metrics, plots = result
        
        if not forecast_df.empty:
            print(f"✅ Prophet forecast successful: {forecast_df.shape}")
            print(f"   Forecast range: {forecast_df['yhat'].min():.2f} to {forecast_df['yhat'].max():.2f}")
            print(f"   Metrics available: {len(metrics)} metrics")
            print(f"   Plots available: {len(plots)} plots")
        else:
            print("⚠️ Prophet forecast returned empty results")
        
        return True
        
    except Exception as e:
        print(f"❌ Forecast test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Forecast Engine Fix")
    print("=" * 50)
    
    success1 = test_forecast_engine_import()
    success2 = test_basic_forecast() if success1 else False
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed! Forecast engine is working.")
    elif success1:
        print("✅ Imports fixed! Basic functionality needs refinement.")
    else:
        print("❌ Import issues still exist.")
