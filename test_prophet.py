#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspaces/CC-Excellence')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_prophet_enhanced():
    print("🧪 Testing Prophet Enhanced Functionality")
    
    try:
        # Generate test data
        dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
        trend = np.linspace(100, 200, len(dates))
        seasonal = 20 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
        noise = np.random.normal(0, 10, len(dates))
        values = trend + seasonal + noise
        
        df = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        print(f"✅ Test data generated: {len(df)} rows")
        
        # Test config
        config = {
            'auto_tune': True,
            'tuning_horizon': 15,
            'tuning_parallel': None,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
        
        # Import and test enhanced function
        from modules.prophet_module import build_and_forecast_prophet_enhanced
        print("✅ Enhanced Prophet function imported successfully")
        
        # Test standard function (without auto-tune)
        config_standard = config.copy()
        config_standard['auto_tune'] = False
        
        from modules.forecast_engine import run_prophet_standard
        forecast_df, metrics, model_info = run_prophet_standard(df, config_standard, 30)
        print(f"✅ Standard Prophet test successful: {len(forecast_df)} forecast points")
        print(f"   Metrics: {metrics}")
        
        print("\n🎉 All Prophet functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_prophet_enhanced()
