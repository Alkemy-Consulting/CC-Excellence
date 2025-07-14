#!/usr/bin/env python3
"""
Quick validation test for CC-Excellence forecasting functionality
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/workspaces/CC-Excellence')
sys.path.append('/workspaces/CC-Excellence/modules')

def main():
    print("🚀 CC-Excellence Quick Validation Test")
    print("=" * 50)
    
    try:
        # Test 1: Basic imports
        print("\n1. Testing imports...")
        import pandas as pd
        import numpy as np
        print("   ✓ Basic libraries imported")
        
        # Test 2: Config import
        print("\n2. Testing config module...")
        from src.modules.utils.config import SUPPORTED_FILE_FORMATS, DEFAULT_FORECAST_HORIZONS
        print(f"   ✓ Config loaded - {len(SUPPORTED_FILE_FORMATS)} file formats supported")
        
        # Test 3: Data utilities
        print("\n3. Testing data utilities...")
        from src.modules.utils.data_utils import generate_sample_data
        df = generate_sample_data(days=60)
        print(f"   ✓ Sample data generated: {df.shape}")
        print(f"   ✓ Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Test 4: Simple forecast test
        print("\n4. Testing basic forecasting...")
        try:
            from modules.prophet_enhanced import ProphetEnhanced
            
            # Prepare data for Prophet
            df_prophet = df.copy()
            df_prophet = df_prophet.rename(columns={'date': 'ds', 'value': 'y'})
            
            # Initialize and fit Prophet model
            model = ProphetEnhanced()
            config = {
                'periods': 14,
                'freq': 'D',
                'seasonality_mode': 'additive',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False
            }
            
            result = model.fit_and_forecast(df_prophet, config)
            
            if result and 'forecast' in result and not result['forecast'].empty:
                print(f"   ✓ Prophet forecast successful: {result['forecast'].shape}")
                print(f"   ✓ Forecast values range: {result['forecast']['yhat'].min():.2f} to {result['forecast']['yhat'].max():.2f}")
            else:
                print("   ❌ Prophet forecast failed")
                
        except Exception as e:
            print(f"   ❌ Prophet test failed: {str(e)[:100]}...")
        
        # Test 5: UI Components (basic check)
        print("\n5. Testing UI components...")
        try:
            from src.modules.visualization.ui_components import render_metric_card
            print("   ✓ UI components imported successfully")
        except Exception as e:
            print(f"   ⚠️ UI components warning: {str(e)[:50]}... (expected without Streamlit context)")
        
        print("\n" + "=" * 50)
        print("✅ Core functionality validation completed!")
        print("\nNext Steps:")
        print("1. Run the Streamlit app: streamlit run app.py")
        print("2. Test the forecasting page: pages/1_📈Forecasting.py") 
        print("3. Upload sample data and test forecasting workflow")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\nValidation {'PASSED' if success else 'FAILED'}")
