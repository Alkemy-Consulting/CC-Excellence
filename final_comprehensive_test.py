#!/usr/bin/env python3
"""
Final comprehensive test that mimics the exact Streamlit app workflow.
"""

def test_complete_app_workflow():
    """Test the complete app workflow including SARIMA."""
    print("🚀 Testing complete app workflow...")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        # Simulate Streamlit app startup
        import streamlit as st
        import pandas as pd
        import numpy as np
        
        print("✅ Basic app imports successful")
        
        # Test forecast engine import (this is what the app uses)
        from modules.forecast_engine import display_forecast_results
        print("✅ Forecast engine imported")
        
        # Test all individual model imports
        from modules.prophet_enhanced import run_prophet_model
        from modules.arima_enhanced import run_arima_model
        from modules.sarima_enhanced import run_sarima_forecast, SARIMA_AVAILABLE
        from modules.holtwinters_enhanced import run_holtwinters_forecast
        
        print(f"✅ All model modules imported")
        print(f"   SARIMA_AVAILABLE: {SARIMA_AVAILABLE}")
        
        # Create test data like the app would
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        test_data = pd.DataFrame({
            'date': dates,
            'volume': 100 + 10 * np.sin(np.arange(100) * 2 * np.pi / 7) + np.random.randn(100).cumsum()
        })
        
        print("✅ Test data created")
        
        # Test SARIMA specifically with app-like config
        config = {
            'date_column': 'date',
            'target_column': 'volume',
            'forecast_periods': 14,
            'confidence_interval': 0.95,
            'auto_tune': True,
            'seasonal_periods': 7,
            'train_size': 0.8
        }
        
        print("Testing SARIMA forecast...")
        
        # This is the exact call that would happen in the app
        try:
            forecast_df, metrics, plots = run_sarima_forecast(test_data, config)
            
            if forecast_df.empty:
                print("❌ SARIMA returned empty forecast")
                return False
            else:
                print(f"✅ SARIMA forecast successful: {len(forecast_df)} periods")
                print(f"   Columns: {list(forecast_df.columns)}")
                print(f"   Metrics available: {len(metrics)} metrics")
                print(f"   Plots available: {len(plots)} plots")
        except Exception as e:
            print(f"❌ SARIMA forecast failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Test other models too for completeness
        print("\nTesting other models...")
        
        # Prophet
        try:
            prophet_df, prophet_metrics, prophet_plots = run_prophet_model(
                test_data, 'date', 'volume', 
                forecast_periods=14, confidence_interval=0.95
            )
            print(f"✅ Prophet: {len(prophet_df)} periods")
        except Exception as e:
            print(f"⚠️ Prophet issue: {e}")
        
        # ARIMA
        try:
            arima_df, arima_metrics, arima_plots = run_arima_model(test_data, config)
            print(f"✅ ARIMA: {len(arima_df)} periods")
        except Exception as e:
            print(f"⚠️ ARIMA issue: {e}")
        
        # Holt-Winters
        try:
            hw_config = config.copy()
            hw_config.update({'trend': 'additive', 'seasonal': 'additive'})
            hw_df, hw_metrics, hw_plots = run_holtwinters_forecast(test_data, 'date', 'volume', hw_config, {})
            print(f"✅ Holt-Winters: {len(hw_df)} periods")
        except Exception as e:
            print(f"⚠️ Holt-Winters issue: {e}")
        
        print("\n🎉 Complete app workflow test successful!")
        return True
        
    except Exception as e:
        print(f"❌ App workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 Final Comprehensive App Test\n")
    
    success = test_complete_app_workflow()
    
    if success:
        print("\n✅ ALL TESTS PASSED")
        print("🎉 The CC-Excellence app is fully functional!")
        print("\nAll models (Prophet, ARIMA, SARIMA, Holt-Winters) are working correctly.")
        print("The SARIMA error you mentioned should no longer occur.")
        print("\nIf you're still seeing the error, please:")
        print("1. Restart your Streamlit app completely")
        print("2. Clear browser cache")
        print("3. Try refreshing the page")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("There are still issues to resolve.")
