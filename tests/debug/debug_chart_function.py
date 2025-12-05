#!/usr/bin/env python3
"""
Specific test for the create_prophet_plots function
"""

import sys
import pandas as pd
import traceback
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/workspaces/CC-Excellence')

print("🔍 SPECIFIC TEST: create_prophet_plots")
print("=" * 60)

# Load test data
df = pd.read_csv('/workspaces/CC-Excellence/test_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['value'] = pd.to_numeric(df['value'])

try:
    from modules.prophet_module import create_prophet_plots, ProphetForecastResult
    from prophet import Prophet
    
    # Create Prophet model
    prophet_df = df.copy()
    prophet_df = prophet_df.rename(columns={'date': 'ds', 'value': 'y'})
    
    model = Prophet(
        seasonality_mode='additive',
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality='auto'
    )
    
    print("🔮 Training Prophet model...")
    model.fit(prophet_df)
    
    # Create future dataframe and forecast
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    print("✅ Prophet model and forecast ready")
    print(f"Forecast shape: {forecast.shape}")
    print(f"Forecast columns: {list(forecast.columns)}")
    
    # Create ProphetForecastResult for the new function signature
    result = ProphetForecastResult(
        success=True,
        error=None,
        model=model,
        raw_forecast=forecast,
        metrics={'mae': 0, 'mape': 0, 'rmse': 0}  # Dummy metrics
    )
    
    # Now test our function
    print("\n📊 Testing create_prophet_plots...")
    try:
        plots = create_prophet_plots(
            result=result,
            df=df,
            date_col='date',
            target_col='value'
        )
        
        if plots is not None:
            print("✅ Plots created successfully!")
            print(f"Plots type: {type(plots)}")
            print(f"Plots keys: {list(plots.keys()) if isinstance(plots, dict) else 'Not a dict'}")
        else:
            print("❌ Plots is None - function returned None due to error")
            
    except Exception as e:
        print(f"❌ Plot creation error: {e}")
        print(f"Error type: {type(e).__name__}")
        traceback.print_exc()
        
        # Check if this is the timestamp arithmetic error
        if "Addition/subtraction of integers" in str(e):
            print("\n🎯 CONFIRMED: This is the timestamp arithmetic error!")
            print("Looking for the exact line...")
            
            # Get the full traceback
            tb = traceback.format_exc()
            lines = tb.split('\n')
            for i, line in enumerate(lines):
                if "Addition/subtraction" in line:
                    print(f"Error line: {line}")
                    # Print some context
                    for j in range(max(0, i-3), min(len(lines), i+3)):
                        prefix = ">>> " if j == i else "    "
                        print(f"{prefix}{lines[j]}")
                    break
        
except Exception as e:
    print(f"❌ Overall error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎯 SPECIFIC TEST COMPLETE")
print("=" * 60)
sys.exit(0)
