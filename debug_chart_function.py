#!/usr/bin/env python3
"""
Test specifico della funzione create_prophet_forecast_chart
"""

import sys
import pandas as pd
import traceback
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/workspaces/CC-Excellence')

print("🔍 TEST SPECIFICO: create_prophet_forecast_chart")
print("=" * 60)

# Load test data
df = pd.read_csv('/workspaces/CC-Excellence/test_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['value'] = pd.to_numeric(df['value'])

try:
    from modules.prophet_module import create_prophet_forecast_chart
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
    
    # Now test our function
    print("\n📊 Testing create_prophet_forecast_chart...")
    try:
        chart = create_prophet_forecast_chart(
            model=model,
            forecast_df=forecast,
            actual_data=df,
            date_col='date',
            target_col='value',
            confidence_interval=0.95
        )
        
        if chart is not None:
            print("✅ Chart created successfully!")
            print(f"Chart type: {type(chart)}")
        else:
            print("❌ Chart is None - function returned None due to error")
            
    except Exception as e:
        print(f"❌ Chart creation error: {e}")
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

print("\n" + "=" * 60)
print("🎯 TEST SPECIFICO COMPLETE")
print("=" * 60)
