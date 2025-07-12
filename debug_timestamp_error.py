#!/usr/bin/env python3
"""
Debug script to identify exactly where the timestamp arithmetic error occurs
"""

import sys
import pandas as pd
import numpy as np
import traceback
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/workspaces/CC-Excellence')

print("🔍 DEBUG: TIMESTAMP ARITHMETIC ERROR LOCATION")
print("=" * 60)

# Load test data
df = pd.read_csv('/workspaces/CC-Excellence/test_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['value'] = pd.to_numeric(df['value'])

print(f"✅ Data loaded: {df.shape}")

# Test each part of the Prophet forecast chart creation
try:
    from modules.prophet_module import run_prophet_forecast
    from prophet import Prophet
    import plotly.graph_objects as go
    
    # Create a minimal Prophet model
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
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=30)
    
    # Generate forecast
    forecast = model.predict(future)
    
    print("✅ Prophet model ready, now testing chart creation step by step...")
    
    # Test 1: Basic figure creation
    print("\n📊 Test 1: Basic figure creation...")
    try:
        fig = go.Figure()
        print("✅ Basic figure created")
    except Exception as e:
        print(f"❌ Basic figure error: {e}")
        traceback.print_exc()
    
    # Test 2: Adding actual data trace
    print("\n📊 Test 2: Adding actual data trace...")
    try:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='markers',
            name='Actual Values'
        ))
        print("✅ Actual data trace added")
    except Exception as e:
        print(f"❌ Actual data trace error: {e}")
        traceback.print_exc()
    
    # Test 3: Adding forecast trace
    print("\n📊 Test 3: Adding forecast trace...")
    try:
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast'
        ))
        print("✅ Forecast trace added")
    except Exception as e:
        print(f"❌ Forecast trace error: {e}")
        traceback.print_exc()
    
    # Test 4: Adding vertical line (this might be problematic)
    print("\n📊 Test 4: Adding vertical line...")
    try:
        last_historical_date = df['date'].max()
        fig.add_vline(x=last_historical_date, line=dict(color='gray', width=2))
        print("✅ Vertical line added")
    except Exception as e:
        print(f"❌ Vertical line error: {e}")
        traceback.print_exc()
    
    # Test 5: Adding changepoints (this is likely problematic)
    print("\n📊 Test 5: Adding changepoints...")
    try:
        if hasattr(model, 'changepoints') and len(model.changepoints) > 0:
            print(f"Model has {len(model.changepoints)} changepoints")
            print(f"Changepoints type: {type(model.changepoints)}")
            print(f"Changepoints index: {model.changepoints.index}")
            print(f"First few changepoints: {model.changepoints.head()}")
            
            # Iterate using iloc instead of direct indexing
            for i in range(min(3, len(model.changepoints))):  # Limit to first 3
                changepoint = model.changepoints.iloc[i]
                print(f"Processing changepoint {i}: {changepoint} (type: {type(changepoint)})")
                
                # Test different conversion approaches
                try:
                    cp_date_1 = pd.to_datetime(changepoint)
                    print(f"  ✅ pd.to_datetime worked: {cp_date_1}")
                except Exception as e:
                    print(f"  ❌ pd.to_datetime failed: {e}")
                
                try:
                    # This might be the problematic line
                    fig.add_vline(
                        x=cp_date_1,
                        line=dict(color='orange', width=1, dash='dot')
                    )
                    print(f"  ✅ Changepoint {i} vline added")
                except Exception as e:
                    print(f"  ❌ Changepoint {i} vline error: {e}")
                    # This is likely where the error occurs
                    if "Addition/subtraction" in str(e):
                        print(f"  🎯 FOUND THE ERROR! Changepoint processing causes timestamp arithmetic error")
                        traceback.print_exc()
                        break
        else:
            print("No changepoints available")
    except Exception as e:
        print(f"❌ Changepoints processing error: {e}")
        traceback.print_exc()
    
    # Test 6: Layout with range selectors (this should work now)
    print("\n📊 Test 6: Layout with range selectors...")
    try:
        fig.update_layout(
            title="Test Chart",
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=30, label="1M", step="day", stepmode="backward"),
                        dict(count=90, label="3M", step="day", stepmode="backward"),
                        dict(count=180, label="6M", step="day", stepmode="backward"),
                        dict(count=365, label="1Y", step="day", stepmode="backward"),
                        dict(count=730, label="2Y", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ]
                ),
                type="date"
            )
        )
        print("✅ Layout with range selectors applied")
    except Exception as e:
        print(f"❌ Layout error: {e}")
        traceback.print_exc()
        
except Exception as e:
    print(f"❌ Overall error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎯 DEBUG COMPLETE")
print("=" * 60)
