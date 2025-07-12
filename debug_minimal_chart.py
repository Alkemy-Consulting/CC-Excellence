#!/usr/bin/env python3
"""
Versione minimalista per testare ogni parte della chart function
"""

import sys
import pandas as pd
import traceback
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

sys.path.append('/workspaces/CC-Excellence')

print("🔍 TEST MINIMALISTA: create_prophet_forecast_chart")
print("=" * 60)

# Load test data
df = pd.read_csv('/workspaces/CC-Excellence/test_data.csv')
df['date'] = pd.to_datetime(df['date'])
df['value'] = pd.to_numeric(df['value'])

try:
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
    
    # NOW: Test our chart function step by step
    print("\n📊 STEP-BY-STEP CHART CREATION")
    
    # Calculate confidence percentage for display
    confidence_interval = 0.95
    confidence_percentage = int(confidence_interval * 100)
    
    # Ensure proper datetime conversion for all data
    forecast_df = forecast.copy()
    actual_data = df.copy()
    
    print("Step 1: Data preparation...")
    try:
        # Convert date columns to datetime if they aren't already
        forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
        actual_data['date'] = pd.to_datetime(actual_data['date'])
        print("✅ Step 1 passed")
    except Exception as e:
        print(f"❌ Step 1 failed: {e}")
        traceback.print_exc()
    
    print("Step 2: Create basic figure...")
    try:
        # Create the main plot
        fig = go.Figure()
        print("✅ Step 2 passed")
    except Exception as e:
        print(f"❌ Step 2 failed: {e}")
        traceback.print_exc()
    
    print("Step 3: Data separation...")
    try:
        # Separate historical and future data in forecast for better visualization
        last_historical_date = actual_data['date'].max()
        historical_forecast = forecast_df[forecast_df['ds'] <= last_historical_date]
        future_forecast = forecast_df[forecast_df['ds'] > last_historical_date]
        print(f"✅ Step 3 passed - Historical: {len(historical_forecast)}, Future: {len(future_forecast)}")
    except Exception as e:
        print(f"❌ Step 3 failed: {e}")
        traceback.print_exc()
    
    print("Step 4: Add actual values trace...")
    try:
        # Add actual values (white points for training period)
        fig.add_trace(go.Scatter(
            x=actual_data['date'],
            y=actual_data['value'],
            mode='markers',
            name='Actual Values',
            marker=dict(color='white', size=4, line=dict(color='black', width=1)),
            hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        print("✅ Step 4 passed")
    except Exception as e:
        print(f"❌ Step 4 failed: {e}")
        traceback.print_exc()
    
    print("Step 5: Add forecast traces...")
    try:
        # Add historical predictions (blue line, lighter)
        if not historical_forecast.empty:
            fig.add_trace(go.Scatter(
                x=historical_forecast['ds'],
                y=historical_forecast['yhat'],
                mode='lines',
                name='Historical Fit',
                line=dict(color='lightblue', width=2, dash='dot'),
                hovertemplate='<b>Historical Fit</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
        
        # Add future predictions (blue line, solid and thicker)
        if not future_forecast.empty:
            fig.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat'],
                mode='lines',
                name='Future Forecast',
                line=dict(color='blue', width=3),
                hovertemplate='<b>Future Forecast</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ))
        print("✅ Step 5 passed")
    except Exception as e:
        print(f"❌ Step 5 failed: {e}")
        traceback.print_exc()
    
    print("Step 6: Add confidence intervals...")
    try:
        # Add uncertainty interval for future periods only (blue shade)
        if not future_forecast.empty:
            fig.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=future_forecast['ds'],
                y=future_forecast['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 100, 255, 0.2)',
                name=f'{confidence_percentage}% Confidence Interval',
                hovertemplate=f'<b>{confidence_percentage}% Confidence Interval</b><br>Date: %{{x}}<br>Upper: %{{text}}<br>Lower: %{{y:.2f}}<extra></extra>',
                text=future_forecast['yhat_upper'].round(2)
            ))
        print("✅ Step 6 passed")
    except Exception as e:
        print(f"❌ Step 6 failed: {e}")
        traceback.print_exc()
    
    print("Step 7: Add trend line...")
    try:
        # Add trend line for full forecast period
        fig.add_trace(go.Scatter(
            x=forecast_df['ds'],
            y=forecast_df['trend'],
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2),
            hovertemplate='<b>Trend</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        print("✅ Step 7 passed")
    except Exception as e:
        print(f"❌ Step 7 failed: {e}")
        traceback.print_exc()
    
    print("Step 8: Add vertical separation line...")
    try:
        # Add vertical line to separate historical from future data
        fig.add_vline(
            x=last_historical_date,
            line=dict(color='gray', width=2, dash='dash'),
            opacity=0.7,
            annotation_text="Start of Forecast",
            annotation_position="top"
        )
        print("✅ Step 8 passed")
    except Exception as e:
        print(f"❌ Step 8 failed: {e}")
        traceback.print_exc()
    
    print("Step 9: Add changepoints (THIS IS THE SUSPECT)...")
    try:
        # Add changepoints (vertical lines) - Enhanced with proper error handling
        if hasattr(model, 'changepoints') and len(model.changepoints) > 0:
            print(f"   Model has {len(model.changepoints)} changepoints")
            
            # Filter changepoints to those within the actual data range
            data_start = actual_data['date'].min()
            data_end = actual_data['date'].max()
            print(f"   Data range: {data_start} to {data_end}")
            
            # Fixed: Use iloc for proper Series indexing instead of enumerate
            for i in range(len(model.changepoints)):
                changepoint = model.changepoints.iloc[i]
                print(f"   Processing changepoint {i}: {changepoint}")
                
                # Convert changepoint to pandas Timestamp for comparison
                cp_date = pd.to_datetime(changepoint)
                print(f"   Converted to: {cp_date}")
                
                # Only add changepoints within the historical data range
                if data_start <= cp_date <= data_end:
                    print(f"   Adding vline for changepoint {i}...")
                    fig.add_vline(
                        x=cp_date,
                        line=dict(color='orange', width=1, dash='dot'),
                        opacity=0.7,
                        annotation_text=f"CP {i+1}",
                        annotation_position="top"
                    )
                    print(f"   ✅ Changepoint {i} added successfully")
                else:
                    print(f"   Skipping changepoint {i} (outside data range)")
                    
                # Test only first few to avoid spam
                if i >= 2:
                    print(f"   ... (testing only first 3 changepoints)")
                    break
            
            print("✅ Step 9 passed")
        else:
            print("   No changepoints available")
            print("✅ Step 9 passed (no changepoints)")
    except Exception as e:
        print(f"❌ Step 9 failed: {e}")
        traceback.print_exc()
        
        # This is likely our culprit
        if "Addition/subtraction" in str(e):
            print("🎯 FOUND IT! The changepoints step causes the timestamp arithmetic error!")
    
    print("Step 10: Apply layout with range selectors...")
    try:
        # Apply same chart formatting as Historical Time Series with Trend Analysis (Tab 1)
        fig.update_layout(
            title="Prophet Forecast Results",
            xaxis_title="Date",
            yaxis_title='value',
            height=500,  # Same height as PLOT_CONFIG['height']
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",  # Align legend to the right
                x=0.95  # Position at 95% of the width (right side)
            ),
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=30, label="1M", step="day", stepmode="backward"),
                        dict(count=90, label="3M", step="day", stepmode="backward"),
                        dict(count=180, label="6M", step="day", stepmode="backward"),
                        dict(count=365, label="1Y", step="day", stepmode="backward"),
                        dict(count=730, label="2Y", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ],
                    x=0.02,  # Position range selector at left (2% from left edge)
                    xanchor="left",  # Anchor to the left
                    y=1.02,  # Same height as legend
                    yanchor="bottom"
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        print("✅ Step 10 passed")
    except Exception as e:
        print(f"❌ Step 10 failed: {e}")
        traceback.print_exc()
        
        if "Addition/subtraction" in str(e):
            print("🎯 ALTERNATIVE POSSIBILITY! The layout step causes the timestamp arithmetic error!")
    
    print("\n🎉 If all steps passed, the chart should work!")
        
except Exception as e:
    print(f"❌ Overall error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎯 TEST MINIMALISTA COMPLETE")
print("=" * 60)
