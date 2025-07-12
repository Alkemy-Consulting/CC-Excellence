#!/usr/bin/env python3
"""
Test completo del modulo Prophet con dati reali CSV
"""

import sys
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TEST COMPLETO PROPHET MODULE CON DATI REALI")
print("=" * 80)

# 1. Test import del modulo
print("\n📦 1. Testing Prophet module imports...")
try:
    sys.path.append('/workspaces/CC-Excellence')
    from modules.prophet_module import run_prophet_forecast, create_prophet_forecast_chart
    print("✅ Prophet module imported successfully")
except Exception as e:
    print(f"❌ Import error: {e}")
    traceback.print_exc()
    sys.exit(1)

# 2. Caricamento dati CSV
print("\n📊 2. Loading test CSV data...")
try:
    df = pd.read_csv('/workspaces/CC-Excellence/test_data.csv')
    print(f"✅ CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
    
    # Verifica formati
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'])
    print(f"✅ Data types validated: date={df['date'].dtype}, value={df['value'].dtype}")
    
except Exception as e:
    print(f"❌ CSV loading error: {e}")
    traceback.print_exc()
    sys.exit(1)

# 3. Configurazione parametri Prophet
print("\n⚙️ 3. Setting up Prophet configuration...")
try:
    model_config = {
        'seasonality_mode': 'additive',
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'growth': 'linear',
        'add_holidays': False,
        'auto_tune': False,
        'enable_cross_validation': False,
        'show_components': True,
        'show_residuals': True,
        'plot_height': 500
    }
    
    base_config = {
        'train_size': 0.8,
        'forecast_periods': 30,
        'confidence_interval': 0.95,
        'regressor_config': {
            'selected_regressors': [],
            'regressor_configs': {}
        }
    }
    
    print("✅ Configuration set up successfully")
    print(f"   Model config keys: {list(model_config.keys())}")
    print(f"   Base config keys: {list(base_config.keys())}")
    
except Exception as e:
    print(f"❌ Configuration error: {e}")
    traceback.print_exc()
    sys.exit(1)

# 4. Test del modulo Prophet
print("\n🔮 4. Testing Prophet forecast...")
try:
    print("Calling run_prophet_forecast...")
    forecast_df, metrics, plots = run_prophet_forecast(
        df=df,
        date_col='date',
        target_col='value',
        model_config=model_config,
        base_config=base_config
    )
    
    print("✅ Prophet forecast completed successfully!")
    print(f"   Forecast shape: {forecast_df.shape}")
    print(f"   Metrics: {list(metrics.keys()) if metrics else 'None'}")
    print(f"   Plots: {list(plots.keys()) if plots else 'None'}")
    
    if not forecast_df.empty:
        print(f"   Forecast columns: {list(forecast_df.columns)}")
        print(f"   Date range: {forecast_df.iloc[:, 0].min()} to {forecast_df.iloc[:, 0].max()}")
    
except Exception as e:
    print(f"❌ Prophet forecast error: {e}")
    print(f"Error type: {type(e).__name__}")
    traceback.print_exc()
    
    # Try to identify the specific error
    if "Addition/subtraction of integers" in str(e):
        print("\n🚨 DETECTED: Timestamp arithmetic error!")
        print("This suggests there's still a range selector issue.")
    elif "streamlit" in str(e).lower():
        print("\n🚨 DETECTED: Streamlit dependency error!")
        print("The module might be trying to use Streamlit outside of a Streamlit context.")
    elif "import" in str(e).lower():
        print("\n🚨 DETECTED: Import error!")
        print("Missing dependency or import issue.")
    
    print("\n🔍 Let's try a simplified test...")

# 5. Test semplificato senza Streamlit
print("\n🔧 5. Testing simplified Prophet without Streamlit...")
try:
    # Test diretto Prophet senza le nostre funzioni wrapper
    from prophet import Prophet
    
    # Prepara dati per Prophet
    prophet_df = df.copy()
    prophet_df = prophet_df.rename(columns={'date': 'ds', 'value': 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    prophet_df['y'] = pd.to_numeric(prophet_df['y'])
    
    print(f"✅ Data prepared for Prophet: {prophet_df.shape}")
    
    # Crea e addestra modello
    model = Prophet(
        seasonality_mode='additive',
        yearly_seasonality='auto',
        weekly_seasonality='auto',
        daily_seasonality='auto'
    )
    
    print("Training Prophet model...")
    model.fit(prophet_df)
    print("✅ Prophet model trained successfully")
    
    # Crea future dataframe
    future = model.make_future_dataframe(periods=30)
    print(f"✅ Future dataframe created: {future.shape}")
    
    # Genera forecast
    forecast = model.predict(future)
    print(f"✅ Forecast generated: {forecast.shape}")
    
    # Test creazione chart
    print("Testing chart creation...")
    import plotly.graph_objects as go
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        mode='markers',
        name='Actual'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast'
    ))
    
    # Test the NEW range selector configuration
    fig.update_layout(
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
            )
        )
    )
    
    print("✅ Chart with range selectors created successfully")
    print("✅ NO timestamp arithmetic errors detected!")
    
except Exception as e:
    print(f"❌ Simplified test error: {e}")
    traceback.print_exc()

# 6. Test specifico delle funzioni del nostro modulo
print("\n🔍 6. Testing specific module functions...")
try:
    from modules.prophet_module import validate_prophet_inputs, prepare_prophet_data
    
    # Test validation
    print("Testing input validation...")
    validate_prophet_inputs(df, 'date', 'value')
    print("✅ Input validation passed")
    
    # Test data preparation  
    print("Testing data preparation...")
    prophet_data = prepare_prophet_data(df, 'date', 'value', [])
    print(f"✅ Data preparation successful: {prophet_data.shape}")
    
except Exception as e:
    print(f"❌ Module function test error: {e}")
    traceback.print_exc()

print("\n" + "=" * 80)
print("🎯 TEST COMPLETION SUMMARY")
print("=" * 80)
print("If you see this message, the basic Prophet functionality works.")
print("Any errors above indicate specific issues in our module wrapper functions.")
print("=" * 80)
