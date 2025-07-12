#!/usr/bin/env python3
"""
Test specifico per diagnosticare il problema del forecast Prophet
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append('/workspaces/CC-Excellence')

def diagnose_forecast_dates():
    """Diagnosi del problema delle date nel forecast"""
    print("🔍 DIAGNOSI FORECAST PROPHET - PROBLEMA DATE")
    print("=" * 60)
    
    # Carica dati CSV reali
    df = pd.read_csv('/workspaces/CC-Excellence/test_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"📄 Dati originali:")
    print(f"   Range: {df['date'].min()} → {df['date'].max()}")
    print(f"   Totale righe: {len(df)}")
    print(f"   Ultima data: {df['date'].max()}")
    
    # Test Prophet forecast step by step
    from modules.prophet_module import run_prophet_forecast
    
    model_config = {
        'seasonality_mode': 'additive',
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'yearly_seasonality': 'auto',
        'weekly_seasonality': 'auto',
        'daily_seasonality': 'auto',
        'growth': 'linear',
        'add_holidays': False
    }
    
    base_config = {
        'train_size': 0.8,
        'forecast_periods': 30,  # 30 giorni dopo l'ultima data
        'confidence_interval': 0.95
    }
    
    print(f"\n⚙️  Configurazione:")
    print(f"   Train size: {base_config['train_size']} ({base_config['train_size'] * 100}%)")
    print(f"   Forecast periods: {base_config['forecast_periods']} giorni")
    
    # Calcola manualmente split point
    split_point = int(len(df) * base_config['train_size'])
    train_end_date = df.iloc[split_point-1]['date']
    test_start_date = df.iloc[split_point]['date'] if split_point < len(df) else df.iloc[-1]['date']
    
    print(f"\n📊 Split dati:")
    print(f"   Split point: {split_point}")
    print(f"   Train end: {train_end_date}")
    print(f"   Test start: {test_start_date}")
    print(f"   Train rows: {split_point}")
    print(f"   Test rows: {len(df) - split_point}")
    
    # Expected forecast dates
    expected_forecast_start = df['date'].max() + timedelta(days=1)
    expected_forecast_end = expected_forecast_start + timedelta(days=base_config['forecast_periods']-1)
    
    print(f"\n🎯 Forecast atteso:")
    print(f"   Dovrebbe iniziare: {expected_forecast_start}")
    print(f"   Dovrebbe finire: {expected_forecast_end}")
    
    # Esegui forecast
    print(f"\n🚀 Eseguendo forecast...")
    forecast_df, metrics, plots = run_prophet_forecast(
        df=df,
        date_col='date',
        target_col='value',
        model_config=model_config,
        base_config=base_config
    )
    
    if not forecast_df.empty:
        print(f"\n📈 Risultato forecast:")
        print(f"   Shape: {forecast_df.shape}")
        print(f"   Range date: {forecast_df['date'].min()} → {forecast_df['date'].max()}")
        
        # Analizza le date del forecast
        future_dates = forecast_df[forecast_df['date'] > df['date'].max()]
        print(f"\n🔮 Date future (oltre i dati originali):")
        print(f"   Count: {len(future_dates)}")
        if len(future_dates) > 0:
            print(f"   Range: {future_dates['date'].min()} → {future_dates['date'].max()}")
            print(f"   Prime 5 date future:")
            for i, date in enumerate(future_dates['date'].head()):
                print(f"      {i+1}: {date}")
        else:
            print("   ❌ PROBLEMA: Nessuna data futura trovata!")
        
        # Verifica se ci sono date sbagliate
        unexpected_dates = forecast_df[forecast_df['date'] < df['date'].min()]
        if len(unexpected_dates) > 0:
            print(f"\n⚠️  Date inaspettate (prima dei dati originali):")
            print(f"   Count: {len(unexpected_dates)}")
            print(f"   Range: {unexpected_dates['date'].min()} → {unexpected_dates['date'].max()}")
    else:
        print("   ❌ Forecast vuoto!")
    
    return forecast_df

def test_prophet_internal_dates():
    """Test delle date interne di Prophet"""
    print(f"\n🔬 TEST PROPHET INTERNO")
    print("-" * 40)
    
    from prophet import Prophet
    
    # Prepara dati come fa il modulo
    df = pd.read_csv('/workspaces/CC-Excellence/test_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    prophet_df = df[['date', 'value']].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Split come nel modulo
    train_size = 0.8
    split_point = int(len(prophet_df) * train_size)
    train_df = prophet_df[:split_point]
    
    print(f"📊 Train data:")
    print(f"   Range: {train_df['ds'].min()} → {train_df['ds'].max()}")
    print(f"   Ultima data train: {train_df['ds'].max()}")
    
    # Crea e fit model
    model = Prophet()
    model.fit(train_df)
    
    # Crea future dataframe
    forecast_periods = 30
    future = model.make_future_dataframe(periods=forecast_periods)
    
    print(f"\n🔮 Future dataframe:")
    print(f"   Shape: {future.shape}")
    print(f"   Range: {future['ds'].min()} → {future['ds'].max()}")
    print(f"   Ultima data: {future['ds'].max()}")
    
    # Future dates beyond training
    future_beyond_train = future[future['ds'] > train_df['ds'].max()]
    print(f"\n📅 Date oltre training:")
    print(f"   Count: {len(future_beyond_train)}")
    if len(future_beyond_train) > 0:
        print(f"   Range: {future_beyond_train['ds'].min()} → {future_beyond_train['ds'].max()}")
        print(f"   Prime 5:")
        for i, date in enumerate(future_beyond_train['ds'].head()):
            print(f"      {i+1}: {date}")
    
    # Generate forecast
    forecast = model.predict(future)
    print(f"\n📈 Forecast generato:")
    print(f"   Shape: {forecast.shape}")
    print(f"   Range: {forecast['ds'].min()} → {forecast['ds'].max()}")
    
    return future, forecast

if __name__ == "__main__":
    # Test 1: Diagnosi del modulo Prophet
    forecast_result = diagnose_forecast_dates()
    
    # Test 2: Test Prophet interno
    future_df, forecast_df = test_prophet_internal_dates()
    
    print(f"\n" + "=" * 60)
    print("🎯 CONCLUSIONI")
    print("=" * 60)
    
    if not forecast_result.empty:
        print("✅ Il modulo Prophet genera forecast")
        future_count = len(forecast_result[forecast_result['date'] > pd.to_datetime('2024-06-30')])
        print(f"📊 Date future generate: {future_count}")
        if future_count > 0:
            print("✅ Forecast si estende oltre i dati originali")
        else:
            print("❌ Forecast NON si estende oltre i dati originali")
    else:
        print("❌ Problema nella generazione del forecast")
