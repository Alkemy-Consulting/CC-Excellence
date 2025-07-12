#!/usr/bin/env python3
"""
Audit completo del modulo Prophet per verificare:
1. Generazione corretta dei periodi di forecast
2. Visualizzazione dei dati futuri nel grafico
3. Struttura dati di output
4. Allineamento date
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add modules path
sys.path.append('/workspaces/CC-Excellence')
sys.path.append('/workspaces/CC-Excellence/modules')

def create_test_data():
    """Crea dati di test con pattern riconoscibili"""
    print("📊 Creando dati di test...")
    
    # Genera 100 giorni di dati storici
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Pattern sinusoidale con trend crescente
    values = []
    for i, date in enumerate(dates):
        # Trend crescente
        trend = i * 0.5
        # Seasonality settimanale 
        weekly_season = 10 * np.sin(2 * np.pi * i / 7)
        # Noise
        noise = np.random.normal(0, 2)
        
        value = 100 + trend + weekly_season + noise
        values.append(max(0, value))  # Evita valori negativi
    
    df = pd.DataFrame({
        'date': dates,
        'calls': values
    })
    
    print(f"✅ Dati creati: {len(df)} punti dal {df['date'].min()} al {df['date'].max()}")
    print(f"   Ultimo valore: {df['calls'].iloc[-1]:.2f}")
    print(f"   Range valori: {df['calls'].min():.2f} - {df['calls'].max():.2f}")
    
    return df

def test_prophet_forecast_periods():
    """Test specifico per verificare i periodi di forecast"""
    print("\n🔮 Test: Verifica periodi di forecast Prophet")
    
    try:
        from modules.prophet_module import run_prophet_forecast
        
        # Crea dati di test
        df = create_test_data()
        
        # Configurazione test
        model_config = {
            'yearly_seasonality': 'auto',
            'weekly_seasonality': 'auto',
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'interval_width': 0.8
        }
        
        # Configurazione base con periodi di forecast specifici
        base_config = {
            'forecast_periods': 30,  # 30 giorni nel futuro
            'confidence_interval': 0.8,
            'train_size': 0.8
        }
        
        print(f"📋 Configurazione test:")
        print(f"   - Dati storici: {len(df)} punti")
        print(f"   - Periodi forecast: {base_config['forecast_periods']}")
        print(f"   - Train size: {base_config['train_size']}")
        print(f"   - Confidence interval: {base_config['confidence_interval']}")
        
        # Calcola date attese
        last_historical_date = df['date'].max()
        expected_forecast_start = last_historical_date + timedelta(days=1)
        expected_forecast_end = expected_forecast_start + timedelta(days=base_config['forecast_periods']-1)
        
        print(f"📅 Date attese:")
        print(f"   - Ultimo dato storico: {last_historical_date}")
        print(f"   - Inizio forecast: {expected_forecast_start}")
        print(f"   - Fine forecast: {expected_forecast_end}")
        
        # Esegui Prophet forecast
        print("\n🔄 Eseguendo Prophet forecast...")
        forecast_df, metrics, plots = run_prophet_forecast(
            df, 'date', 'calls', model_config, base_config
        )
        
        print(f"✅ Forecast completato!")
        print(f"   - Shape output: {forecast_df.shape}")
        print(f"   - Colonne: {list(forecast_df.columns)}")
        print(f"   - Metrics: {metrics}")
        print(f"   - Plots disponibili: {list(plots.keys()) if plots else 'Nessuno'}")
        
        # AUDIT 1: Verifica numero di righe del forecast
        print(f"\n🔍 AUDIT 1: Verifica numero di righe")
        expected_total_rows = len(df) + base_config['forecast_periods']
        actual_rows = len(forecast_df)
        
        print(f"   - Righe attese: {expected_total_rows} (storici: {len(df)} + forecast: {base_config['forecast_periods']})")
        print(f"   - Righe effettive: {actual_rows}")
        
        if actual_rows == expected_total_rows:
            print("   ✅ PASS: Numero di righe corretto")
        else:
            print("   ❌ FAIL: Numero di righe non corretto")
            print(f"      Differenza: {actual_rows - expected_total_rows}")
        
        # AUDIT 2: Verifica range di date
        print(f"\n🔍 AUDIT 2: Verifica range di date")
        if not forecast_df.empty:
            first_forecast_date = forecast_df['date'].min()
            last_forecast_date = forecast_df['date'].max()
            
            print(f"   - Prima data forecast: {first_forecast_date}")
            print(f"   - Ultima data forecast: {last_forecast_date}")
            print(f"   - Prima data attesa: {df['date'].min()}")
            print(f"   - Ultima data attesa: {expected_forecast_end}")
            
            # Verifica che includa sia dati storici che futuri
            historical_start = df['date'].min()
            if first_forecast_date <= historical_start:
                print("   ✅ PASS: Include dati storici")
            else:
                print("   ❌ FAIL: Non include tutti i dati storici")
            
            if last_forecast_date >= expected_forecast_end:
                print("   ✅ PASS: Include periodi di forecast futuri")
            else:
                print("   ❌ FAIL: Non include tutti i periodi di forecast")
                print(f"      Mancano {(expected_forecast_end - last_forecast_date).days} giorni")
        
        # AUDIT 3: Verifica contenuto dati futuri
        print(f"\n🔍 AUDIT 3: Verifica dati futuri")
        future_data = forecast_df[forecast_df['date'] > last_historical_date]
        print(f"   - Righe di dati futuri: {len(future_data)}")
        print(f"   - Righe attese: {base_config['forecast_periods']}")
        
        if len(future_data) == base_config['forecast_periods']:
            print("   ✅ PASS: Numero di dati futuri corretto")
        else:
            print("   ❌ FAIL: Numero di dati futuri non corretto")
        
        if not future_data.empty:
            print(f"   - Range valori futuri: {future_data['calls_forecast'].min():.2f} - {future_data['calls_forecast'].max():.2f}")
            
            # Verifica che i valori futuri siano ragionevoli
            historical_mean = df['calls'].mean()
            future_mean = future_data['calls_forecast'].mean()
            
            print(f"   - Media storica: {historical_mean:.2f}")
            print(f"   - Media forecast: {future_mean:.2f}")
            
            # I valori del forecast dovrebbero essere nell'ordine di grandezza dei dati storici
            if 0.5 * historical_mean <= future_mean <= 2.0 * historical_mean:
                print("   ✅ PASS: Valori futuri ragionevoli")
            else:
                print("   ⚠️  WARNING: Valori futuri potrebbero essere irrealistici")
        
        # AUDIT 4: Verifica plot e visualizzazione
        print(f"\n🔍 AUDIT 4: Verifica plot")
        if 'forecast_plot' in plots:
            plot = plots['forecast_plot']
            print("   ✅ PASS: Plot di forecast generato")
            
            # Controlla le tracce del plot
            if hasattr(plot, 'data'):
                traces = plot.data
                trace_names = [trace.name for trace in traces if hasattr(trace, 'name')]
                print(f"   - Tracce nel plot: {trace_names}")
                
                # Verifica che ci siano le tracce essenziali
                required_traces = ['Actual Values', 'Predictions']
                missing_traces = [trace for trace in required_traces if trace not in trace_names]
                
                if not missing_traces:
                    print("   ✅ PASS: Tutte le tracce essenziali presenti")
                else:
                    print(f"   ❌ FAIL: Tracce mancanti: {missing_traces}")
                
                # Verifica range temporale del plot
                prediction_trace = None
                for trace in traces:
                    if hasattr(trace, 'name') and trace.name == 'Predictions':
                        prediction_trace = trace
                        break
                
                if prediction_trace and hasattr(prediction_trace, 'x'):
                    plot_dates = pd.to_datetime(prediction_trace.x)
                    plot_start = plot_dates.min()
                    plot_end = plot_dates.max()
                    
                    print(f"   - Range date nel plot: {plot_start} - {plot_end}")
                    
                    if plot_end >= expected_forecast_end:
                        print("   ✅ PASS: Plot include periodi futuri")
                    else:
                        print("   ❌ FAIL: Plot non include tutti i periodi futuri")
        else:
            print("   ❌ FAIL: Plot di forecast non generato")
        
        # RIEPILOGO AUDIT
        print(f"\n📋 RIEPILOGO AUDIT PROPHET:")
        print(f"   1. Numero righe: {'✅' if actual_rows == expected_total_rows else '❌'}")
        print(f"   2. Range date: {'✅' if not forecast_df.empty and last_forecast_date >= expected_forecast_end else '❌'}")
        print(f"   3. Dati futuri: {'✅' if len(future_data) == base_config['forecast_periods'] else '❌'}")
        print(f"   4. Plot: {'✅' if 'forecast_plot' in plots else '❌'}")
        
        return True, forecast_df, metrics, plots
        
    except Exception as e:
        print(f"❌ ERRORE durante test Prophet: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False, None, None, None

def test_prophet_data_alignment():
    """Test per verificare allineamento dei dati storici vs forecast"""
    print("\n📊 Test: Verifica allineamento dati storici vs forecast")
    
    success, forecast_df, metrics, plots = test_prophet_forecast_periods()
    
    if not success or forecast_df is None or forecast_df.empty:
        print("❌ Test precedente fallito, impossibile verificare allineamento")
        return False
    
    try:
        # Crea dati di test per confronto
        original_df = create_test_data()
        
        # Separa dati storici e forecast
        last_historical_date = original_df['date'].max()
        historical_forecast = forecast_df[forecast_df['date'] <= last_historical_date]
        future_forecast = forecast_df[forecast_df['date'] > last_historical_date]
        
        print(f"📋 Analisi allineamento:")
        print(f"   - Dati originali: {len(original_df)} punti")
        print(f"   - Forecast storico: {len(historical_forecast)} punti")
        print(f"   - Forecast futuro: {len(future_forecast)} punti")
        
        # Verifica overlap tra dati originali e forecast storico
        merged = pd.merge(
            original_df[['date', 'calls']], 
            historical_forecast[['date', 'calls_forecast']], 
            on='date', 
            how='inner'
        )
        
        print(f"   - Punti in comune: {len(merged)} / {len(original_df)}")
        
        if len(merged) == len(original_df):
            print("   ✅ PASS: Perfetto allineamento dati storici")
        else:
            print("   ⚠️  WARNING: Allineamento incompleto")
            
            # Trova date mancanti
            missing_dates = set(original_df['date']) - set(historical_forecast['date'])
            if missing_dates:
                print(f"   Date mancanti nel forecast: {sorted(missing_dates)[:5]}...")
        
        # Verifica continuità temporale
        all_dates = sorted(forecast_df['date'])
        date_gaps = []
        
        for i in range(1, len(all_dates)):
            expected_next = all_dates[i-1] + timedelta(days=1)
            if all_dates[i] != expected_next:
                date_gaps.append((all_dates[i-1], all_dates[i]))
        
        if not date_gaps:
            print("   ✅ PASS: Nessun gap temporale")
        else:
            print(f"   ❌ FAIL: {len(date_gaps)} gap temporali trovati")
            for gap in date_gaps[:3]:
                print(f"      Gap: {gap[0]} -> {gap[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRORE durante test allineamento: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔍 AUDIT COMPLETO PROPHET - Verifica periodi e visualizzazione")
    print("=" * 70)
    
    # Test 1: Verifica periodi di forecast
    test1_success = test_prophet_forecast_periods()[0]
    
    print("\n" + "=" * 70)
    
    # Test 2: Verifica allineamento dati
    test2_success = test_prophet_data_alignment()
    
    print("\n" + "=" * 70)
    print("🏁 RISULTATO FINALE AUDIT:")
    print(f"   Test periodi forecast: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"   Test allineamento dati: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 AUDIT COMPLETATO CON SUCCESSO!")
        print("   Prophet sembra funzionare correttamente per periodi e visualizzazione.")
    else:
        print("\n⚠️  AUDIT EVIDENZIA PROBLEMI!")
        print("   Prophet richiede correzioni per periodi o visualizzazione.")
