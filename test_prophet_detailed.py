#!/usr/bin/env python3
"""
Test mirato per identificare problemi specifici nel forecast di Prophet
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add modules path
sys.path.append('/workspaces/CC-Excellence')

def test_prophet_forecast_detailed():
    """Test dettagliato per identificare problemi nel forecast Prophet"""
    
    print("🔍 TEST DETTAGLIATO PROPHET - Identificazione problemi")
    print("=" * 60)
    
    try:
        # Import Prophet module
        from modules.prophet_module import run_prophet_forecast
        print("✅ Import Prophet module successful")
        
        # Create test data - simple and predictable
        print("\n📊 Creazione dati di test...")
        start_date = datetime(2024, 1, 1)
        n_historical = 60  # 60 giorni storici
        n_forecast = 15    # 15 giorni forecast
        
        # Create historical data with simple pattern
        dates = [start_date + timedelta(days=i) for i in range(n_historical)]
        values = [100 + i * 0.5 + 5 * np.sin(i / 7 * 2 * np.pi) for i in range(n_historical)]
        
        df = pd.DataFrame({
            'date': dates,
            'calls': values
        })
        
        print(f"   - Dati storici: {len(df)} punti")
        print(f"   - Prima data: {df['date'].min()}")
        print(f"   - Ultima data: {df['date'].max()}")
        print(f"   - Range valori: {df['calls'].min():.1f} - {df['calls'].max():.1f}")
        
        # Expected forecast dates
        expected_forecast_start = df['date'].max() + timedelta(days=1)
        expected_forecast_end = expected_forecast_start + timedelta(days=n_forecast-1)
        
        print(f"   - Forecast atteso dal: {expected_forecast_start}")
        print(f"   - Forecast atteso al: {expected_forecast_end}")
        
        # Configure Prophet
        model_config = {
            'yearly_seasonality': False,  # Disable for simplicity
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'interval_width': 0.8
        }
        
        base_config = {
            'forecast_periods': n_forecast,
            'confidence_interval': 0.8,
            'train_size': 1.0  # Use all data for training
        }
        
        print(f"\n🔧 Configurazione Prophet:")
        print(f"   - Periodi forecast: {base_config['forecast_periods']}")
        print(f"   - Train size: {base_config['train_size']}")
        print(f"   - Confidence: {base_config['confidence_interval']}")
        
        # Execute Prophet forecast
        print(f"\n🚀 Esecuzione Prophet forecast...")
        forecast_df, metrics, plots = run_prophet_forecast(
            df, 'date', 'calls', model_config, base_config
        )
        
        print(f"✅ Forecast eseguito!")
        
        # DETAILED ANALYSIS OF RESULTS
        print(f"\n📋 ANALISI DETTAGLIATA RISULTATI:")
        print(f"   - Shape forecast_df: {forecast_df.shape}")
        print(f"   - Colonne: {list(forecast_df.columns)}")
        print(f"   - Metrics: {metrics}")
        print(f"   - Plots disponibili: {list(plots.keys()) if plots else 'Nessuno'}")
        
        if forecast_df.empty:
            print("❌ ERRORE CRITICO: forecast_df è vuoto!")
            return False
        
        # Check date range in forecast_df
        print(f"\n📅 ANALISI DATE NEL FORECAST:")
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        first_date = forecast_df['date'].min()
        last_date = forecast_df['date'].max()
        total_days = (last_date - first_date).days + 1
        
        print(f"   - Prima data forecast: {first_date}")
        print(f"   - Ultima data forecast: {last_date}")
        print(f"   - Totale giorni: {total_days}")
        print(f"   - Righe nel DataFrame: {len(forecast_df)}")
        
        # Expected total: historical + forecast periods
        expected_total = n_historical + n_forecast
        print(f"   - Righe attese: {expected_total} (storici: {n_historical} + forecast: {n_forecast})")
        
        if len(forecast_df) == expected_total:
            print("   ✅ NUMERO RIGHE CORRETTO")
        else:
            print(f"   ❌ NUMERO RIGHE ERRATO - Differenza: {len(forecast_df) - expected_total}")
        
        # Check if future dates are included
        future_data = forecast_df[forecast_df['date'] > df['date'].max()]
        print(f"\n🔮 ANALISI PERIODI FUTURI:")
        print(f"   - Righe con date future: {len(future_data)}")
        print(f"   - Righe future attese: {n_forecast}")
        
        if len(future_data) == n_forecast:
            print("   ✅ PERIODI FUTURI CORRETTI")
        else:
            print(f"   ❌ PERIODI FUTURI MANCANTI - Mancano: {n_forecast - len(future_data)} periodi")
        
        if len(future_data) > 0:
            future_start = future_data['date'].min()
            future_end = future_data['date'].max()
            print(f"   - Primo periodo futuro: {future_start}")
            print(f"   - Ultimo periodo futuro: {future_end}")
            
            # Check if future dates are consecutive
            future_dates_sorted = sorted(future_data['date'])
            gaps = []
            for i in range(1, len(future_dates_sorted)):
                expected_next = future_dates_sorted[i-1] + timedelta(days=1)
                if future_dates_sorted[i] != expected_next:
                    gaps.append((future_dates_sorted[i-1], future_dates_sorted[i]))
            
            if not gaps:
                print("   ✅ DATE FUTURE CONSECUTIVE")
            else:
                print(f"   ⚠️ GAP NELLE DATE FUTURE: {gaps}")
        
        # Check forecast values
        print(f"\n📊 ANALISI VALORI FORECAST:")
        if 'calls_forecast' in forecast_df.columns:
            forecast_col = 'calls_forecast'
        else:
            # Find the forecast column
            possible_cols = [col for col in forecast_df.columns if 'forecast' in col.lower()]
            if possible_cols:
                forecast_col = possible_cols[0]
                print(f"   - Usando colonna forecast: {forecast_col}")
            else:
                print("   ❌ NESSUNA COLONNA FORECAST TROVATA")
                print(f"   - Colonne disponibili: {list(forecast_df.columns)}")
                return False
        
        historical_forecast = forecast_df[forecast_df['date'] <= df['date'].max()]
        future_forecast = forecast_df[forecast_df['date'] > df['date'].max()]
        
        if len(historical_forecast) > 0:
            hist_mean = historical_forecast[forecast_col].mean()
            print(f"   - Media forecast storico: {hist_mean:.2f}")
        
        if len(future_forecast) > 0:
            future_mean = future_forecast[forecast_col].mean()
            future_min = future_forecast[forecast_col].min()
            future_max = future_forecast[forecast_col].max()
            
            print(f"   - Media forecast futuro: {future_mean:.2f}")
            print(f"   - Range forecast futuro: {future_min:.2f} - {future_max:.2f}")
            
            # Check if future values are reasonable
            original_mean = df['calls'].mean()
            print(f"   - Media originale: {original_mean:.2f}")
            
            ratio = future_mean / original_mean
            print(f"   - Rapporto future/original: {ratio:.2f}")
            
            if 0.5 <= ratio <= 2.0:
                print("   ✅ VALORI FUTURI RAGIONEVOLI")
            else:
                print("   ⚠️ VALORI FUTURI POTREBBERO ESSERE ANOMALI")
        
        # Check plot data
        print(f"\n🎨 ANALISI PLOT:")
        if 'forecast_plot' in plots:
            plot = plots['forecast_plot']
            print("   ✅ Plot di forecast generato")
            
            if hasattr(plot, 'data'):
                traces = plot.data
                trace_names = [trace.name for trace in traces if hasattr(trace, 'name')]
                print(f"   - Tracce nel plot: {trace_names}")
                
                # Find predictions trace
                pred_trace = None
                for trace in traces:
                    if hasattr(trace, 'name') and 'prediction' in trace.name.lower():
                        pred_trace = trace
                        break
                
                if pred_trace and hasattr(pred_trace, 'x'):
                    plot_dates = pd.to_datetime(pred_trace.x)
                    plot_start = plot_dates.min()
                    plot_end = plot_dates.max()
                    
                    print(f"   - Range date nel plot: {plot_start} - {plot_end}")
                    print(f"   - Punti nel plot: {len(plot_dates)}")
                    
                    # Check if plot includes future periods
                    future_in_plot = plot_dates[plot_dates > df['date'].max()]
                    print(f"   - Punti futuri nel plot: {len(future_in_plot)}")
                    
                    if len(future_in_plot) >= n_forecast:
                        print("   ✅ PLOT INCLUDE PERIODI FUTURI")
                    else:
                        print(f"   ❌ PLOT NON INCLUDE TUTTI I PERIODI FUTURI")
                        print(f"      Mancano: {n_forecast - len(future_in_plot)} periodi")
        else:
            print("   ❌ PLOT NON GENERATO")
        
        # SUMMARY
        print(f"\n📋 RIEPILOGO AUDIT PROPHET:")
        
        issues_found = []
        
        if len(forecast_df) != expected_total:
            issues_found.append("Numero righe forecast non corretto")
        
        if len(future_data) != n_forecast:
            issues_found.append("Periodi futuri mancanti")
        
        if 'forecast_plot' not in plots:
            issues_found.append("Plot non generato")
        
        if not issues_found:
            print("   🎉 NESSUN PROBLEMA TROVATO - Prophet funziona correttamente!")
            return True
        else:
            print("   ⚠️ PROBLEMI IDENTIFICATI:")
            for issue in issues_found:
                print(f"      - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ ERRORE durante test: {str(e)}")
        import traceback
        print(f"Traceback completo:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_prophet_forecast_detailed()
    
    if success:
        print("\n🎯 CONCLUSIONE: Prophet sembra funzionare correttamente")
    else:
        print("\n🚨 CONCLUSIONE: Prophet ha problemi che richiedono correzioni")
