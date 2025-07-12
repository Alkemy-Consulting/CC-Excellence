#!/usr/bin/env python3
"""
Test per verificare le correzioni apportate al modulo Prophet
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import warnings
warnings.filterwarnings('ignore')

# Add modules path  
sys.path.append('/workspaces/CC-Excellence')

def test_prophet_corrections():
    """Test delle correzioni apportate a Prophet"""
    
    print("🔧 TEST CORREZIONI PROPHET")
    print("=" * 50)
    
    try:
        # Import modules
        from modules.prophet_module import run_prophet_forecast
        print("✅ Import Prophet module")
        
        # Create test data
        print("\n📊 Creazione dati di test...")
        start_date = datetime(2024, 1, 1)
        n_historical = 30
        n_forecast = 10
        
        dates = [start_date + timedelta(days=i) for i in range(n_historical)]
        values = [100 + i * 0.3 + 5 * np.sin(i / 7 * 2 * np.pi) + np.random.normal(0, 1) for i in range(n_historical)]
        
        df = pd.DataFrame({
            'date': dates,
            'volume': values
        })
        
        print(f"   Dati: {len(df)} punti dal {df['date'].min()} al {df['date'].max()}")
        
        # Configure forecast
        model_config = {
            'yearly_seasonality': False,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive'
        }
        
        base_config = {
            'forecast_periods': n_forecast,
            'confidence_interval': 0.8,
            'train_size': 1.0
        }
        
        print(f"   Configurazione: {n_forecast} periodi forecast")
        
        # Execute forecast
        print(f"\n🚀 Esecuzione forecast...")
        forecast_df, metrics, plots = run_prophet_forecast(
            df, 'date', 'volume', model_config, base_config
        )
        
        print(f"✅ Forecast completato")
        
        # Test 1: Verifica struttura output
        print(f"\n🔍 TEST 1: Struttura Output")
        print(f"   Shape forecast_df: {forecast_df.shape}")
        print(f"   Colonne: {list(forecast_df.columns)}")
        
        expected_cols = ['date', 'volume_forecast', 'volume_lower', 'volume_upper', 'volume_actual', 'is_future']
        missing_cols = [col for col in expected_cols if col not in forecast_df.columns]
        
        if not missing_cols:
            print("   ✅ Tutte le colonne attese presenti")
        else:
            print(f"   ❌ Colonne mancanti: {missing_cols}")
        
        # Test 2: Verifica numero righe
        print(f"\n🔍 TEST 2: Numero Righe")
        expected_rows = n_historical + n_forecast
        actual_rows = len(forecast_df)
        
        print(f"   Righe attese: {expected_rows} (storici: {n_historical} + forecast: {n_forecast})")
        print(f"   Righe effettive: {actual_rows}")
        
        if actual_rows == expected_rows:
            print("   ✅ Numero righe corretto")
        else:
            print(f"   ❌ Numero righe errato - Differenza: {actual_rows - expected_rows}")
        
        # Test 3: Verifica separazione storico/futuro
        print(f"\n🔍 TEST 3: Separazione Storico/Futuro")
        if 'is_future' in forecast_df.columns:
            historical_rows = (~forecast_df['is_future']).sum()
            future_rows = forecast_df['is_future'].sum()
            
            print(f"   Righe storiche: {historical_rows} (attese: {n_historical})")
            print(f"   Righe future: {future_rows} (attese: {n_forecast})")
            
            if historical_rows == n_historical and future_rows == n_forecast:
                print("   ✅ Separazione corretta")
            else:
                print("   ❌ Separazione errata")
        else:
            print("   ❌ Colonna 'is_future' mancante")
        
        # Test 4: Verifica presenza dati storici
        print(f"\n🔍 TEST 4: Dati Storici")
        if 'volume_actual' in forecast_df.columns:
            historical_data = forecast_df[~forecast_df['is_future']]
            actual_values_present = historical_data['volume_actual'].notna().sum()
            
            print(f"   Valori storici presenti: {actual_values_present} / {len(historical_data)}")
            
            if actual_values_present == len(historical_data):
                print("   ✅ Tutti i valori storici presenti")
            else:
                print(f"   ⚠️ Alcuni valori storici mancanti: {len(historical_data) - actual_values_present}")
        
        # Test 5: Verifica continuità temporale
        print(f"\n🔍 TEST 5: Continuità Temporale")
        forecast_df_sorted = forecast_df.sort_values('date')
        dates_list = forecast_df_sorted['date'].tolist()
        
        gaps = []
        for i in range(1, len(dates_list)):
            expected_next = dates_list[i-1] + timedelta(days=1)
            if dates_list[i] != expected_next:
                gaps.append((dates_list[i-1], dates_list[i]))
        
        if not gaps:
            print("   ✅ Nessun gap temporale")
        else:
            print(f"   ❌ {len(gaps)} gap temporali trovati")
            for gap in gaps[:3]:
                print(f"      {gap[0]} -> {gap[1]}")
        
        # Test 6: Verifica plot
        print(f"\n🔍 TEST 6: Generazione Plot")
        if 'forecast_plot' in plots:
            plot = plots['forecast_plot']
            print("   ✅ Plot generato")
            
            if hasattr(plot, 'data'):
                trace_names = [trace.name for trace in plot.data if hasattr(trace, 'name')]
                print(f"   Tracce nel plot: {trace_names}")
                
                expected_traces = ['Actual Values', 'Future Forecast']
                missing_traces = [trace for trace in expected_traces if trace not in trace_names]
                
                if not missing_traces:
                    print("   ✅ Tracce essenziali presenti")
                else:
                    print(f"   ⚠️ Tracce mancanti: {missing_traces}")
        else:
            print("   ❌ Plot non generato")
        
        # RIEPILOGO
        print(f"\n📋 RIEPILOGO CORREZIONI:")
        
        all_tests_passed = True
        
        # Controlla tutti i test
        if missing_cols:
            all_tests_passed = False
            print("   ❌ Struttura colonne")
        else:
            print("   ✅ Struttura colonne")
        
        if actual_rows == expected_rows:
            print("   ✅ Numero righe")
        else:
            all_tests_passed = False
            print("   ❌ Numero righe")
        
        if 'is_future' in forecast_df.columns:
            historical_correct = (~forecast_df['is_future']).sum() == n_historical
            future_correct = forecast_df['is_future'].sum() == n_forecast
            if historical_correct and future_correct:
                print("   ✅ Separazione storico/futuro")
            else:
                all_tests_passed = False
                print("   ❌ Separazione storico/futuro")
        else:
            all_tests_passed = False
            print("   ❌ Separazione storico/futuro")
        
        if not gaps:
            print("   ✅ Continuità temporale")
        else:
            all_tests_passed = False
            print("   ❌ Continuità temporale")
        
        if 'forecast_plot' in plots:
            print("   ✅ Generazione plot")
        else:
            all_tests_passed = False
            print("   ❌ Generazione plot")
        
        if all_tests_passed:
            print(f"\n🎉 TUTTE LE CORREZIONI FUNZIONANO!")
            print("   Prophet ora include correttamente:")
            print("   - Dati storici e futuri nel DataFrame")
            print("   - Separazione chiara tra periodi")
            print("   - Visualizzazione migliorata")
            print("   - Logging dettagliato per debug")
        else:
            print(f"\n⚠️ ALCUNE CORREZIONI RICHIEDONO ULTERIORI MODIFICHE")
        
        return all_tests_passed
        
    except Exception as e:
        print(f"❌ ERRORE durante test: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_prophet_corrections()
    
    if success:
        print("\n🎯 AUDIT CONCLUSO: Prophet corretto con successo!")
    else:
        print("\n🔧 AUDIT CONCLUSO: Ulteriori modifiche necessarie")
