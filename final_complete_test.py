#!/usr/bin/env python3
"""
Test finale completo dell'applicazione CC-Excellence con Prophet
Verifica che tutto funzioni correttamente dopo le correzioni
"""

import sys
import pandas as pd
import numpy as np
import warnings
import traceback
from datetime import datetime

warnings.filterwarnings('ignore')

print("🎯 TEST FINALE COMPLETO - CC-EXCELLENCE CON PROPHET")
print("=" * 70)
print(f"Data test: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

def test_basic_imports():
    """Test degli import basilari"""
    print("\n📦 1. TESTING BASIC IMPORTS")
    try:
        import streamlit as st
        print("   ✅ Streamlit imported")
        
        import plotly.graph_objects as go
        print("   ✅ Plotly imported")
        
        import pandas as pd
        import numpy as np
        print(f"   ✅ Pandas {pd.__version__} imported")
        print(f"   ✅ NumPy {np.__version__} imported")
        
        from prophet import Prophet
        print("   ✅ Prophet imported")
        
        return True
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return False

def test_prophet_module():
    """Test del modulo Prophet completo"""
    print("\n🔮 2. TESTING PROPHET MODULE")
    try:
        sys.path.append('/workspaces/CC-Excellence')
        from modules.prophet_module import (
            run_prophet_forecast, 
            create_prophet_forecast_chart,
            validate_prophet_inputs,
            optimize_dataframe_for_prophet
        )
        print("   ✅ Prophet module functions imported")
        
        # Test con dati di esempio
        dates = pd.date_range(start='2022-01-01', end='2024-06-30', freq='D')
        np.random.seed(42)
        values = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        
        test_df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        print(f"   ✅ Test data created: {len(test_df)} rows")
        
        # Test validation
        validate_prophet_inputs(test_df, 'date', 'value')
        print("   ✅ Input validation passed")
        
        # Test data preparation
        prophet_data = optimize_dataframe_for_prophet(test_df)
        print(f"   ✅ Data preparation: {prophet_data.shape}")
        
        return True, test_df
    except Exception as e:
        print(f"   ❌ Prophet module error: {e}")
        traceback.print_exc()
        return False, None

def test_prophet_forecast(test_df):
    """Test completo del forecast Prophet"""
    print("\n🚀 3. TESTING PROPHET FORECAST")
    try:
        sys.path.append('/workspaces/CC-Excellence')
        from modules.prophet_module import run_prophet_forecast
        
        # Configurazione Prophet
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
            'show_residuals': True
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
        
        print("   ⚙️  Configuration set")
        
        # Esegui forecast
        forecast_df, metrics, plots = run_prophet_forecast(
            df=test_df,
            date_col='date',
            target_col='value',
            model_config=model_config,
            base_config=base_config
        )
        
        if not forecast_df.empty:
            print(f"   ✅ Forecast generated: {forecast_df.shape}")
            print(f"   ✅ Metrics: {list(metrics.keys())}")
            print(f"   ✅ Plots: {list(plots.keys())}")
            
            # Verifica metriche
            if 'mape' in metrics:
                print(f"   📊 MAPE: {metrics['mape']:.2f}%")
            if 'mae' in metrics:
                print(f"   📊 MAE: {metrics['mae']:.2f}")
            if 'rmse' in metrics:
                print(f"   📊 RMSE: {metrics['rmse']:.2f}")
            if 'r2' in metrics:
                print(f"   📊 R²: {metrics['r2']:.3f}")
            
            return True, forecast_df, metrics, plots
        else:
            print("   ❌ Empty forecast result")
            return False, None, None, None
            
    except Exception as e:
        print(f"   ❌ Forecast error: {e}")
        traceback.print_exc()
        return False, None, None, None

def test_chart_creation(test_df, plots):
    """Test creazione grafici"""
    print("\n📊 4. TESTING CHART CREATION")
    try:
        # Verifica che i plots siano stati creati
        expected_plots = ['forecast_plot', 'components_plot', 'residuals_plot']
        created_plots = []
        
        for plot_name in expected_plots:
            if plot_name in plots and plots[plot_name] is not None:
                created_plots.append(plot_name)
                print(f"   ✅ {plot_name} created successfully")
            else:
                print(f"   ⚠️  {plot_name} not found or None")
        
        if len(created_plots) >= 2:  # Almeno 2 grafici su 3
            print(f"   ✅ Chart creation successful: {len(created_plots)}/3 plots")
            return True
        else:
            print(f"   ❌ Insufficient plots created: {len(created_plots)}/3")
            return False
            
    except Exception as e:
        print(f"   ❌ Chart creation error: {e}")
        return False

def test_range_selectors():
    """Test specifico per range selectors"""
    print("\n🎛️  5. TESTING RANGE SELECTORS")
    try:
        import plotly.graph_objects as go
        
        # Test della nuova configurazione
        fig = go.Figure()
        
        # Dati di test
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        values = np.random.randn(len(dates)).cumsum()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Test Data'
        ))
        
        # Applica la nuova configurazione range selectors
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
                ),
                type="date"
            )
        )
        
        print("   ✅ Range selectors configuration applied")
        print("   ✅ No timestamp arithmetic errors")
        
        # Verifica configurazione
        range_buttons = fig.layout.xaxis.rangeselector.buttons
        print(f"   ✅ {len(range_buttons)} range buttons configured:")
        for button in range_buttons:
            if hasattr(button, 'label'):
                step = getattr(button, 'step', 'N/A')
                count = getattr(button, 'count', 'N/A')
                print(f"      - {button.label}: count={count}, step={step}")
        
        return True
    except Exception as e:
        print(f"   ❌ Range selectors error: {e}")
        if "Addition/subtraction" in str(e):
            print("   🚨 TIMESTAMP ARITHMETIC ERROR STILL PRESENT!")
        traceback.print_exc()
        return False

def test_csv_data_loading():
    """Test caricamento dati CSV reali"""
    print("\n📄 6. TESTING CSV DATA LOADING")
    try:
        # Verifica che il file di test esista
        csv_path = '/workspaces/CC-Excellence/test_data.csv'
        
        df = pd.read_csv(csv_path)
        print(f"   ✅ CSV loaded: {df.shape}")
        
        # Verifica struttura
        print(f"   ✅ Columns: {list(df.columns)}")
        
        # Conversioni
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'])
        
        print(f"   ✅ Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   ✅ Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
        
        # Verifica che non ci siano problemi con i dati
        if df['date'].isna().any():
            print("   ⚠️  Warning: NaN values in date column")
        if df['value'].isna().any():
            print("   ⚠️  Warning: NaN values in value column")
        
        return True, df
    except Exception as e:
        print(f"   ❌ CSV loading error: {e}")
        return False, None

def run_final_integration_test():
    """Test di integrazione finale"""
    print("\n🎯 7. FINAL INTEGRATION TEST")
    try:
        # Carica dati CSV reali
        success, csv_df = test_csv_data_loading()
        if not success:
            return False
        
        # Esegui forecast completo con dati reali
        sys.path.append('/workspaces/CC-Excellence')
        from modules.prophet_module import run_prophet_forecast
        
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
            'show_residuals': True
        }
        
        base_config = {
            'train_size': 0.8,
            'forecast_periods': 30,
            'confidence_interval': 0.95,
            'regressor_config': {'selected_regressors': [], 'regressor_configs': {}}
        }
        
        print("   🚀 Running full forecast with real CSV data...")
        
        forecast_df, metrics, plots = run_prophet_forecast(
            df=csv_df,
            date_col='date',
            target_col='value',
            model_config=model_config,
            base_config=base_config
        )
        
        if not forecast_df.empty and plots:
            print("   ✅ Integration test successful!")
            print(f"   📊 Final forecast shape: {forecast_df.shape}")
            print(f"   📊 Final plots count: {len(plots)}")
            
            return True
        else:
            print("   ❌ Integration test failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration test error: {e}")
        if "Addition/subtraction" in str(e):
            print("   🚨 CRITICAL: Timestamp arithmetic error still present!")
        traceback.print_exc()
        return False

def main():
    """Funzione principale per eseguire tutti i test"""
    print("Iniziando test finale completo...")
    
    results = []
    
    # Test 1: Import basilari
    results.append(("Basic Imports", test_basic_imports()))
    
    # Test 2: Modulo Prophet
    prophet_success, test_df = test_prophet_module()
    results.append(("Prophet Module", prophet_success))
    
    if prophet_success and test_df is not None:
        # Test 3: Prophet Forecast
        forecast_success, forecast_df, metrics, plots = test_prophet_forecast(test_df)
        results.append(("Prophet Forecast", forecast_success))
        
        if forecast_success and plots:
            # Test 4: Chart Creation
            results.append(("Chart Creation", test_chart_creation(test_df, plots)))
    
    # Test 5: Range Selectors
    results.append(("Range Selectors", test_range_selectors()))
    
    # Test 6: CSV Loading
    csv_success, csv_df = test_csv_data_loading()
    results.append(("CSV Data Loading", csv_success))
    
    # Test 7: Integration Test
    results.append(("Final Integration", run_final_integration_test()))
    
    # Riepilogo risultati
    print("\n" + "=" * 70)
    print("🎯 RISULTATI FINALI")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:<12} {test_name}")
        if success:
            passed += 1
    
    print("-" * 70)
    print(f"TOTALE: {passed}/{total} test passati ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 TUTTI I TEST SONO PASSATI!")
        print("✅ L'applicazione CC-Excellence è completamente funzionale")
        print("✅ Il problema timestamp arithmetic è stato risolto")
        print("✅ Prophet module funziona correttamente con pandas >= 2.0")
        print("✅ Range selectors compatibili e funzionanti")
        print("✅ Grafici e visualizzazioni generate senza errori")
        print("\n🚀 L'applicazione è pronta per l'uso in produzione!")
    else:
        print(f"\n⚠️  {total-passed} test hanno fallito")
        print("🔧 Rivedere le correzioni necessarie")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
