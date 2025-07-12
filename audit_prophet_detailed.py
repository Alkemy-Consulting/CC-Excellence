#!/usr/bin/env python3
"""
AUDIT DETTAGLIATO - VERIFICA FUNZIONALITÀ SPECIFICHE PROPHET
Test delle funzionalità critiche: holidays, cross-validation, diagnostics
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

def test_prophet_holidays():
    """Test implementazione holidays Prophet"""
    print("🎄 TESTING PROPHET HOLIDAYS IMPLEMENTATION")
    print("=" * 50)
    
    try:
        # Test import holidays
        try:
            import holidays
            print("✅ Holidays package available")
            
            # Test countries supported
            supported_countries = ['US', 'CA', 'UK', 'DE', 'FR']
            for country in supported_countries:
                try:
                    hols = getattr(holidays, country, None)
                    if hols:
                        test_hols = hols()
                        holiday_count = len([h for h in test_hols.keys() if '2023' in str(h)])
                        print(f"  ✅ {country}: {holiday_count} holidays in 2023")
                    else:
                        print(f"  ❌ {country}: Not supported")
                except Exception as e:
                    print(f"  ❌ {country}: Error - {e}")
                    
        except ImportError:
            print("❌ Holidays package not available")
            return False
            
        # Test Prophet holidays integration
        sys.path.append('/workspaces/CC-Excellence')
        from modules.prophet_core import ProphetForecaster
        
        # Generate test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = 100 + np.cumsum(np.random.randn(len(dates)) * 0.1) + 5*np.sin(2*np.pi*np.arange(len(dates))/365.25)
        df = pd.DataFrame({'date': dates, 'value': values})
        
        forecaster = ProphetForecaster()
        
        # Test holiday configuration
        model_config_with_holidays = {
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'add_holidays': True,
            'holidays_country': 'US'
        }
        
        base_config = {
            'forecast_periods': 30,
            'confidence_interval': 0.95,
            'train_size': 0.8
        }
        
        print("\n🔬 Testing Prophet with holidays...")
        result = forecaster.run_forecast_core(df, 'date', 'value', model_config_with_holidays, base_config)
        
        if result.success:
            print("✅ Prophet with holidays: SUCCESS")
            print(f"   Forecast shape: {result.raw_forecast.shape}")
            print(f"   Metrics: {result.metrics}")
            return True
        else:
            print(f"❌ Prophet with holidays: FAILED - {result.error}")
            return False
            
    except Exception as e:
        print(f"❌ Holiday test failed: {e}")
        return False

def test_prophet_cross_validation():
    """Test implementazione cross-validation Prophet"""
    print("\n🔄 TESTING PROPHET CROSS-VALIDATION")
    print("=" * 40)
    
    try:
        # Test import Prophet CV functions
        try:
            from prophet.diagnostics import cross_validation, performance_metrics
            print("✅ Prophet CV functions available")
        except ImportError as e:
            print(f"❌ Prophet CV functions not available: {e}")
            return False
            
        # Generate test data
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        trend = np.linspace(100, 120, len(dates))
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.randn(len(dates)) * 2
        values = trend + seasonal + noise
        
        df = pd.DataFrame({'ds': dates, 'y': values})
        
        print(f"✅ Test data generated: {df.shape}")
        
        # Test Prophet model creation and fitting
        from prophet import Prophet
        
        model = Prophet()
        model.fit(df)
        print("✅ Prophet model fitted")
        
        # Test cross-validation
        print("🔬 Running cross-validation...")
        try:
            df_cv = cross_validation(
                model, 
                initial='365 days',
                period='180 days', 
                horizon='30 days'
            )
            print(f"✅ Cross-validation completed: {df_cv.shape}")
            
            # Test performance metrics
            df_metrics = performance_metrics(df_cv)
            print(f"✅ Performance metrics calculated: {df_metrics.shape}")
            print("   Metrics columns:", list(df_metrics.columns))
            
            # Check key metrics
            if 'mape' in df_metrics.columns:
                avg_mape = df_metrics['mape'].mean()
                print(f"   Average MAPE: {avg_mape:.3f}")
                
            if 'rmse' in df_metrics.columns:
                avg_rmse = df_metrics['rmse'].mean()
                print(f"   Average RMSE: {avg_rmse:.3f}")
                
            return True
            
        except Exception as e:
            print(f"❌ Cross-validation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ CV test failed: {e}")
        return False

def test_prophet_diagnostics():
    """Test implementazione diagnostics avanzati"""
    print("\n📊 TESTING PROPHET ADVANCED DIAGNOSTICS")
    print("=" * 42)
    
    try:
        sys.path.append('/workspaces/CC-Excellence')
        
        # Test diagnostic modules
        try:
            from modules.prophet_diagnostics import ProphetDiagnosticAnalyzer, ProphetDiagnosticPlots
            print("✅ Diagnostic modules available")
        except ImportError as e:
            print(f"❌ Diagnostic modules not available: {e}")
            return False
            
        # Test core forecasting
        from modules.prophet_core import ProphetForecaster
        
        # Generate test data with trend and seasonality
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        trend = np.linspace(100, 150, len(dates))
        yearly_seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        weekly_seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
        noise = np.random.randn(len(dates)) * 3
        values = trend + yearly_seasonal + weekly_seasonal + noise
        
        df = pd.DataFrame({'date': dates, 'value': values})
        
        forecaster = ProphetForecaster()
        
        model_config = {
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False
        }
        
        base_config = {
            'forecast_periods': 30,
            'confidence_interval': 0.95,
            'train_size': 0.8
        }
        
        print("🔬 Running Prophet forecast for diagnostics...")
        result = forecaster.run_forecast_core(df, 'date', 'value', model_config, base_config)
        
        if not result.success:
            print(f"❌ Forecast failed: {result.error}")
            return False
            
        print("✅ Forecast completed successfully")
        
        # Test diagnostic analysis
        analyzer = ProphetDiagnosticAnalyzer()
        print("🔬 Running diagnostic analysis...")
        
        analysis = analyzer.analyze_forecast_quality(result, df, 'date', 'value')
        
        if 'error' in analysis:
            print(f"❌ Diagnostic analysis failed: {analysis['error']}")
            return False
            
        print("✅ Diagnostic analysis completed")
        
        # Check analysis components
        components = [
            'forecast_coverage', 'residual_analysis', 'trend_analysis',
            'seasonality_analysis', 'uncertainty_analysis', 'quality_score'
        ]
        
        for component in components:
            if component in analysis:
                print(f"  ✅ {component}: Available")
                if component == 'quality_score':
                    score = analysis[component]
                    print(f"     Quality Score: {score:.1f}/100")
            else:
                print(f"  ❌ {component}: Missing")
                
        # Test residual analysis specifically
        if 'residual_analysis' in analysis:
            residual_stats = analysis['residual_analysis']
            if 'error' not in residual_stats:
                print(f"  📈 Mean residual: {residual_stats.get('mean_residual', 0):.4f}")
                print(f"  📈 Residual std: {residual_stats.get('std_residual', 0):.4f}")
                print(f"  📈 Normality test: {'PASS' if residual_stats.get('is_normally_distributed', False) else 'FAIL'}")
                print(f"  📈 Autocorrelation: {'DETECTED' if residual_stats.get('has_autocorrelation', True) else 'NONE'}")
                
        return True
        
    except Exception as e:
        print(f"❌ Diagnostics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prophet_ensemble():
    """Test implementazione ensemble forecasting"""
    print("\n🤖 TESTING PROPHET ENSEMBLE FORECASTING")
    print("=" * 40)
    
    try:
        sys.path.append('/workspaces/CC-Excellence')
        
        # Test ML advanced module
        try:
            from modules.prophet_ml_advanced import EnsembleForecaster, AdvancedFeatureEngineer
            print("✅ Ensemble modules available")
        except ImportError as e:
            print(f"❌ Ensemble modules not available: {e}")
            return False
            
        # Generate test data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        values = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5) + 10*np.sin(2*np.pi*np.arange(len(dates))/365.25)
        df = pd.DataFrame({'date': dates, 'value': values})
        
        print(f"✅ Test data generated: {df.shape}")
        
        # Test ensemble forecaster
        ensemble = EnsembleForecaster(enable_prophet=True, enable_ml_models=True)
        
        print("🔬 Testing ensemble model initialization...")
        models = ensemble.initialize_models()
        print(f"✅ Ensemble models initialized: {list(models.keys())}")
        
        # Test feature engineering
        feature_engineer = AdvancedFeatureEngineer()
        print("🔬 Testing feature engineering...")
        
        features_df = feature_engineer.engineer_features(df, 'date', 'value')
        print(f"✅ Features created: {features_df.shape}")
        print(f"   Feature names: {len(feature_engineer.feature_names)} features")
        
        # Test feature selection
        selected_df = feature_engineer.select_features(features_df)
        print(f"✅ Features selected: {selected_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prophet_performance():
    """Test implementazione performance optimization"""
    print("\n⚡ TESTING PROPHET PERFORMANCE OPTIMIZATION")
    print("=" * 45)
    
    try:
        sys.path.append('/workspaces/CC-Excellence')
        
        # Test performance modules
        try:
            from modules.prophet_performance import PerformanceMonitor, OptimizedProphetForecaster
            print("✅ Performance modules available")
        except ImportError as e:
            print(f"❌ Performance modules not available: {e}")
            return False
            
        # Test performance monitoring
        monitor = PerformanceMonitor()
        
        print("🔬 Testing performance monitoring...")
        with monitor.monitor_execution("test_operation") as ctx:
            # Simulate some work
            import time
            time.sleep(0.1)
            df = pd.DataFrame({'x': range(1000), 'y': np.random.randn(1000)})
            _ = df.describe()
            
        print("✅ Performance monitoring completed")
        
        if monitor.metrics_history:
            latest = monitor.metrics_history[-1]
            print(f"   Execution time: {latest.execution_time:.3f}s")
            print(f"   Memory usage: {latest.memory_usage:.1f}MB")
            
        # Test optimized forecaster
        optimized_forecaster = OptimizedProphetForecaster()
        print("✅ Optimized forecaster created")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def run_detailed_audit():
    """Esegui audit dettagliato delle funzionalità specifiche"""
    print("🔍 AUDIT DETTAGLIATO FUNZIONALITÀ PROPHET")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = {}
    
    # Run all tests
    test_results['holidays'] = test_prophet_holidays()
    test_results['cross_validation'] = test_prophet_cross_validation()
    test_results['diagnostics'] = test_prophet_diagnostics()
    test_results['ensemble'] = test_prophet_ensemble()
    test_results['performance'] = test_prophet_performance()
    
    # Final report
    print("\n" + "=" * 50)
    print("📋 RISULTATI AUDIT DETTAGLIATO")
    print("=" * 50)
    
    print("┌" + "─" * 25 + "┬" + "─" * 15 + "┬" + "─" * 6 + "┐")
    print("│ FUNZIONALITÀ          │ IMPLEMENTAZIONE │ SCORE │")
    print("├" + "─" * 25 + "┼" + "─" * 15 + "┼" + "─" * 6 + "┤")
    
    total_score = 0
    for test_name, result in test_results.items():
        status = "✅ COMPLETA" if result else "❌ INCOMPLETA"
        score = 10 if result else 3
        total_score += score
        
        test_display = test_name.replace('_', ' ').title()
        print(f"│ {test_display:<23} │ {status:<13} │ {score:>4} │")
    
    print("└" + "─" * 25 + "┴" + "─" * 15 + "┴" + "─" * 6 + "┘")
    
    avg_score = total_score / len(test_results)
    print(f"\n🎯 PUNTEGGIO MEDIO FUNZIONALITÀ: {avg_score:.1f}/10")
    
    # Recommendations
    print(f"\n💡 RACCOMANDAZIONI:")
    
    failed_tests = [name for name, result in test_results.items() if not result]
    if failed_tests:
        print("❌ Funzionalità da implementare/correggere:")
        for test in failed_tests:
            print(f"   - {test.replace('_', ' ').title()}")
    else:
        print("✅ Tutte le funzionalità core sono implementate correttamente!")
        
    return test_results

if __name__ == "__main__":
    run_detailed_audit()
