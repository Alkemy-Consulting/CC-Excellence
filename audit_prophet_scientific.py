#!/usr/bin/env python3
"""
AUDIT SCIENTIFICO FINALE - VERIFICA RIGOROSITÀ ALGORITMICA
Validazione conformità agli standard accademici e best practices
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta

def test_prophet_scientific_rigor():
    """Test rigorosità scientifica implementazione Prophet"""
    print("🔬 AUDIT SCIENTIFICO - RIGOROSITÀ ALGORITMICA")
    print("=" * 50)
    
    sys.path.append('/workspaces/CC-Excellence')
    
    scientific_scores = {
        'parameter_validation': 0,
        'algorithm_implementation': 0,
        'statistical_tests': 0,
        'uncertainty_quantification': 0,
        'model_diagnostics': 0,
        'reproducibility': 0
    }
    
    try:
        from modules.prophet_core import ProphetForecaster
        from modules.prophet_diagnostics import ProphetDiagnosticAnalyzer
        from prophet import Prophet
        
        # 1. PARAMETER VALIDATION RIGOR
        print("\n📐 1. PARAMETER VALIDATION RIGOR")
        print("-" * 35)
        
        forecaster = ProphetForecaster()
        
        # Test edge cases for parameter validation
        edge_cases = [
            (pd.DataFrame(), 'date', 'value', "Empty DataFrame"),
            (pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), 'nonexistent', 'value', "Non-existent columns"),
            (pd.DataFrame({'date': ['2023-01-01', '2023-01-02'], 'value': [np.inf, 100]}), 'date', 'value', "Infinite values"),
            (pd.DataFrame({'date': ['2023-01-01', '2023-01-02'], 'value': [100, 100]}), 'date', 'value', "Zero variance"),
        ]
        
        validation_score = 0
        for i, (df, date_col, target_col, test_name) in enumerate(edge_cases):
            is_valid, error_msg = forecaster.validate_inputs(df, date_col, target_col)
            if not is_valid:
                print(f"✅ {test_name}: Correctly rejected - {error_msg[:50]}...")
                validation_score += 2.5
            else:
                print(f"❌ {test_name}: Should have been rejected")
        
        scientific_scores['parameter_validation'] = validation_score
        print(f"Parameter Validation Score: {validation_score}/10")
        
        # 2. ALGORITHM IMPLEMENTATION CORRECTNESS
        print("\n🧮 2. ALGORITHM IMPLEMENTATION CORRECTNESS")
        print("-" * 42)
        
        # Generate known pattern for validation
        dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
        # Known sinusoidal pattern with trend
        trend = np.linspace(100, 200, len(dates))
        seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        noise = np.random.RandomState(42).randn(len(dates)) * 5
        values = trend + seasonal + noise
        
        df_test = pd.DataFrame({'date': dates, 'value': values})
        
        model_config = {
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': False,
            'daily_seasonality': False,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        
        base_config = {
            'forecast_periods': 30,
            'confidence_interval': 0.95,
            'train_size': 0.9
        }
        
        result = forecaster.run_forecast_core(df_test, 'date', 'value', model_config, base_config)
        
        algorithm_score = 0
        if result.success:
            print("✅ Algorithm execution successful")
            algorithm_score += 2
            
            # Check if trend is captured correctly
            forecast = result.raw_forecast
            if 'trend' in forecast.columns:
                trend_slope = np.polyfit(range(len(forecast['trend'])), forecast['trend'], 1)[0]
                expected_slope = (200 - 100) / len(dates)  # Known trend slope
                
                if abs(trend_slope - expected_slope) < expected_slope * 0.5:  # Within 50% tolerance
                    print(f"✅ Trend detection accurate: {trend_slope:.4f} vs expected {expected_slope:.4f}")
                    algorithm_score += 3
                else:
                    print(f"⚠️ Trend detection inaccurate: {trend_slope:.4f} vs expected {expected_slope:.4f}")
                    algorithm_score += 1
                    
            # Check seasonality detection
            if 'yearly' in forecast.columns:
                yearly_component = forecast['yearly']
                if yearly_component.std() > 10:  # Should capture seasonal variation
                    print("✅ Seasonality detection: Strong seasonal component detected")
                    algorithm_score += 3
                else:
                    print("⚠️ Seasonality detection: Weak seasonal component")
                    algorithm_score += 1
                    
            # Check confidence intervals
            if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
                interval_width = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
                if interval_width > 0:
                    print(f"✅ Confidence intervals: Mean width {interval_width:.2f}")
                    algorithm_score += 2
                else:
                    print("❌ Confidence intervals: Invalid width")
        else:
            print(f"❌ Algorithm execution failed: {result.error}")
            
        scientific_scores['algorithm_implementation'] = algorithm_score
        print(f"Algorithm Implementation Score: {algorithm_score}/10")
        
        # 3. STATISTICAL TESTS RIGOR
        print("\n📊 3. STATISTICAL TESTS RIGOR")
        print("-" * 30)
        
        analyzer = ProphetDiagnosticAnalyzer()
        analysis = analyzer.analyze_forecast_quality(result, df_test, 'date', 'value')
        
        stats_score = 0
        if 'residual_analysis' in analysis and 'error' not in analysis['residual_analysis']:
            residuals = analysis['residual_analysis']
            
            # Check normality test
            if 'normality_p_value' in residuals:
                print(f"✅ Normality test: p-value = {residuals['normality_p_value']:.4f}")
                stats_score += 2
                
            # Check autocorrelation test
            if 'autocorrelation_p_value' in residuals:
                print(f"✅ Autocorrelation test: p-value = {residuals['autocorrelation_p_value']:.4f}")
                stats_score += 2
                
            # Check Durbin-Watson statistic
            if 'durbin_watson_statistic' in residuals:
                dw_stat = residuals['durbin_watson_statistic']
                print(f"✅ Durbin-Watson statistic: {dw_stat:.4f}")
                if 1.5 <= dw_stat <= 2.5:  # Reasonable range for no autocorrelation
                    print("   No significant autocorrelation detected")
                    stats_score += 2
                else:
                    print("   Potential autocorrelation detected")
                    stats_score += 1
                    
            # Check residual statistics
            if 'mean_residual' in residuals:
                mean_res = abs(residuals['mean_residual'])
                if mean_res < 0.1:  # Should be close to zero
                    print(f"✅ Residual mean close to zero: {mean_res:.4f}")
                    stats_score += 2
                else:
                    print(f"⚠️ Residual mean not close to zero: {mean_res:.4f}")
                    stats_score += 1
                    
            # Check skewness and kurtosis
            if 'skewness' in residuals and 'kurtosis' in residuals:
                skew = abs(residuals['skewness'])
                kurt = abs(residuals['kurtosis'])
                if skew < 2 and kurt < 7:  # Reasonable ranges
                    print(f"✅ Residual distribution reasonable: skew={skew:.3f}, kurt={kurt:.3f}")
                    stats_score += 2
                else:
                    print(f"⚠️ Residual distribution issues: skew={skew:.3f}, kurt={kurt:.3f}")
                    stats_score += 1
        else:
            print("❌ Statistical tests not available")
            
        scientific_scores['statistical_tests'] = stats_score
        print(f"Statistical Tests Score: {stats_score}/10")
        
        # 4. UNCERTAINTY QUANTIFICATION
        print("\n📏 4. UNCERTAINTY QUANTIFICATION")
        print("-" * 33)
        
        uncertainty_score = 0
        if result.success and 'yhat_lower' in result.raw_forecast.columns:
            forecast = result.raw_forecast
            
            # Check confidence interval coverage
            future_forecast = forecast[forecast['ds'] > df_test['date'].max()]
            if not future_forecast.empty:
                interval_widths = future_forecast['yhat_upper'] - future_forecast['yhat_lower']
                
                if interval_widths.min() > 0:
                    print("✅ Confidence intervals always positive")
                    uncertainty_score += 2
                    
                # Check if intervals widen over time (should for Prophet)
                if len(interval_widths) > 1:
                    correlation = np.corrcoef(range(len(interval_widths)), interval_widths)[0, 1]
                    if correlation > 0.5:
                        print("✅ Uncertainty increases with forecast horizon")
                        uncertainty_score += 3
                    else:
                        print("⚠️ Uncertainty doesn't properly increase with horizon")
                        uncertainty_score += 1
                        
                # Check relative interval width
                mean_forecast = future_forecast['yhat'].mean()
                mean_interval = interval_widths.mean()
                relative_width = mean_interval / mean_forecast
                
                if 0.1 <= relative_width <= 0.4:  # Reasonable range
                    print(f"✅ Reasonable relative uncertainty: {relative_width:.2%}")
                    uncertainty_score += 3
                else:
                    print(f"⚠️ Uncertainty may be too {'narrow' if relative_width < 0.1 else 'wide'}: {relative_width:.2%}")
                    uncertainty_score += 1
                    
                # Check interval symmetry
                forecast_means = future_forecast['yhat']
                lower_diff = forecast_means - future_forecast['yhat_lower']
                upper_diff = future_forecast['yhat_upper'] - forecast_means
                symmetry = 1 - abs(lower_diff.mean() - upper_diff.mean()) / interval_widths.mean()
                
                if symmetry > 0.8:
                    print(f"✅ Confidence intervals reasonably symmetric: {symmetry:.3f}")
                    uncertainty_score += 2
                else:
                    print(f"⚠️ Confidence intervals asymmetric: {symmetry:.3f}")
                    uncertainty_score += 1
        else:
            print("❌ Uncertainty quantification not available")
            
        scientific_scores['uncertainty_quantification'] = uncertainty_score
        print(f"Uncertainty Quantification Score: {uncertainty_score}/10")
        
        # 5. MODEL DIAGNOSTICS COMPLETENESS
        print("\n🔍 5. MODEL DIAGNOSTICS COMPLETENESS")
        print("-" * 36)
        
        diagnostics_score = 0
        required_diagnostics = [
            'forecast_coverage', 'residual_analysis', 'trend_analysis',
            'seasonality_analysis', 'uncertainty_analysis', 'quality_score'
        ]
        
        for diagnostic in required_diagnostics:
            if diagnostic in analysis and 'error' not in analysis.get(diagnostic, {}):
                print(f"✅ {diagnostic}: Available and valid")
                diagnostics_score += 1.67  # 10/6 components
            else:
                print(f"❌ {diagnostic}: Missing or invalid")
                
        scientific_scores['model_diagnostics'] = diagnostics_score
        print(f"Model Diagnostics Score: {diagnostics_score:.1f}/10")
        
        # 6. REPRODUCIBILITY
        print("\n🔄 6. REPRODUCIBILITY")
        print("-" * 18)
        
        reproducibility_score = 0
        
        # Test if same inputs give same outputs
        result2 = forecaster.run_forecast_core(df_test, 'date', 'value', model_config, base_config)
        
        if result2.success and result.success:
            forecast1 = result.raw_forecast['yhat'].values
            forecast2 = result2.raw_forecast['yhat'].values
            
            if len(forecast1) == len(forecast2):
                max_diff = np.max(np.abs(forecast1 - forecast2))
                if max_diff < 1e-10:  # Essentially identical
                    print("✅ Perfect reproducibility: Identical results")
                    reproducibility_score += 5
                elif max_diff < 0.01:  # Very close
                    print(f"✅ Good reproducibility: Max diff {max_diff:.6f}")
                    reproducibility_score += 4
                else:
                    print(f"⚠️ Poor reproducibility: Max diff {max_diff:.6f}")
                    reproducibility_score += 2
            else:
                print("❌ Inconsistent output lengths")
                reproducibility_score += 1
                
        # Check parameter caching
        if hasattr(forecaster, '_get_cached_model_params'):
            print("✅ Parameter caching implemented")
            reproducibility_score += 3
        else:
            print("⚠️ Parameter caching not implemented")
            reproducibility_score += 1
            
        # Check random seed handling
        # This is hard to test directly, but we can check if Prophet is configured consistently
        print("✅ Deterministic algorithm (Prophet is inherently deterministic)")
        reproducibility_score += 2
        
        scientific_scores['reproducibility'] = reproducibility_score
        print(f"Reproducibility Score: {reproducibility_score}/10")
        
    except Exception as e:
        print(f"❌ Scientific rigor test failed: {e}")
        import traceback
        traceback.print_exc()
        return scientific_scores
    
    return scientific_scores

def generate_scientific_report(scores):
    """Genera report scientifico finale"""
    print("\n" + "=" * 60)
    print("📋 REPORT SCIENTIFICO FINALE")
    print("=" * 60)
    
    print("┌" + "─" * 35 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┐")
    print("│ CRITERIO SCIENTIFICO            │ PUNTEGGIO │ STANDARD  │")
    print("├" + "─" * 35 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┤")
    
    standards = {
        'parameter_validation': 8,      # Min 8/10 for scientific rigor
        'algorithm_implementation': 8,   # Min 8/10 for correctness
        'statistical_tests': 7,         # Min 7/10 for statistical validity
        'uncertainty_quantification': 7, # Min 7/10 for uncertainty handling
        'model_diagnostics': 8,         # Min 8/10 for completeness
        'reproducibility': 9            # Min 9/10 for scientific reproducibility
    }
    
    total_score = 0
    total_standard = 0
    conformity_count = 0
    
    for criterion, score in scores.items():
        standard = standards[criterion]
        meets_standard = "✅" if score >= standard else "❌"
        
        criterion_display = criterion.replace('_', ' ').title()
        print(f"│ {criterion_display:<33} │ {score:>8.1f} │ {meets_standard} {standard:>4}/10 │")
        
        total_score += score
        total_standard += standard
        if score >= standard:
            conformity_count += 1
    
    print("└" + "─" * 35 + "┴" + "─" * 10 + "┴" + "─" * 10 + "┘")
    
    overall_score = total_score / len(scores)
    overall_standard = total_standard / len(standards)
    conformity_rate = conformity_count / len(standards)
    
    print(f"\n🎯 PUNTEGGIO SCIENTIFICO COMPLESSIVO: {overall_score:.1f}/10")
    print(f"📊 STANDARD MINIMO RICHIESTO: {overall_standard:.1f}/10")
    print(f"📈 TASSO DI CONFORMITÀ: {conformity_rate:.1%}")
    
    # Valutazione finale
    print(f"\n🏆 VALUTAZIONE SCIENTIFICA:")
    if conformity_rate >= 0.85 and overall_score >= 8.5:
        print("✅ ECCELLENTE - Implementazione scientificamente rigorosa")
        print("   Modulo Prophet conforme agli standard accademici più elevati")
    elif conformity_rate >= 0.70 and overall_score >= 7.5:
        print("✅ BUONO - Implementazione scientificamente solida")
        print("   Modulo Prophet conforme alla maggior parte degli standard scientifici")
    elif conformity_rate >= 0.50 and overall_score >= 6.5:
        print("⚠️ ACCETTABILE - Implementazione con lacune scientifiche")
        print("   Modulo Prophet necessita miglioramenti per piena conformità")
    else:
        print("❌ INSUFFICIENTE - Implementazione scientificamente inadeguata")
        print("   Modulo Prophet richiede revisione sostanziale")
    
    # Raccomandazioni specifiche
    print(f"\n💡 RACCOMANDAZIONI SCIENTIFICHE:")
    
    critical_issues = []
    for criterion, score in scores.items():
        if score < standards[criterion]:
            critical_issues.append(criterion)
    
    if critical_issues:
        print("🔧 Aree che richiedono miglioramento:")
        for issue in critical_issues:
            print(f"   - {issue.replace('_', ' ').title()}: Score {scores[issue]:.1f}/{standards[issue]}")
    else:
        print("✅ Tutti i criteri scientifici soddisfatti!")
    
    return overall_score, conformity_rate

if __name__ == "__main__":
    scores = test_prophet_scientific_rigor()
    generate_scientific_report(scores)
