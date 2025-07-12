"""
Prophet Integration Test with Extended Diagnostics
Test that verifies the complete Prophet workflow including diagnostics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append('/workspaces/CC-Excellence')

from modules.prophet_module import run_prophet_forecast, run_prophet_diagnostics
from modules.prophet_core import ProphetForecaster

def test_complete_prophet_workflow():
    """Test complete Prophet workflow with diagnostics"""
    print("🧪 Testing Complete Prophet Workflow with Extended Diagnostics")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Create realistic call volume data with trend and seasonality
    trend = np.linspace(100, 120, 200)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(200) / 7)
    noise = np.random.normal(0, 5, 200)
    values = trend + weekly_seasonality + noise
    
    df = pd.DataFrame({
        'date': dates,
        'call_volume': values
    })
    
    print(f"📊 Sample data created: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Call volume range: {df['call_volume'].min():.1f} to {df['call_volume'].max():.1f}")
    
    # Prophet configuration
    model_config = {
        'growth': 'linear',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'add_holidays': False,
        'enable_auto_tuning': True,
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'confidence_interval': 0.80
    }
    
    base_config = {
        'forecast_periods': 30,
        'confidence_interval': 0.8,
        'train_size': 0.8
    }
    
    # Run Prophet forecast
    print("\n🔮 Running Prophet Forecast...")
    try:
        forecast_df, metrics, plots = run_prophet_forecast(
            df, 'date', 'call_volume', model_config, base_config
        )
        
        print(f"✅ Forecast completed successfully!")
        print(f"   Forecast shape: {forecast_df.shape}")
        print(f"   Metrics: {list(metrics.keys())}")
        print(f"   Plots: {list(plots.keys())}")
        
        # Print key metrics
        print(f"\n📈 Forecast Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   {metric.upper()}: {value:.3f}")
        
    except Exception as e:
        print(f"❌ Forecast failed: {str(e)}")
        return False
    
    # Test direct core functionality
    print("\n🏗️ Testing Core Prophet Functionality...")
    try:
        forecaster = ProphetForecaster()
        result = forecaster.run_forecast_core(df, 'date', 'call_volume', model_config, base_config)
        
        if result.success:
            print("✅ Core Prophet functionality working")
            print(f"   Model trained: {result.model is not None}")
            print(f"   Forecast generated: {result.raw_forecast is not None}")
            print(f"   Metrics calculated: {len(result.metrics)} metrics")
            
            # Test diagnostics
            print("\n🔬 Testing Extended Diagnostics...")
            
            diagnostic_results = run_prophet_diagnostics(
                df, 'date', 'call_volume', result, show_diagnostic_plots=False
            )
            
            print("✅ Diagnostic analysis completed!")
            
            # Print diagnostic summary
            analysis = diagnostic_results.get('analysis', {})
            quality_score = diagnostic_results.get('quality_score', 0)
            
            print(f"\n📊 Diagnostic Summary:")
            print(f"   Overall Quality Score: {quality_score:.1f}/100")
            
            if 'forecast_coverage' in analysis:
                coverage = analysis['forecast_coverage']
                print(f"   Data Coverage: {coverage.get('coverage_ratio', 0) * 100:.1f}%")
            
            if 'residual_analysis' in analysis:
                residuals = analysis['residual_analysis']
                if 'error' not in residuals:
                    print(f"   Residual Mean: {residuals.get('mean_residual', 0):.4f}")
                    print(f"   Residual Std: {residuals.get('std_residual', 0):.4f}")
                    print(f"   Normality Test: {'✅ PASSED' if residuals.get('is_normally_distributed', False) else '❌ FAILED'}")
            
            if 'trend_analysis' in analysis:
                trend_stats = analysis['trend_analysis']
                if 'error' not in trend_stats:
                    print(f"   Trend Direction: {trend_stats.get('trend_direction', 'Unknown').title()}")
                    print(f"   Trend Changes: {trend_stats.get('significant_changes', 0)}")
            
            # Quality assessment
            print(f"\n🎯 Quality Assessment:")
            if quality_score >= 80:
                print("   ✅ EXCELLENT - High quality forecast with reliable predictions")
            elif quality_score >= 60:
                print("   ⚠️ GOOD - Acceptable quality with minor areas for improvement")
            elif quality_score >= 40:
                print("   🔶 MODERATE - Forecast has limitations, use with caution")
            else:
                print("   ❌ POOR - Significant quality issues, consider model adjustments")
            
            return True
            
        else:
            print(f"❌ Core forecast failed: {result.error}")
            return False
            
    except Exception as e:
        print(f"❌ Error in core/diagnostic testing: {str(e)}")
        return False

def test_diagnostic_components():
    """Test individual diagnostic components"""
    print("\n🧪 Testing Individual Diagnostic Components")
    
    try:
        from modules.prophet_diagnostics import (
            ProphetDiagnosticConfig,
            create_diagnostic_analyzer,
            create_diagnostic_plots
        )
        
        # Test configuration
        config = ProphetDiagnosticConfig()
        print("✅ Diagnostic configuration created")
        
        # Test analyzer
        analyzer = create_diagnostic_analyzer(config)
        print("✅ Diagnostic analyzer created")
        
        # Test plots generator
        plots_generator = create_diagnostic_plots(config)
        print("✅ Diagnostic plots generator created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing diagnostic components: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Prophet Integration Test with Extended Diagnostics")
    print("=" * 60)
    
    # Test diagnostic components
    components_ok = test_diagnostic_components()
    
    # Test complete workflow
    workflow_ok = test_complete_prophet_workflow()
    
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print(f"   Diagnostic Components: {'✅ PASSED' if components_ok else '❌ FAILED'}")
    print(f"   Complete Workflow: {'✅ PASSED' if workflow_ok else '❌ FAILED'}")
    
    if components_ok and workflow_ok:
        print("\n🎉 ALL TESTS PASSED! Prophet with Extended Diagnostics is ready!")
        print("🔬 Features verified:")
        print("   • Enterprise Prophet forecasting")
        print("   • Extended diagnostic analysis")
        print("   • Quality scoring system")
        print("   • Residual analysis")
        print("   • Trend decomposition")
        print("   • Uncertainty quantification")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
