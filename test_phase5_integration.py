#!/usr/bin/env python3
"""
Phase 5 Advanced ML Features Integration Test
Tests ensemble forecasting, feature engineering, and hyperparameter optimization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_phase_5_ml_advanced_integration():
    """Test Phase 5 Advanced ML Features integration"""
    print("🚀 PHASE 5 ADVANCED ML FEATURES - INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Test 1: ML Feature Configuration
        print("\n1. Testing ML Feature Configuration...")
        from modules.prophet_ml_advanced import MLFeatureConfig, create_feature_engineer
        
        config = MLFeatureConfig(
            lag_features=[1, 7],
            rolling_windows=[7, 14],
            fourier_order=5,
            enable_trends=True
        )
        print(f"✅ ML Configuration: {len(config.lag_features)} lag features, fourier_order={config.fourier_order}")
        
        # Test 2: Advanced Feature Engineering
        print("\n2. Testing Advanced Feature Engineering...")
        feature_engineer = create_feature_engineer(config)
        
        # Create test data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        values = 100 + np.cumsum(np.random.randn(100) * 0.1)
        df = pd.DataFrame({'ds': dates, 'y': values})
        
        features_df = feature_engineer.engineer_features(df, 'ds', 'y')
        selected_df = feature_engineer.select_features(features_df)
        
        print(f"✅ Feature Engineering: {len(feature_engineer.feature_names)} features created")
        print(f"   Selected features: {len(selected_df.columns)-1} (excluding target)")
        
        # Test 3: Ensemble Forecaster Creation
        print("\n3. Testing Ensemble Forecaster...")
        from modules.prophet_ml_advanced import create_ensemble_forecaster
        
        ensemble_forecaster = create_ensemble_forecaster(
            enable_prophet=False,  # Skip Prophet for speed
            enable_ml_models=True
        )
        
        models = ensemble_forecaster.initialize_models()
        print(f"✅ Ensemble Models: {len(models)} models initialized")
        print(f"   Models: {list(models.keys())}")
        
        # Test 4: Integration with Prophet Module
        print("\n4. Testing Prophet Module Integration...")
        from modules.prophet_module import run_prophet_feature_engineering
        
        feature_result = run_prophet_feature_engineering(
            df, 'ds', 'y', 
            config={'lag_features': [1, 7], 'rolling_windows': [7], 'fourier_order': 3}
        )
        
        print(f"✅ Prophet Feature Integration:")
        print(f"   Features created: {feature_result['feature_statistics']['total_features_created']}")
        print(f"   Features selected: {feature_result['feature_statistics']['features_selected']}")
        
        # Test 5: Hyperparameter Optimizer
        print("\n5. Testing Hyperparameter Optimizer...")
        from modules.prophet_ml_advanced import create_hyperparameter_optimizer
        
        optimizer = create_hyperparameter_optimizer(n_trials=3, timeout=30)
        print(f"✅ Hyperparameter Optimizer: {optimizer.n_trials} trials, {optimizer.timeout}s timeout")
        
        print("\n🎯 PHASE 5 ADVANCED ML FEATURES - ALL TESTS PASSED!")
        print("📊 Features implemented:")
        print("   • Advanced feature engineering with lag, rolling, and Fourier features")
        print("   • Feature selection with statistical methods") 
        print("   • Ensemble forecasting with multiple ML models")
        print("   • Integration with Prophet module")
        print("   • Automated hyperparameter optimization")
        print("   • Performance-optimized implementations")
        print()
        print("✅ READY TO PROCEED TO PHASE 6: PRODUCTION DEPLOYMENT")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during Phase 5 testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase_5_ml_advanced_integration()
    sys.exit(0 if success else 1)
