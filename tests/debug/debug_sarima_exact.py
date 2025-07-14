#!/usr/bin/env python3
"""
Test to identify the exact SARIMA import error.
"""

def test_exact_sarima_error():
    """Test the exact import block from SARIMA module."""
    print("Testing exact SARIMA import block...")
    
    try:
        # Replicate the exact import sequence from sarima_enhanced.py
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        # Apply sklearn fixes first
        print("Applying sklearn compatibility fixes...")
        
        # Handle sklearn compatibility issues
        try:
            from sklearn.utils import check_matplotlib_support
        except ImportError:
            # Fallback for newer sklearn versions where check_matplotlib_support was removed
            def check_matplotlib_support(caller_name):
                """Compatibility fallback for removed sklearn function."""
                try:
                    import matplotlib.pyplot as plt
                    return True
                except ImportError:
                    import warnings
                    warnings.warn(f"{caller_name} requires matplotlib which is not installed.")
                    return False
            
            # Monkey patch for pmdarima compatibility
            import sklearn.utils
            sklearn.utils.check_matplotlib_support = check_matplotlib_support

        # Handle _check_fit_params compatibility issue
        try:
            from sklearn.utils.validation import _check_fit_params
        except ImportError:
            # Fallback for newer sklearn versions where _check_fit_params was removed/moved
            def _check_fit_params(X, fit_params, indices=None):
                """Compatibility fallback for removed sklearn function."""
                if fit_params is None:
                    return {}
                
                fit_params_validated = {}
                for key, value in fit_params.items():
                    if hasattr(value, '__len__') and hasattr(value, '__getitem__'):
                        if indices is not None:
                            try:
                                fit_params_validated[key] = value[indices]
                            except (IndexError, TypeError):
                                fit_params_validated[key] = value
                        else:
                            fit_params_validated[key] = value
                    else:
                        fit_params_validated[key] = value
                
                return fit_params_validated
            
            # Monkey patch for pmdarima compatibility
            import sklearn.utils.validation
            sklearn.utils.validation._check_fit_params = _check_fit_params

        print("✅ sklearn compatibility fixes applied")
        
        # Now test the exact import block
        SARIMA_AVAILABLE = False
        try:
            import pmdarima as pm
            print("✅ pmdarima imported")
            
            from pmdarima import auto_arima
            print("✅ auto_arima imported")
            
            from statsmodels.tsa.arima.model import ARIMA
            print("✅ ARIMA imported")
            
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            print("✅ SARIMAX imported")
            
            from statsmodels.tsa.seasonal import seasonal_decompose
            print("✅ seasonal_decompose imported")
            
            from statsmodels.stats.diagnostic import acorr_ljungbox
            print("✅ acorr_ljungbox imported")
            
            from statsmodels.tsa.stattools import adfuller, kpss
            print("✅ adfuller, kpss imported")
            
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            print("✅ plot_acf, plot_pacf imported")
            
            from scipy import stats
            print("✅ scipy.stats imported")
            
            from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
            print("✅ sklearn.metrics imported")
            
            import joblib
            print("✅ joblib imported")
            
            import io
            print("✅ io imported")
            
            SARIMA_AVAILABLE = True
            print("🎉 All imports successful - SARIMA_AVAILABLE = True")
            
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print(f"❌ SARIMA_AVAILABLE = False")
            SARIMA_AVAILABLE = False
            
            # Don't call st.error here, just print
            import traceback
            traceback.print_exc()
            
            return False
            
        # Test the config import too
        try:
            from src.modules.utils.config import (
                MODEL_LABELS, SARIMA_DEFAULTS, FORECAST_DEFAULTS,
                VISUALIZATION_CONFIG, ERROR_MESSAGES
            )
            print("✅ Config imports successful")
        except ImportError as e:
            print(f"❌ Config import failed: {e}")
            return False
            
        return SARIMA_AVAILABLE
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exact_sarima_error()
    print(f"\nFinal result: SARIMA available = {success}")
