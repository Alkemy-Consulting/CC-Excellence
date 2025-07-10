#!/usr/bin/env python3
"""
Test SARIMA imports to identify the exact error.
"""

def test_sarima_imports():
    """Test SARIMA imports step by step."""
    print("Testing SARIMA imports step by step...")
    
    try:
        # Test basic imports first
        import pandas as pd
        import numpy as np
        print("✅ Basic imports successful")
        
        # Test sklearn fixes
        print("Testing sklearn compatibility fixes...")
        
        # Handle sklearn compatibility issues - replicate exact code from sarima_enhanced.py
        try:
            from sklearn.utils import check_matplotlib_support
            print("✅ check_matplotlib_support imported from sklearn")
        except ImportError:
            print("⚠️ check_matplotlib_support not in sklearn, using fallback")
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
            print("✅ check_matplotlib_support monkey-patched")
        
        # Handle _check_fit_params compatibility issue
        try:
            from sklearn.utils.validation import _check_fit_params
            print("✅ _check_fit_params imported from sklearn")
        except ImportError:
            print("⚠️ _check_fit_params not in sklearn, using fallback")
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
            print("✅ _check_fit_params monkey-patched")
        
        # Test pmdarima import
        try:
            import pmdarima as pm
            print("✅ pmdarima imported successfully")
        except ImportError as e:
            print(f"❌ pmdarima import failed: {e}")
            return False
        
        # Test statsmodels imports
        try:
            from statsmodels.tsa.arima.model import ARIMA
            print("✅ ARIMA imported")
        except ImportError as e:
            print(f"❌ ARIMA import failed: {e}")
            return False
            
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            print("✅ SARIMAX imported")
        except ImportError as e:
            print(f"❌ SARIMAX import failed: {e}")
            return False
        
        # Test other statsmodels components
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.stats.diagnostic import acorr_ljungbox
            from statsmodels.tsa.stattools import adfuller, kpss
            print("✅ statsmodels utilities imported")
        except ImportError as e:
            print(f"❌ statsmodels utilities import failed: {e}")
            return False
        
        # Test scipy and sklearn
        try:
            from scipy import stats
            from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
            print("✅ scipy and sklearn metrics imported")
        except ImportError as e:
            print(f"❌ scipy/sklearn import failed: {e}")
            return False
        
        print("🎉 All SARIMA dependencies imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_sarima_imports()
