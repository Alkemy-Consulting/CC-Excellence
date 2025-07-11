"""
Enhanced forecasting execution module that integrates all the advanced model implementations.
Provides a unified interface for running Prophet, ARIMA, SARIMA, and Holt-Winters models with advanced features.
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

from .prophet_module import run_prophet_forecast
from .arima_enhanced import run_arima_forecast
from .sarima_enhanced import run_sarima_forecast  
from .holtwinters_enhanced import run_holtwinters_forecast

def run_enhanced_forecast(df: pd.DataFrame, date_col: str, target_col: str,
                         model_name: str, model_config: Dict[str, Any], 
                         base_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Run enhanced forecasting with the specified model
    """
    try:
        print(f"DEBUG: Running {model_name} with model_config: {model_config}")
        print(f"DEBUG: Base config: {base_config}")
        print(f"DEBUG: Data validation - shape: {df.shape}, date_col: {date_col}, target_col: {target_col}")
        
        # Basic data validation
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        if len(df) < 10:
            raise ValueError(f"Insufficient data points: {len(df)} (minimum 10 required)")
            
        # Check for model-specific requirements
        if model_name in ["SARIMA", "Holt-Winters"]:
            seasonal_periods = model_config.get('seasonal_periods', model_config.get('seasonal_period', 12))
            if len(df) < seasonal_periods * 2:
                raise ValueError(f"Insufficient data for seasonal model: {len(df)} points, need at least {seasonal_periods * 2}")
        
        if model_name == "Prophet":
            return run_prophet_forecast(df, date_col, target_col, model_config, base_config)
        elif model_name == "ARIMA":
            return run_arima_forecast(df, date_col, target_col, model_config, base_config)
        elif model_name == "SARIMA":
            return run_sarima_forecast(df, date_col, target_col, model_config, base_config)
        elif model_name == "Holt-Winters":
            return run_holtwinters_forecast(df, date_col, target_col, model_config, base_config)
        else:
            st.error(f"Unknown model: {model_name}")
            return pd.DataFrame(), {}, {}
            
    except Exception as e:
        error_msg = f"Error in {model_name} forecasting: {str(e)}"
        st.error(error_msg)
        print(f"DEBUG: {error_msg}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        return pd.DataFrame(), {}, {}

def run_auto_select_forecast(df: pd.DataFrame, date_col: str, target_col: str,
                           model_configs: Dict[str, Any], base_config: Dict[str, Any]) -> Tuple[str, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Run auto-select forecast with robust metrics validation
    """
    models_to_test = ["Prophet", "ARIMA", "SARIMA", "Holt-Winters"]
    results = {}
    
    st.info("🔄 Testing multiple models to find the best performer...")
    
    for model_name in models_to_test:
        st.info(f"Testing {model_name}...")
        
        try:
            # Get model configuration with validation
            model_config = model_configs.get(model_name, {})
            
            # CRITICAL: Validate and convert parameter types
            if model_name == "SARIMA":
                model_config = validate_sarima_config(model_config)
            elif model_name == "Holt-Winters":
                model_config = validate_holtwinters_config(model_config)
            
            print(f"DEBUG Auto-Select: Running {model_name} with validated config: {model_config}")
            
            forecast_df, metrics, plots = run_enhanced_forecast(
                df, date_col, target_col, model_name, model_config, base_config
            )
            
            print(f"DEBUG Auto-Select: {model_name} returned - forecast_df.shape: {forecast_df.shape if not forecast_df.empty else 'empty'}")
            print(f"DEBUG Auto-Select: {model_name} returned - metrics: {metrics}")
            
            # Enhanced validation with fallback
            is_valid = False
            mape_value = 100.0  # Default high error
            
            if not forecast_df.empty and metrics:
                if 'mape' in metrics:
                    try:
                        mape_value = float(metrics['mape'])
                        if np.isfinite(mape_value) and mape_value >= 0:
                            is_valid = True
                        else:
                            print(f"DEBUG Auto-Select: {model_name} - Invalid MAPE value: {mape_value}")
                            mape_value = 100.0
                    except (ValueError, TypeError) as conv_error:
                        print(f"DEBUG Auto-Select: {model_name} - MAPE conversion error: {conv_error}")
                        mape_value = 100.0
                else:
                    print(f"DEBUG Auto-Select: {model_name} - MAPE key missing from metrics: {list(metrics.keys())}")
            else:
                print(f"DEBUG Auto-Select: {model_name} - forecast_df empty or no metrics")
            
            # Store results even if not perfect (for debugging)
            if not forecast_df.empty:  # At least have forecast data
                results[model_name] = {
                    'forecast_df': forecast_df,
                    'metrics': metrics if metrics else {'mape': 100.0, 'mae': 0.0, 'rmse': 0.0, 'r2': 0.0},
                    'plots': plots if plots else {},
                    'mape': mape_value
                }
                
                if is_valid:
                    st.success(f"✅ {model_name} completed - MAPE: {mape_value:.3f}%")
                else:
                    st.warning(f"⚠️ {model_name} completed with issues - MAPE: {mape_value:.3f}%")
            else:
                st.error(f"❌ {model_name} failed - No forecast data generated")
                
        except Exception as e:
            st.error(f"❌ {model_name} failed: {str(e)}")
            print(f"DEBUG Auto-Select Error in {model_name}: {str(e)}")
            import traceback
            print(f"DEBUG Auto-Select Full traceback for {model_name}: {traceback.format_exc()}")
            continue
    
    if not results:
        st.error("❌ No models produced valid results")
        return "None", pd.DataFrame(), {}, {}
    
    # Select best model based on MAPE (lowest is best)
    best_model = min(results.keys(), key=lambda x: results[x]['mape'])
    best_result = results[best_model]
    
    st.success(f"🏆 Best model: {best_model} (MAPE: {best_result['mape']:.3f}%)")
    
    return best_model, best_result['forecast_df'], best_result['metrics'], best_result['plots']

def validate_sarima_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and convert SARIMA configuration parameters to correct types"""
    validated = config.copy()
    
    # If config is empty, provide safe defaults
    if not validated:
        validated = {
            'auto_sarima': True,
            'max_p': 3, 'max_d': 2, 'max_q': 3,
            'max_P': 2, 'max_D': 1, 'max_Q': 2,
            'seasonal_period': 12,
            'information_criterion': 'aic'
        }
        print(f"DEBUG: Using default SARIMA config: {validated}")
        return validated
    
    # Convert string parameters to integers
    int_params = ['p', 'd', 'q', 'P', 'D', 'Q', 'seasonal_period', 'max_p', 'max_d', 'max_q', 'max_P', 'max_D', 'max_Q']
    for param in int_params:
        if param in validated and validated[param] is not None:
            try:
                validated[param] = int(float(str(validated[param])))  # Handle both string and float inputs
            except (ValueError, TypeError):
                # Use defaults if conversion fails
                defaults = {
                    'p': 1, 'd': 1, 'q': 1, 'P': 1, 'D': 1, 'Q': 1, 
                    'seasonal_period': 12, 'max_p': 3, 'max_d': 2, 'max_q': 3,
                    'max_P': 2, 'max_D': 1, 'max_Q': 2
                }
                validated[param] = defaults.get(param, 1)
                print(f"DEBUG: SARIMA param {param} conversion failed, using default: {validated[param]}")
    
    # Ensure auto_sarima is boolean
    if 'auto_sarima' not in validated:
        validated['auto_sarima'] = True
        
    print(f"DEBUG: Validated SARIMA config: {validated}")
    return validated

def validate_holtwinters_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and convert Holt-Winters configuration parameters to correct types"""
    validated = config.copy()
    
    # If config is empty, provide safe defaults
    if not validated:
        validated = {
            'auto_holtwinters': True,
            'trend_type': 'add',
            'seasonal_type': 'add',
            'damped_trend': False,
            'seasonal_periods': 12,
            'alpha': None,  # Auto-optimize
            'beta': None,   # Auto-optimize  
            'gamma': None   # Auto-optimize
        }
        print(f"DEBUG: Using default Holt-Winters config: {validated}")
        return validated
    
    # Convert string parameters to integers
    if 'seasonal_periods' in validated and validated['seasonal_periods'] is not None:
        try:
            validated['seasonal_periods'] = int(float(str(validated['seasonal_periods'])))
        except (ValueError, TypeError):
            validated['seasonal_periods'] = 12
            print(f"DEBUG: Holt-Winters seasonal_periods conversion failed, using default: 12")
    else:
        validated['seasonal_periods'] = 12
    
    # Convert string parameters to floats (smoothing parameters)
    float_params = ['alpha', 'beta', 'gamma']
    for param in float_params:
        if param in validated and validated[param] is not None:
            try:
                validated[param] = float(str(validated[param]))
            except (ValueError, TypeError):
                validated[param] = None  # Let the model auto-optimize
                print(f"DEBUG: Holt-Winters param {param} conversion failed, using None for auto-optimization")
    
    # Ensure auto_holtwinters is boolean
    if 'auto_holtwinters' not in validated:
        validated['auto_holtwinters'] = True
        
    # Ensure trend and seasonal types are valid
    if 'trend_type' not in validated or validated['trend_type'] not in ['add', 'mul', None]:
        validated['trend_type'] = 'add'
        
    if 'seasonal_type' not in validated or validated['seasonal_type'] not in ['add', 'mul', None]:
        validated['seasonal_type'] = 'add'
        
    if 'damped_trend' not in validated:
        validated['damped_trend'] = False
        
    print(f"DEBUG: Validated Holt-Winters config: {validated}")
    return validated

def display_forecast_results(model_name: str, forecast_df: pd.DataFrame, metrics: Dict[str, Any], plots: Dict[str, Any]):
    """Display forecast results without header text"""
    try:
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAPE", f"{metrics.get('mape', 0):.3f}%")
        with col2:
            st.metric("MAE", f"{metrics.get('mae', 0):.3f}")
        with col3:
            st.metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
        with col4:
            st.metric("R²", f"{metrics.get('r2', 0):.3f}")
        
        # Display forecast plot if available
        if 'forecast_plot' in plots:
            st.plotly_chart(plots['forecast_plot'], use_container_width=True)
        
        # Display additional plots if available
        for plot_name, plot_fig in plots.items():
            if plot_name != 'forecast_plot' and plot_fig is not None:
                st.plotly_chart(plot_fig, use_container_width=True)
        
        # Display forecast data table
        if not forecast_df.empty:
            with st.expander("📊 Forecast Data Table"):
                st.dataframe(forecast_df)
                
                # Download forecast data
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast Data (CSV)",
                    data=csv,
                    file_name=f"{model_name}_forecast.csv",
                    mime="text/csv"
                )
        
    except Exception as e:
        st.error(f"Error displaying forecast results: {str(e)}")
