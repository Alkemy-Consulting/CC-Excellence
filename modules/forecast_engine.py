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
        st.error(f"Error in {model_name} forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}

def run_auto_select_forecast(df: pd.DataFrame, date_col: str, target_col: str,
                           model_configs: Dict[str, Any], base_config: Dict[str, Any]) -> Tuple[str, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Run multiple models and select the best one based on performance metrics
    """
    models_to_test = ["Prophet", "ARIMA", "SARIMA", "Holt-Winters"]
    results = {}
    
    st.info("🔄 Testing multiple models to find the best performer...")
    
    for model_name in models_to_test:
        st.info(f"Testing {model_name}...")
        
        try:
            model_config = model_configs.get(model_name, {})
            forecast_df, metrics, plots = run_enhanced_forecast(
                df, date_col, target_col, model_name, model_config, base_config
            )
            
            if not forecast_df.empty and metrics:
                results[model_name] = {
                    'forecast_df': forecast_df,
                    'metrics': metrics,
                    'plots': plots,
                    'mape': metrics.get('mape', float('inf'))
                }
                st.success(f"✅ {model_name} completed - MAPE: {metrics.get('mape', 'N/A'):.3f}")
            else:
                st.warning(f"⚠️ {model_name} failed to produce results")
                
        except Exception as e:
            st.warning(f"⚠️ {model_name} failed: {str(e)}")
            continue
    
    if not results:
        return "None", pd.DataFrame(), {}, {}
    
    # Select best model based on MAPE
    best_model = min(results.keys(), key=lambda x: results[x]['mape'])
    best_result = results[best_model]
    
    st.success(f"🏆 Best model: {best_model} (MAPE: {best_result['mape']:.3f})")
    
    return best_model, best_result['forecast_df'], best_result['metrics'], best_result['plots']

def display_forecast_results(model_name: str, forecast_df: pd.DataFrame, 
                           metrics: Dict[str, Any], plots: Dict[str, Any]):
    """
    Display comprehensive forecast results
    """
    st.subheader(f"📊 {model_name} Forecast Results")
    
    # Display metrics
    if metrics:
        st.subheader("📏 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'mape' in metrics:
                st.metric("MAPE", f"{metrics['mape']:.3f}%")
        with col2:
            if 'mae' in metrics:
                st.metric("MAE", f"{metrics['mae']:.3f}")
        with col3:
            if 'rmse' in metrics:
                st.metric("RMSE", f"{metrics['rmse']:.3f}")
        with col4:
            if 'r2' in metrics:
                st.metric("R²", f"{metrics['r2']:.3f}")
    
    # Display plots
    if plots:
        if 'forecast' in plots:
            st.subheader("📈 Forecast Plot")
            st.plotly_chart(plots['forecast'], use_container_width=True)
        
        if 'components' in plots:
            st.subheader("🔍 Forecast Components")
            st.plotly_chart(plots['components'], use_container_width=True)
        
        if 'tuning_results' in plots:
            st.subheader("🎯 Auto-Tuning Results")
            tuning_data = plots['tuning_results']['all_results']
            if tuning_data:
                tuning_df = pd.DataFrame(tuning_data)
                st.dataframe(tuning_df.round(4), use_container_width=True)
    
    # Display forecast data
    if not forecast_df.empty:
        st.subheader("📋 Forecast Data")
        st.dataframe(forecast_df.round(3), use_container_width=True)
        
        # Download options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"{model_name}_forecast.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel download would require xlsxwriter
            st.info("Excel download available with xlsxwriter")
        
        with col3:
            st.info("PDF report feature coming soon")
