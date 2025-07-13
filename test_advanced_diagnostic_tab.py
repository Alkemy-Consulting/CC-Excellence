"""
Test rapido per verificare che il nuovo tab Advanced Diagnostic funzioni.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_advanced_diagnostic_tab():
    """Test basic functionality of the new tab structure"""
    st.title("🧪 Test Advanced Diagnostic Tab")
    
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    values = np.random.randn(len(dates)).cumsum() + 100
    
    # Simulate session state data as if a forecast was run
    if st.button("Simulate Forecast Results"):
        st.session_state.last_forecast_metrics = {
            'mape': 15.5,
            'mae': 12.3,
            'rmse': 18.7,
            'r2': 0.85
        }
        
        st.session_state.last_forecast_df = pd.DataFrame({
            'ds': dates[-30:],  # Last 30 days as forecast
            'yhat': values[-30:] + np.random.randn(30) * 5,
            'yhat_lower': values[-30:] + np.random.randn(30) * 5 - 10,
            'yhat_upper': values[-30:] + np.random.randn(30) * 5 + 10
        })
        
        st.session_state.selected_model = 'Prophet'
        st.session_state.forecast_results_available = True
        st.session_state.cleaned_data = pd.DataFrame({
            'ds': dates,
            'y': values
        })
        st.session_state.date_col = 'ds'
        st.session_state.target_col = 'y'
        
        st.success("✅ Forecast results simulated!")
        st.info("Now you can test the Advanced Diagnostic tab")
    
    # Test tab structure
    tab1, tab2, tab3 = st.tabs(["📊 Data Analysis", "Forecasting Results", "🔬 Advanced Diagnostic"])
    
    with tab1:
        st.write("Sample data analysis tab")
    
    with tab2:
        st.write("Sample forecasting results tab")
    
    with tab3:
        # Test the advanced diagnostic logic
        if st.session_state.get('forecast_results_available', False):
            st.markdown("## 🔬 Advanced Diagnostic Dashboard")
            st.success("✅ Advanced diagnostic tab is working!")
            
            # Show sample metrics
            if hasattr(st.session_state, 'last_forecast_metrics'):
                metrics = st.session_state.last_forecast_metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAPE", f"{metrics['mape']:.2f}%")
                with col2:
                    st.metric("MAE", f"{metrics['mae']:.2f}")
                with col3:
                    st.metric("RMSE", f"{metrics['rmse']:.2f}")
                with col4:
                    st.metric("R²", f"{metrics['r2']:.2f}")
        else:
            st.info("Execute the 'Simulate Forecast Results' button above to test the advanced diagnostics")

if __name__ == "__main__":
    test_advanced_diagnostic_tab()
