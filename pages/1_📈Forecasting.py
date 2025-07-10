import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import io
import warnings

from modules.forecast_engine import (
    run_enhanced_forecast, 
    run_auto_select_forecast, 
    display_forecast_results
)
from modules.config import *
from modules.data_utils import *
from modules.ui_components import *

# Page configuration
st.set_page_config(
    page_title="📈 Contact Center Forecasting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Contact Center Forecasting Tool")
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = {}

# Main workflow in sidebar
with st.sidebar:
    st.markdown("## 🔧 Configuration Workflow")
    
    # Step 1: Data Upload
    df, date_col, target_col, upload_config = render_data_upload_section()
    
    if df is not None and date_col and target_col:
        st.session_state.data_loaded = True
        
        # Step 2: Data Preview
        st.markdown("---")
        df_with_stats = render_data_preview_section(df, date_col, target_col, upload_config)
        
        # Step 3: Data Cleaning
        st.markdown("---")
        df_clean, cleaning_config = render_data_cleaning_section(df_with_stats, date_col, target_col)
        st.session_state.cleaned_data = df_clean
        
        # Step 4: External Regressors
        st.markdown("---")
        regressor_config = render_external_regressors_section(df_clean, date_col, target_col)
        
        # Step 5: Model Selection
        st.markdown("---")
        selected_model, model_configs = render_model_selection_section()
        st.session_state.model_configs = model_configs
        
        # Step 6: Forecast Configuration
        st.markdown("---")
        forecast_config = render_forecast_config_section()
        
        # Step 7: Output Configuration
        st.markdown("---")
        output_config = render_output_config_section()
        
        # Run Forecast Button
        st.markdown("---")
        if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
            st.session_state.run_forecast = True
        
        # Quick Stats Summary
        if st.session_state.cleaned_data is not None:
            st.markdown("---")
            st.subheader("📊 Data Summary")
            final_stats = get_data_statistics(st.session_state.cleaned_data, date_col, target_col)
            
            st.metric("📈 Records", final_stats['total_records'])
            st.metric("📅 Date Range", f"{final_stats['date_range_days']} days")
            if final_stats['missing_values'] > 0:
                st.metric("❌ Missing", f"{final_stats['missing_values']}")

# Main content area
if not st.session_state.data_loaded:
    # Welcome screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ## 🎯 Welcome to Contact Center Forecasting!
        
        This advanced forecasting tool helps you predict future call volumes, 
        staffing needs, and capacity requirements using state-of-the-art 
        time series models.
        
        ### 🚀 Features:
        - **📊 Multiple Models**: Prophet, ARIMA, SARIMA, Holt-Winters
        - **🔧 Auto-tuning**: Automatic parameter optimization
        - **📈 Advanced Analytics**: Seasonality, holidays, external regressors
        - **📋 Export Options**: CSV, Excel, PDF reports
        - **🎨 Rich Visualizations**: Interactive plots and diagnostics
        
        ### 🏁 Getting Started:
        1. **Upload your data** or use the sample dataset
        2. **Configure preprocessing** options
        3. **Select and tune** your forecasting model
        4. **Generate forecasts** and analyze results
        
        👈 **Start by configuring your data in the sidebar**
        """)
        
        # Quick start with sample data
        if st.button("🎯 Quick Start with Sample Data", type="primary"):
            st.rerun()

elif st.session_state.data_loaded and 'run_forecast' not in st.session_state:
    # Data loaded but forecast not run yet
    st.markdown("## 📊 Data Analysis & Preparation")
    
    if st.session_state.cleaned_data is not None:
        df_clean = st.session_state.cleaned_data
        
        # Enhanced data visualization
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Time series plot with statistical overlays
            st.subheader("📈 Time Series Overview")
            
            # Create enhanced plot
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=df_clean[date_col],
                y=df_clean[target_col],
                mode='lines',
                name='Historical Data',
                line=dict(color=PLOT_CONFIG['colors']['historical'])
            ))
            
            # Add trend line
            if len(df_clean) > 1:
                x_numeric = np.arange(len(df_clean))
                z = np.polyfit(x_numeric, df_clean[target_col], 1)
                p = np.poly1d(z)
                
                fig.add_trace(go.Scatter(
                    x=df_clean[date_col],
                    y=p(x_numeric),
                    mode='lines',
                    name='Trend Line',
                    line=dict(color=PLOT_CONFIG['colors']['trend'], dash='dash')
                ))
            
            # Add moving average
            if len(df_clean) >= 7:
                ma_7 = df_clean[target_col].rolling(window=7, center=True).mean()
                fig.add_trace(go.Scatter(
                    x=df_clean[date_col],
                    y=ma_7,
                    mode='lines',
                    name='7-day Moving Average',
                    line=dict(color='orange', width=2),
                    opacity=0.7
                ))
            
            fig.update_layout(
                title="Historical Time Series with Trend Analysis",
                xaxis_title="Date",
                yaxis_title=target_col,
                height=PLOT_CONFIG['height'],
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Data quality indicators
            st.subheader("📋 Data Quality Report")
            
            stats = get_data_statistics(df_clean, date_col, target_col)
            
            # Quality indicators
            quality_score = 100
            issues = []
            
            if stats['missing_percentage'] > 5:
                quality_score -= 20
                issues.append(f"High missing values ({stats['missing_percentage']:.1f}%)")
            
            if stats.get('duplicate_dates', 0) > 0:
                quality_score -= 15
                issues.append(f"Duplicate dates ({stats['duplicate_dates']})")
            
            # Calculate variance
            if stats['std_value'] == 0:
                quality_score -= 30
                issues.append("Zero variance in target")
            
            # Display quality score
            if quality_score >= 80:
                st.success(f"✅ Data Quality: {quality_score}/100")
            elif quality_score >= 60:
                st.warning(f"⚠️ Data Quality: {quality_score}/100")
            else:
                st.error(f"❌ Data Quality: {quality_score}/100")
            
            # List issues
            if issues:
                st.markdown("**Issues detected:**")
                for issue in issues:
                    st.markdown(f"• {issue}")
            else:
                st.success("🎉 No data quality issues detected!")
            
            # Distribution analysis
            st.subheader("📊 Value Distribution")
            
            # Histogram
            fig_hist = px.histogram(
                df_clean, 
                x=target_col,
                nbins=30,
                title="Value Distribution"
            )
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Seasonality analysis
        st.subheader("🔄 Seasonality Analysis")
        
        # Auto-detect seasonality patterns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(df_clean) >= 30:
                # Monthly seasonality
                df_clean['month'] = pd.to_datetime(df_clean[date_col]).dt.month
                monthly_avg = df_clean.groupby('month')[target_col].mean()
                
                fig_monthly = px.bar(
                    x=monthly_avg.index,
                    y=monthly_avg.values,
                    title="Average by Month",
                    labels={'x': 'Month', 'y': 'Average Value'}
                )
                fig_monthly.update_layout(height=250)
                st.plotly_chart(fig_monthly, use_container_width=True)
        
        with col2:
            if len(df_clean) >= 14:
                # Weekly seasonality
                df_clean['weekday'] = pd.to_datetime(df_clean[date_col]).dt.day_name()
                weekday_avg = df_clean.groupby('weekday')[target_col].mean()
                
                # Reorder weekdays
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                weekday_avg = weekday_avg.reindex([day for day in weekday_order if day in weekday_avg.index])
                
                fig_weekly = px.bar(
                    x=weekday_avg.index,
                    y=weekday_avg.values,
                    title="Average by Weekday",
                    labels={'x': 'Weekday', 'y': 'Average Value'}
                )
                fig_weekly.update_layout(height=250)
                fig_weekly.update_xaxes(tickangle=45)
                st.plotly_chart(fig_weekly, use_container_width=True)
        
        with col3:
            if len(df_clean) >= 24:
                # Daily seasonality (if timestamp data available)
                try:
                    df_clean['hour'] = pd.to_datetime(df_clean[date_col]).dt.hour
                    if df_clean['hour'].nunique() > 1:
                        hourly_avg = df_clean.groupby('hour')[target_col].mean()
                        
                        fig_hourly = px.line(
                            x=hourly_avg.index,
                            y=hourly_avg.values,
                            title="Average by Hour",
                            labels={'x': 'Hour', 'y': 'Average Value'}
                        )
                        fig_hourly.update_layout(height=250)
                        st.plotly_chart(fig_hourly, use_container_width=True)
                    else:
                        st.info("📊 Hourly patterns not available (daily data)")
                except:
                    st.info("📊 Hourly patterns not available")
        
        # Clean up temporary columns
        cols_to_drop = ['month', 'weekday', 'hour']
        for col in cols_to_drop:
            if col in df_clean.columns:
                df_clean.drop(col, axis=1, inplace=True)
        
        # Ready to forecast message
        st.markdown("---")
        st.success("🎯 **Data is ready for forecasting!** Configure your model settings in the sidebar and click 'Run Forecast'")

elif st.session_state.data_loaded and st.session_state.get('run_forecast', False):
    # Run the forecast
    st.markdown("## 🔮 Forecasting Results")
    
    # Get all necessary variables from session state and sidebar
    df_clean = st.session_state.cleaned_data
    
    if df_clean is not None and len(df_clean) > 0:
        # Prepare forecast configuration
        base_config = {
            'forecast_periods': forecast_config.get('forecast_periods', 30),
            'confidence_interval': forecast_config.get('confidence_interval', 0.95),
            'train_size': 0.8
        }
        
        st.markdown("### 🚀 Running Forecast...")
        
        try:
            with st.spinner(f"🔄 Running {selected_model} forecast..."):
                if selected_model == "Auto-Select":
                    # Run auto-select with model comparison
                    best_model, forecast_df, metrics, plots = run_auto_select_forecast(
                        df_clean, date_col, target_col, st.session_state.model_configs, base_config
                    )
                    
                    if best_model != "None":
                        display_forecast_results(best_model, forecast_df, metrics, plots)
                    else:
                        st.error("❌ Auto-select failed. No models succeeded.")
                
                else:
                    # Run single model
                    model_config = st.session_state.model_configs.get(selected_model, {})
                    forecast_df, metrics, plots = run_enhanced_forecast(
                        df_clean, date_col, target_col, selected_model, model_config, base_config
                    )
                    
                    if not forecast_df.empty:
                        display_forecast_results(selected_model, forecast_df, metrics, plots)
                    else:
                        st.error(f"❌ {selected_model} forecast failed. Please check your parameters.")
                
                st.success(f"✅ Forecast completed successfully!")
                
        except Exception as e:
            st.error(f"❌ Error during forecast execution:")
            st.exception(e)
            st.info("💡 Try adjusting model parameters or data preprocessing options.")
    
    else:
        st.error("❌ No cleaned data available. Please check data preprocessing steps.")
    
    # Reset forecast trigger
    if st.button("🔄 Run Another Forecast"):
        del st.session_state.run_forecast
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    📈 <strong>CC-Excellence Forecasting Tool</strong> | Built with Streamlit & Advanced Time Series Models
</div>
""", unsafe_allow_html=True)