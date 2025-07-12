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
    page_title="Contact Center Forecasting Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Contact Center Forecasting Tool")
st.markdown("---")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'date_col' not in st.session_state:
    st.session_state.date_col = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = {}
if 'forecast_config' not in st.session_state:
    st.session_state.forecast_config = {}

# Main workflow in sidebar
with st.sidebar:
    st.markdown("## 🔧 Configuration Workflow")
    
    # Step 1: Data Upload
    df, date_col, target_col, upload_config = render_data_upload_section()
    
    if df is not None and date_col and target_col:
        st.session_state.data_loaded = True
        st.session_state.date_col = date_col
        st.session_state.target_col = target_col
        
        # Step 2: Data Cleaning & Preprocessing
        st.markdown("---")
        df_clean, cleaning_config = render_data_cleaning_section(df, date_col, target_col)
        if df_clean is not None:
            st.session_state.cleaned_data = df_clean
        
        # Step 3: External Regressors
        st.markdown("---")
        regressor_config = render_external_regressors_section(df, date_col, target_col)
        
        # Step 4: Model Selection
        st.markdown("---")
        selected_model, model_configs = render_model_selection_section()
        st.session_state.model_configs = model_configs
        st.session_state.selected_model = selected_model
        
        # Step 5: Forecast Configuration
        st.markdown("---")
        forecast_config = render_forecast_config_section()
        st.session_state.forecast_config = forecast_config
        
        # Step 6: Output Configuration
        st.markdown("---")
        output_config = render_output_config_section()
        
        # Run Forecast Button - STRICT BUTTON-ONLY EXECUTION
        st.markdown("---")
        
        # Check if data is ready
        if st.session_state.cleaned_data is not None and len(st.session_state.cleaned_data) > 0:
            # Only enable button if data is available
            forecast_button = st.button(
                "🚀 Run Forecast", 
                type="primary", 
                use_container_width=True,
                key="run_forecast_button"
            )
            
            # Handle button click - ONLY set flag when button is clicked
            if forecast_button:
                st.session_state.run_forecast = True
                st.rerun()
            
            # Show current status
            if st.session_state.get('forecast_results_available', False):
                st.success("✅ Forecast completed! Results shown below.")
        else:
            st.button(
                "🚀 Run Forecast", 
                type="primary", 
                use_container_width=True, 
                disabled=True
            )
            st.warning("⚠️ Please load and clean data first")

# Main content area - FIXED STATE MANAGEMENT
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

elif st.session_state.data_loaded and not st.session_state.get('forecast_results_available', False):
    # Data loaded - show analysis page (regardless of forecast button state)
    
    # Use cleaned data from session state
    df_clean = st.session_state.cleaned_data
    date_col = st.session_state.date_col
    target_col = st.session_state.target_col
    
    # Guard against missing data or column names
    if df_clean is None or not date_col or not target_col:
        st.error("❌ Data not properly loaded. Please check the sidebar configuration.")
        st.stop()
    
    # Get comprehensive statistics
    stats = get_data_statistics(df_clean, date_col, target_col)

    # Create tabs for organizing content
    tab1, tab2 = st.tabs(["📊 Data Series Analysis", "Forecasting Results"])
    
    with tab1:
        # Key Dataset Metrics in horizontal layout
        st.markdown("## Key Dataset Metrics")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("📊 Total Records", f"{stats['total_records']}")
        with col2:
            st.metric("📅 Date Range", f"{stats['date_range_days']} days")
        with col3:
            st.metric("📈 Mean Value", f"{stats['mean_value']:.2f}")
        with col4:
            st.metric("📊 Standard Deviation", f"{stats['std_value']:.2f}")
        with col5:
            st.metric("📉 Minimum Value", f"{stats['min_value']:.2f}")
        with col6:
            st.metric("📈 Maximum Value", f"{stats['max_value']:.2f}")

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
        
        # Add 7-day moving average
        if len(df_clean) >= 7:
            df_clean['ma_7'] = df_clean[target_col].rolling(window=7, center=True).mean()
            fig.add_trace(go.Scatter(
                x=df_clean[date_col],
                y=df_clean['ma_7'],
                mode='lines',
                name='7-Day MA',
                line=dict(color='orange', width=2, dash='dash')
            ))

        # Add 30-day moving average
        if len(df_clean) >= 30:
            df_clean['ma_30'] = df_clean[target_col].rolling(window=30, center=True).mean()
            fig.add_trace(go.Scatter(
                x=df_clean[date_col],
                y=df_clean['ma_30'],
                mode='lines',
                name='30-Day MA',
                line=dict(color='green', width=2, dash='dot')
            ))

        # Add trend line
        z = np.polyfit(range(len(df_clean)), df_clean[target_col], 1)
        trend_line = np.poly1d(z)(range(len(df_clean)))
        fig.add_trace(go.Scatter(
            x=df_clean[date_col],
            y=trend_line,
            mode='lines',
            name='Linear Trend',
            line=dict(color='red', width=2, dash='dashdot')
        ))

        # Historical Time Series with Trend Analysis - Fixed legend positioning
        fig.update_layout(
            title="Historical Time Series with Trend Analysis",
            xaxis_title="Date",
            yaxis_title=target_col,
            height=PLOT_CONFIG['height'],
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",  # Align legend to the right
                x=0.95  # Position at 95% of the width (right side)
            ),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=30, label="1M", step="day", stepmode="backward"),
                        dict(count=90, label="3M", step="day", stepmode="backward"),
                        dict(count=180, label="6M", step="day", stepmode="backward"),
                        dict(count=365, label="1Y", step="day", stepmode="backward"),
                        dict(count=730, label="2Y", step="day", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    x=0.02,  # Position range selector at left (2% from left edge)
                    xanchor="left",  # Anchor to the left
                    y=1.02,  # Same height as legend
                    yanchor="bottom"
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add an expandable box with a preview of the historical data table
        with st.expander("🔍 Preview Historical Data Table"):
            st.dataframe(df_clean[[date_col, target_col]].head(10))
        
        # === DATA PRE-ANALYSIS SECTION ===
        st.markdown("### Dataset Overview & Quality Assessment")
        
        # Get outlier statistics
        outlier_stats = detect_outliers(df_clean, target_col)
        
        # Data quality indicators in horizontal layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if stats['missing_values'] > 0:
                st.metric("❌ Missing Values", f"{stats['missing_values']} ({stats['missing_percentage']:.2f}%)")
            else:
                st.metric("✅ Missing Values", "0 (0.00%)")
        
        with col2:
            if stats.get('duplicate_dates', 0) > 0:
                st.metric("🔄 Duplicate Dates", f"{stats['duplicate_dates']}")
            else:
                st.metric("✅ Duplicate Dates", "0")
        
        with col3:
            if outlier_stats['count'] > 0:
                st.metric("⚠️ Outliers", f"{outlier_stats['count']} ({outlier_stats['percentage']:.2f}%)")
            else:
                st.metric("✅ Outliers", "0 (0.00%)")
        
        with col4:
            # Detected frequency
            try:
                df_sorted = df_clean.sort_values(date_col)
                detected_freq = pd.infer_freq(df_sorted[date_col])
                if detected_freq:
                    # Convert frequency code to full description
                    freq_map = {
                        'D': 'Daily',
                        'W': 'Weekly', 
                        'M': 'Monthly',
                        'Q': 'Quarterly',
                        'Y': 'Yearly',
                        'A': 'Annual',
                        'H': 'Hourly',
                        'T': 'Minutely',
                        'S': 'Secondly',
                        'B': 'Business Daily',
                        'W-MON': 'Weekly (Monday)',
                        'W-TUE': 'Weekly (Tuesday)',
                        'W-WED': 'Weekly (Wednesday)',
                        'W-THU': 'Weekly (Thursday)',
                        'W-FRI': 'Weekly (Friday)',
                        'W-SAT': 'Weekly (Saturday)',
                        'W-SUN': 'Weekly (Sunday)',
                        'MS': 'Month Start',
                        'ME': 'Month End',
                        'QS': 'Quarter Start',
                        'QE': 'Quarter End',
                        'YS': 'Year Start',
                        'YE': 'Year End'
                    }
                    freq_display = freq_map.get(detected_freq, detected_freq)
                    st.metric("🔍 Frequency", freq_display)
                else:
                    st.metric("🔍 Frequency", "Irregular")
            except:
                st.metric("🔍 Frequency", "Unknown")
        
        # Data quality score and consolidated message
        quality_score = 100
        issues = []
        
        if stats['missing_percentage'] > 5:
            quality_score -= 20
            issues.append(f"High missing values ({stats['missing_percentage']:.2f}%)")
        
        if stats.get('duplicate_dates', 0) > 0:
            quality_score -= 15
            issues.append(f"Duplicate dates ({stats['duplicate_dates']})")
        
        if outlier_stats['percentage'] > 10:
            quality_score -= 15
            issues.append(f"High outlier percentage ({outlier_stats['percentage']:.2f}%)")
        
        # Calculate variance
        if stats['std_value'] == 0:
            quality_score -= 30
            issues.append("Zero variance in target")
        
        # Display consolidated quality message
        if quality_score >= 80:
            if issues:
                st.success(f"✅ Excellent Quality: {quality_score}/100")
                st.markdown("**⚠️ Minor Issues Detected:**")
                for issue in issues:
                    st.markdown(f"• {issue}")
            else:
                st.success(f"🎯 **Excellent Quality: {quality_score}/100** | 🎉 No data quality issues detected!")
        elif quality_score >= 60:
            st.warning(f"⚠️ **Good Quality: {quality_score}/100**")
            if issues:
                st.markdown("**Issues to consider:**")
                for issue in issues:
                    st.markdown(f"• {issue}")
        else:
            st.error(f"❌ **Poor Quality: {quality_score}/100**")
            if issues:
                st.markdown("**Critical Issues:**")
                for issue in issues:
                    st.markdown(f"• {issue}")
        
        # Time Series Decomposition Analysis
        st.subheader("🔄 Time Series Decomposition")
        st.write("Time series decomposition breaks down the series into its fundamental components: trend, seasonality, and residuals. This helps understand the underlying patterns in the data.")
        
        try:
            # Prepare the series for decomposition
            df_ts = df_clean.copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
            df_ts = df_ts.sort_values(date_col)
            df_ts = df_ts.set_index(date_col)
            
            # Remove any missing values
            series = df_ts[target_col].dropna()
            
            # Determine appropriate period for decomposition
            freq_to_period = {
                'Daily': 7,      # Weekly seasonality for daily data
                'Weekly': 52,    # Yearly seasonality for weekly data  
                'Monthly': 12,   # Yearly seasonality for monthly data
                'Quarterly': 4,  # Yearly seasonality for quarterly data
                'Hourly': 24,    # Daily seasonality for hourly data
                'Business Daily': 5  # Weekly seasonality for business days
            }
            
            # Get the frequency display name we calculated earlier
            try:
                df_sorted = df_clean.sort_values(date_col)
                detected_freq = pd.infer_freq(df_sorted[date_col])
                freq_map = {
                    'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly',
                    'Y': 'Yearly', 'A': 'Annual', 'H': 'Hourly', 'B': 'Business Daily'
                }
                freq_display = freq_map.get(detected_freq, 'Daily')
            except:
                freq_display = 'Daily'
            
            seasonal_period = freq_to_period.get(freq_display, 7)
            
            # Check if we have enough data for decomposition
            min_periods = 2 * seasonal_period
            if len(series) >= min_periods:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Perform additive decomposition
                decomposition = seasonal_decompose(series, model='additive', period=seasonal_period)
                
                # Create decomposition plot
                fig = go.Figure()
                
                # Original series
                fig.add_trace(go.Scatter(
                    x=decomposition.observed.index,
                    y=decomposition.observed.values,
                    mode='lines',
                    name='Original',
                    line=dict(color='blue', width=2)
                ))
                
                # Trend component
                fig.add_trace(go.Scatter(
                    x=decomposition.trend.index,
                    y=decomposition.trend.values,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', width=2)
                ))
                
                # Seasonal component
                fig.add_trace(go.Scatter(
                    x=decomposition.seasonal.index,
                    y=decomposition.seasonal.values,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='green', width=2)
                ))
                
                # Residual component
                fig.add_trace(go.Scatter(
                    x=decomposition.resid.index,
                    y=decomposition.resid.values,
                    mode='markers',
                    name='Residuals',
                    marker=dict(color='orange', size=4)
                ))
                
                # Time Series Decomposition - Fixed legend positioning
                fig.update_layout(
                    title=f"Time Series Decomposition (Period: {seasonal_period})",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    height=600,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",  # Align legend to the right
                        x=0.95  # Position at 95% of the width (right side)
                    ),
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=30, label="1M", step="day", stepmode="backward"),
                                dict(count=90, label="3M", step="day", stepmode="backward"),
                                dict(count=180, label="6M", step="day", stepmode="backward"),
                                dict(count=365, label="1Y", step="day", stepmode="backward"),
                                dict(count=730, label="2Y", step="day", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            x=0.02,  # Position range selector at left (2% from left edge)
                            xanchor="left",  # Anchor to the left
                            y=1.02,  # Same height as legend
                            yanchor="bottom"
                        ),
                        rangeslider=dict(
                            visible=True
                        ),
                        type="date"
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Decomposition insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    trend_change = decomposition.trend.dropna().iloc[-1] - decomposition.trend.dropna().iloc[0]
                    trend_direction = "📈 Increasing" if trend_change > 0 else "📉 Decreasing" if trend_change < 0 else "➡️ Stable"
                    st.metric("Trend Direction", trend_direction)
                
                with col2:
                    seasonal_strength = np.std(decomposition.seasonal.dropna()) / np.std(decomposition.observed.dropna()) * 100
                    st.metric("Seasonal Strength", f"{seasonal_strength:.2f}%")
                
                with col3:
                    residual_variance = np.var(decomposition.resid.dropna())
                    st.metric("Residual Variance", f"{residual_variance:.2f}")
                    
                # Component analysis
                with st.expander("🔍 Component Analysis"):
                    st.markdown(f"""
                    - **Trend**: Shows the long-term direction of the data (increasing)
                    - **Seasonal**: Captures repeating patterns every 7 periods with 31.5% strength
                    - **Residuals**: Random variations after removing trend and seasonality (variance: 190.94)
                    """)
                
                # Additional Seasonality Analysis
                st.markdown("### Detailed Seasonality Analysis")
                
                # Prepare data for seasonality analysis
                df_seasonal = df_clean.copy()
                df_seasonal[date_col] = pd.to_datetime(df_seasonal[date_col])
                df_seasonal = df_seasonal.sort_values(date_col)
                
                # Create seasonality plots based on data frequency and length
                seasonal_cols = st.columns(2)
                
                with seasonal_cols[0]:
                    # Monthly seasonality (if we have enough data)
                    if len(df_seasonal) >= 24:  # At least 2 years
                        try:
                            df_seasonal['month'] = df_seasonal[date_col].dt.month
                            monthly_stats = df_seasonal.groupby('month')[target_col].agg(['mean', 'std']).reset_index()
                            
                            fig_monthly = go.Figure()
                            fig_monthly.add_trace(go.Scatter(
                                x=monthly_stats['month'],
                                y=monthly_stats['mean'],
                                mode='lines+markers',
                                name='Monthly Average',
                                line=dict(color='blue', width=3),
                                marker=dict(size=8)
                            ))
                            
                            # Add confidence bands
                            fig_monthly.add_trace(go.Scatter(
                                x=monthly_stats['month'],
                                y=monthly_stats['mean'] + monthly_stats['std'],
                                mode='lines',
                                name='+1 Std',
                                line=dict(color='rgba(0,100,80,0)', width=0),
                                showlegend=False
                            ))
                            
                            fig_monthly.add_trace(go.Scatter(
                                x=monthly_stats['month'],
                                y=monthly_stats['mean'] - monthly_stats['std'],
                                mode='lines',
                                name='Monthly Range',
                                line=dict(color='rgba(0,100,80,0)', width=0),
                                fill='tonexty',
                                fillcolor='rgba(0,100,80,0.2)',
                                showlegend=True
                            ))
                            
                            fig_monthly.update_layout(
                                title='Monthly Seasonality Profile',
                                xaxis_title='Month',
                                yaxis_title='Average Value',
                                height=350,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="center",
                                    x=0.5
                                ),
                                xaxis=dict(
                                    tickmode='array',
                                    tickvals=list(range(1, 13)),
                                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                )
                            )
                            st.plotly_chart(fig_monthly, use_container_width=True)
                        except Exception as e:
                            st.info("📊 Monthly seasonality analysis not available")
                    else:
                        st.info("📊 Need at least 24 data points for monthly analysis")
                
                with seasonal_cols[1]:
                    # Weekly seasonality
                    if len(df_seasonal) >= 14:  # At least 2 weeks
                        try:
                            df_seasonal['dayofweek'] = df_seasonal[date_col].dt.dayofweek
                            weekly_stats = df_seasonal.groupby('dayofweek')[target_col].agg(['mean', 'std']).reset_index()
                            
                            fig_weekly = go.Figure()
                            fig_weekly.add_trace(go.Scatter(
                                x=weekly_stats['dayofweek'],
                                y=weekly_stats['mean'],
                                mode='lines+markers',
                                name='Weekly Average',
                                line=dict(color='blue', width=3),
                                marker=dict(size=8)
                            ))
                            
                            # Add confidence bands
                            fig_weekly.add_trace(go.Scatter(
                                x=weekly_stats['dayofweek'],
                                y=weekly_stats['mean'] + weekly_stats['std'],
                                mode='lines',
                                name='+1 Std',
                                line=dict(color='rgba(0,100,80,0)', width=0),
                                showlegend=False
                            ))
                            
                            fig_weekly.add_trace(go.Scatter(
                                x=weekly_stats['dayofweek'],
                                y=weekly_stats['mean'] - weekly_stats['std'],
                                mode='lines',
                                name='Weekly Range',
                                line=dict(color='rgba(0,100,80,0)', width=0),
                                fill='tonexty',
                                fillcolor='rgba(0,100,80,0.2)',
                                showlegend=True
                            ))
                            
                            fig_weekly.update_layout(
                                title='Weekly Seasonality Profile',
                                xaxis_title='Day of Week',
                                yaxis_title='Average Value',
                                height=350,
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="center",
                                    x=0.5
                                ),
                                xaxis=dict(
                                    tickmode='array',
                                    tickvals=list(range(7)),
                                    ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                                )
                            )
                            st.plotly_chart(fig_weekly, use_container_width=True)
                        except Exception as e:
                            st.info("📊 Weekly seasonality analysis not available")
                    else:
                        st.info("📊 Need at least 14 data points for weekly analysis")
                
                # Hourly seasonality (if applicable)
                if len(df_seasonal) >= 48:  # At least 2 days of hourly data
                    try:
                        df_seasonal['hour'] = df_seasonal[date_col].dt.hour
                        if df_seasonal['hour'].nunique() > 1:
                            hourly_stats = df_seasonal.groupby('hour')[target_col].mean()
                            
                            fig_hourly = go.Figure()
                            fig_hourly.add_trace(go.Scatter(
                                x=hourly_stats.index,
                                y=hourly_stats.values,
                                mode='lines+markers',
                                name='Hourly Average',
                                line=dict(color='green', width=2),
                                marker=dict(size=6)
                            ))
                            
                            fig_hourly.update_layout(
                                title="Daily Seasonality Profile (Hourly)",
                                xaxis_title="Hour of Day",
                                yaxis_title="Average Value",
                                height=300
                            )
                            st.plotly_chart(fig_hourly, use_container_width=True)
                    except Exception:
                        pass
                
                # Distribution Analysis
                st.markdown("### 📊 Distribution Analysis")
                dist_cols = st.columns(2)
                
                with dist_cols[0]:
                    # Histogram
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=series.dropna(),
                        nbinsx=30,
                        name='Distribution',
                        marker_color='lightblue',
                        opacity=0.7
                    ))
                    fig_hist.update_layout(
                        title="Value Distribution",
                        xaxis_title="Value",
                        yaxis_title="Frequency",
                        height=300
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with dist_cols[1]:
                    # Box plot for outlier identification
                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=series.dropna(),
                        name='Box Plot',
                        marker_color='lightgreen'
                    ))
                    fig_box.update_layout(
                        title="Box Plot - Outlier Detection",
                        yaxis_title="Value",
                        height=300
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Stationarity Analysis
                with st.expander("🔍 Stationarity Analysis"):
                    st.write("Stationarity tests help determine if the series has constant statistical properties over time, which is important for many forecasting models like ARIMA.")
                    
                    try:
                        # ACF/PACF Analysis
                        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
                        from statsmodels.tsa.stattools import adfuller
                        import matplotlib.pyplot as plt
                        
                        # Create ACF/PACF plots
                        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                        
                        plot_acf(series.dropna(), ax=axes[0], lags=min(40, len(series)//4), title='Autocorrelation Function (ACF)')
                        plot_pacf(series.dropna(), ax=axes[1], lags=min(40, len(series)//4), title='Partial Autocorrelation Function (PACF)')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Interpretation
                        st.info("""
                        📈 **ACF/PACF Interpretation:**
                        - **ACF**: Shows correlation between the series and its lagged values. Slow decay suggests non-stationarity.
                        - **PACF**: Shows direct correlation at each lag, removing intermediate lag effects.
                        - Sharp cutoffs help identify appropriate ARIMA parameters (p, d, q).
                        """)
                        
                        # ADF Test for stationarity
                        adf_result = adfuller(series.dropna())
                        
                        st.markdown("#### 🧪 Augmented Dickey-Fuller Test")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("ADF Statistic", f"{adf_result[0]:.4f}")
                        with col2:
                            st.metric("p-value", f"{adf_result[1]:.6f}")
                        with col3:
                            critical_value = adf_result[4]['5%']
                            st.metric("Critical Value (5%)", f"{critical_value:.4f}")
                        
                        # Stationarity conclusion
                        if adf_result[1] <= 0.05:
                            st.success("✅ **Series is likely STATIONARY** (p-value ≤ 0.05)")
                            st.markdown("The series appears to have constant mean and variance over time, making it suitable for ARIMA modeling without differencing.")
                        else:
                            st.warning("⚠️ **Series is likely NON-STATIONARY** (p-value > 0.05)")
                            st.markdown("The series may require differencing (parameter 'd' in ARIMA) to achieve stationarity. Consider first-order or seasonal differencing.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during stationarity analysis: {str(e)}")
                        st.info("💡 This might be due to insufficient data or computational issues.")
                
                # Add back the final box with model recommendations outside the expandable box
                st.markdown("#### 💡 Modeling Recommendations")

                if adf_result[1] <= 0.05:
                    st.markdown("""
                    **Recommended approaches:**
                    - ✅ ARIMA models can be applied directly
                    - ✅ Prophet handles trends automatically
                    - ✅ Holt-Winters suitable for seasonal patterns
                    """)
                else:
                    st.markdown("""
                    **Recommended preprocessing:**
                    - 🔄 Apply first differencing for ARIMA models
                    - 🔄 Consider seasonal differencing if seasonal non-stationarity detected
                    - ✅ Prophet and Holt-Winters handle non-stationarity automatically
                    """)
                
                # Clean up temporary columns
                temp_cols = ['month', 'weekday', 'hour', 'year']
                for col in temp_cols:
                    if col in df_seasonal.columns:
                        df_seasonal.drop(col, axis=1, inplace=True)
    
        except Exception as e:
            st.error(f"❌ Error during time series decomposition: {str(e)}")
            st.info("💡 This might be due to irregular data patterns or insufficient data points.")
    
    with tab2:
        # Execute forecast when run_forecast flag is set
        if st.session_state.get('run_forecast', False):
            # CRITICAL: Reset the flag immediately to prevent repeated execution
            st.session_state.run_forecast = False
            
            # Get all necessary variables from session state
            df_clean = st.session_state.cleaned_data
            date_col = st.session_state.date_col
            target_col = st.session_state.target_col
            
            if df_clean is not None and len(df_clean) > 0 and date_col and target_col:
                try:
                    # Get stored configurations
                    selected_model = st.session_state.get('selected_model', 'Prophet')
                    forecast_config = st.session_state.get('forecast_config', {
                        'forecast_periods': 30,
                        'confidence_interval': 0.95
                    })
                    model_configs = st.session_state.get('model_configs', {})
                    
                    # Prepare forecast configuration
                    base_config = {
                        'forecast_periods': forecast_config.get('forecast_periods', 30),
                        'confidence_interval': forecast_config.get('confidence_interval', 0.95),
                        'train_size': 0.8
                    }
                    
                    # Run forecast without extra header text
                    with st.spinner(f"🔄 Executing {selected_model} model..."):
                        if selected_model == "Auto-Select":
                            # Run auto-select with model comparison
                            best_model, forecast_df, metrics, plots = run_auto_select_forecast(
                                df_clean, date_col, target_col, model_configs, base_config
                            )
                            
                            if best_model and best_model != "None":
                                display_forecast_results(best_model, forecast_df, metrics, plots)
                            else:
                                st.error("❌ Auto-select failed. No models succeeded.")
                        
                        else:
                            # Run single model
                            model_config = model_configs.get(selected_model, {})
                            forecast_df, metrics, plots = run_enhanced_forecast(
                                df_clean, date_col, target_col, selected_model, model_config, base_config
                            )
                            
                            if not forecast_df.empty:
                                display_forecast_results(selected_model, forecast_df, metrics, plots)
                            else:
                                st.error(f"❌ {selected_model} forecast failed. Please check your parameters.")
                        
                        st.success(f"✅ Forecast completed successfully!")
                        
                        # Set flag to indicate results are available
                        st.session_state.forecast_results_available = True
                        
                except Exception as e:
                    st.error(f"❌ Error during forecast execution:")
                    st.exception(e)
                    st.info("💡 Try adjusting model parameters or data preprocessing options.")
        
        # Check if forecast results are available to show them
        elif st.session_state.get('forecast_results_available', False):
            st.info("📊 I risultati del forecasting sono già stati generati. Clicca su '🔄 Run Another Forecast' per eseguire un nuovo forecasting.")
        
        else:
            # Show message to run forecast if no results available yet
            st.warning("⚠️ **Nessun risultato di forecasting disponibile.** Configura le impostazioni del modello nella barra laterale e clicca su 'Run Forecast' per vedere i risultati.")
            
            # Preview Parameters Box - Prevenzione errori utente
            with st.expander("🔍 **Preview Parametri di Forecasting**", expanded=False):
                st.markdown("### 📋 Configurazione Attuale")
                
                # Get current configurations
                selected_model = st.session_state.get('selected_model', 'Prophet')
                forecast_config = st.session_state.get('forecast_config', {
                    'forecast_periods': 30,
                    'confidence_interval': 0.95
                })
                model_configs = st.session_state.get('model_configs', {})
                
                # Display current model selection
                st.markdown(f"**🤖 Modello Selezionato**: {selected_model}")
                
                # Display forecast configuration
                st.markdown("**⚙️ Configurazione Forecast:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"- **Periodi da prevedere**: {forecast_config.get('forecast_periods', 30)}")
                with col2:
                    st.markdown(f"- **Intervallo di confidenza**: {forecast_config.get('confidence_interval', 0.95):.2f}")
                
                # Display model-specific parameters
                if selected_model in model_configs:
                    model_config = model_configs[selected_model]
                    st.markdown(f"**🎛️ Parametri Specifici {selected_model}:**")
                    
                    if selected_model == "Prophet":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"- **Crescita**: {model_config.get('growth', 'linear')}")
                            st.markdown(f"- **Stagionalità annuale**: {'Abilitata' if model_config.get('yearly_seasonality', True) else 'Disabilitata'}")
                        with col2:
                            st.markdown(f"- **Stagionalità settimanale**: {'Abilitata' if model_config.get('weekly_seasonality', True) else 'Disabilitata'}")
                            st.markdown(f"- **Auto-tuning**: {'Abilitato' if model_config.get('enable_auto_tuning', False) else 'Disabilitato'}")
                    
                    elif "ARIMA" in selected_model:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"- **Ordine p**: {model_config.get('p', 'Auto')}")
                            st.markdown(f"- **Differenziazione d**: {model_config.get('d', 'Auto')}")
                        with col2:
                            st.markdown(f"- **Ordine q**: {model_config.get('q', 'Auto')}")
                            st.markdown(f"- **Auto-ARIMA**: {'Abilitato' if model_config.get('enable_auto_arima', True) else 'Disabilitato'}")
                    
                    elif "Holt" in selected_model:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"- **Tipo stagionalità**: {model_config.get('seasonal_type', 'add')}")
                            st.markdown(f"- **Periodi stagionali**: {model_config.get('seasonal_periods', 'Auto')}")
                        with col2:
                            st.markdown(f"- **Trend smorzato**: {'Abilitato' if model_config.get('damped_trend', False) else 'Disabilitato'}")
                            st.markdown(f"- **Auto-tuning**: {'Abilitato' if model_config.get('enable_auto_tuning', False) else 'Disabilitato'}")
                
                else:
                    st.info("🔧 **Parametri di default** verranno utilizzati per il modello selezionato.")
                
                # Data summary
                if st.session_state.get('cleaned_data') is not None:
                    df_clean = st.session_state.cleaned_data
                    st.markdown("**📊 Riassunto Dati:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"- **Record totali**: {len(df_clean)}")
                    with col2:
                        st.markdown(f"- **Colonna data**: {st.session_state.get('date_col', 'N/A')}")
                    with col3:
                        st.markdown(f"- **Colonna target**: {st.session_state.get('target_col', 'N/A')}")
                
                # Validation warnings
                warnings = []
                if forecast_config.get('forecast_periods', 30) > len(df_clean) * 0.5:
                    warnings.append("⚠️ Periodo di previsione molto lungo rispetto ai dati storici")
                
                if selected_model == "Auto-Select" and len(model_configs) == 0:
                    warnings.append("⚠️ Auto-Select senza configurazioni specifiche userà parametri di default")
                
                if warnings:
                    st.markdown("**🚨 Avvisi:**")
                    for warning in warnings:
                        st.markdown(f"- {warning}")
                else:
                    st.success("✅ **Configurazione validata** - Pronto per l'esecuzione!")

# Navigation section - shown when forecast results are available
if st.session_state.data_loaded and st.session_state.get('forecast_results_available', False):
    # Reset options and navigation - ENHANCED USER CONTROL
    st.markdown("---")
    st.subheader("🔄 Next Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run Another Forecast", type="secondary", use_container_width=True, key="run_another"):
            # Clear forecast results flag to return to data analysis view
            if 'forecast_results_available' in st.session_state:
                del st.session_state.forecast_results_available
            st.rerun()
    
    with col2:
        if st.button("📊 Back to Data Analysis", type="secondary", use_container_width=True, key="back_to_data"):
            # Clear forecast results flag to return to data analysis view
            if 'forecast_results_available' in st.session_state:
                del st.session_state.forecast_results_available
            st.rerun()
    
    with col3:
        if st.button("🆕 Load New Data", type="secondary", use_container_width=True, key="load_new_data"):
            # Clear all session state to start fresh
            for key in ['run_forecast', 'forecast_results_available', 'data_loaded', 'cleaned_data', 'model_configs', 'date_col', 'target_col']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    📈 <strong>CC-Excellence Forecasting Tool</strong> | Built with Streamlit & Advanced Time Series Models
</div>
""", unsafe_allow_html=True)