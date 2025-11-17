import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import datetime
import json
import io
import warnings

from modules.forecast_engine import (
    run_enhanced_forecast, 
    run_auto_select_forecast, 
    display_forecast_results
)
from src.modules.utils.config import *
from src.modules.utils.data_utils import *
from src.modules.visualization.ui_components import *

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
        
        # Output Configuration section removed as requested
        
        # Run Forecast Button - STRICT BUTTON-ONLY EXECUTION
        st.markdown("---")
        
        # Check if data is ready
        if st.session_state.cleaned_data is not None and len(st.session_state.cleaned_data) > 0:
            # Only enable button if data is available
            forecast_button = st.button(
                "🚀 Run Forecast", 
                type="primary", 
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
                disabled=True
            )
            st.warning("⚠️ Please load and clean data first")

# Main content area - FIXED STATE MANAGEMENT
if not st.session_state.data_loaded:
    # Initial state: prompt user to load data from the sidebar
    st.markdown("##  Forecasting Tool")
    st.info("� **Inizia caricando i tuoi dati o selezionando un set di dati di esempio dalla barra laterale.**")

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
    
    tab1, tab2, tab3, tab4 = st.tabs(["Data Series Analysis", "Forecasting Results", "Advanced Diagnostic", "Summary & Export"])
    
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
       
            
        # Time Series Decomposition Analysis
        st.subheader("Time Series Decomposition")
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
                st.markdown("### Distribution Analysis")
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
                    # Get stored configurations with safe defaults
                    selected_model = st.session_state.get('selected_model', 'Prophet')
                    forecast_config = st.session_state.get('forecast_config')
                    if not forecast_config:
                        forecast_config = {
                            'forecast_periods': 30,
                            'confidence_interval': 0.95
                        }
                    model_configs = st.session_state.get('model_configs')
                    if not model_configs:
                        model_configs = {}
                    
                    # Prepare forecast configuration
                    enable_backtesting = forecast_config.get('enable_backtesting', False)
                    train_size = forecast_config.get('train_size', 1.0 if not enable_backtesting else 0.8)
                    base_config = {
                        'forecast_periods': forecast_config.get('forecast_periods', 30),
                        'confidence_interval': forecast_config.get('confidence_interval', 0.95),
                        'train_size': train_size,
                        'enable_backtesting': enable_backtesting
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
                            # Run single model with safe model configuration
                            model_config = model_configs.get(selected_model, {}) if model_configs else {}
                            if model_config is None:
                                model_config = {
                                    'growth': 'linear',
                                    'yearly_seasonality': True,
                                    'weekly_seasonality': True,
                                    'daily_seasonality': False,
                                    'seasonality_mode': 'additive'
                                }
                                
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
                if selected_model in model_configs and model_configs[selected_model] is not None:
                    model_config = model_configs[selected_model]
                    # Additional safety check to ensure model_config is a dictionary
                    if not isinstance(model_config, dict):
                        model_config = {}
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

    with tab3:
        # Advanced Diagnostic Tab - only show if forecast results are available
        if st.session_state.get('forecast_results_available', False):
            st.markdown("## 🔬 Advanced Diagnostic Dashboard")
            st.markdown("Analisi completa del modello di forecasting con suggerimenti pratici per migliorare l'accuratezza delle previsioni.")
            
            # === KEY INSIGHTS AND RECOMMENDATIONS (TOP SECTION) ===
            st.markdown("---")
            st.markdown("## 🎯 **Principali Insights e Raccomandazioni**")
            
            # Get metrics for analysis
            if hasattr(st.session_state, 'last_forecast_metrics'):
                metrics = st.session_state.last_forecast_metrics
                
                # Quality Score Calculation
                quality_score = 100
                quality_issues = []
                recommendations = []
                
                if 'mape' in metrics:
                    mape_val = metrics['mape']
                    if mape_val <= 10:
                        st.success("� **MAPE Eccellente (≤10%)** - Le tue previsioni sono molto accurate!")
                    elif mape_val <= 20:
                        st.success("🟡 **MAPE Buono (10-20%)** - Previsioni accurate con margine di miglioramento")
                        recommendations.append("💡 Per migliorare ulteriormente: considera l'aggiunta di regressori esterni o l'ottimizzazione dei parametri di stagionalità")
                    elif mape_val <= 50:
                        st.warning("🟠 **MAPE Accettabile (20-50%)** - Previsioni utilizzabili ma con errori significativi")
                        recommendations.append("⚠️ Azioni consigliate: verifica la presenza di outliers, considera modelli più complessi o aggiungi festività specifiche del tuo dominio")
                        quality_issues.append(f"MAPE elevato ({mape_val:.2f}%)")
                    else:
                        st.error("🔴 **MAPE Scarso (>50%)** - Previsioni poco affidabili")
                        recommendations.append("🚨 Azioni urgenti: rivedi la qualità dei dati, considera preprocessing aggiuntivo, prova modelli alternativi")
                        quality_issues.append(f"MAPE molto alto ({mape_val:.2f}%)")
                
                if 'r2' in metrics:
                    r2_val = metrics['r2']
                    if r2_val >= 0.8:
                        st.success("🟢 **R² Eccellente (≥0.8)** - Il modello spiega molto bene la varianza dei dati")
                    elif r2_val >= 0.6:
                        st.success("🟡 **R² Buono (0.6-0.8)** - Buona capacità esplicativa del modello")
                    elif r2_val >= 0.4:
                        st.warning("🟠 **R² Moderato (0.4-0.6)** - Il modello cattura solo parte della varianza")
                        recommendations.append("💡 Per migliorare R²: considera l'aggiunta di trend più complessi o componenti stagionali aggiuntive")
                        quality_issues.append(f"R² moderato ({r2_val:.3f})")
                    else:
                        st.error("🔴 **R² Basso (<0.4)** - Scarsa spiegazione della varianza")
                        recommendations.append("🚨 Considera: modelli più sofisticati, più dati storici, o regressori esterni rilevanti")
                        quality_issues.append(f"R² basso ({r2_val:.3f})")
                
                # Display actionable recommendations
                if recommendations:
                    st.markdown("### 💡 **Raccomandazioni Pratiche per Migliorare le Previsioni:**")
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                
                # Key parameter suggestions
                st.markdown("### 🔧 **Suggerimenti per Ottimizzazione Parametri:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **📅 Gestione Eventi e Festività:**
                    - Abilita festività nazionali se il tuo business è influenzato
                    - Aggiungi eventi speciali del tuo settore (saldi, campagne marketing)
                    - Considera festività regionali se operi in aree specifiche
                    """)
                
                with col2:
                    st.markdown("""
                    **⚙️ Tuning Tecnico del Modello:**
                    - Aumenta `changepoint_prior_scale` se i trend cambiano spesso
                    - Riduci `seasonality_prior_scale` se la stagionalità è molto regolare
                    - Prova crescita logistica per dati con saturazione naturale
                    """)
            
            st.markdown("---")
            
            # === DETAILED ANALYSIS SECTIONS ===
            st.markdown("## 📊 **Analisi Dettagliata delle Componenti**")
            
            # === PERFORMANCE METRICS SECTION ===
            st.markdown("### 📈 **1. Metriche di Performance del Modello**")
            
            if hasattr(st.session_state, 'last_forecast_metrics'):
                metrics = st.session_state.last_forecast_metrics
                
                # Display metrics in a comprehensive table
                col1, col2, col3, col4 = st.columns(4)
                
                # Key metrics display
                for i, (metric_name, metric_value) in enumerate(metrics.items()):
                    if isinstance(metric_value, (int, float)):
                        col = [col1, col2, col3, col4][i % 4]
                        
                        with col:
                            # Add interpretation for each metric
                            interpretation = ""
                            color = "normal"
                            
                            if metric_name.lower() == 'mape':
                                if metric_value <= 10:
                                    interpretation, color = "Eccellente", "green"
                                elif metric_value <= 20:
                                    interpretation, color = "Buono", "blue"
                                elif metric_value <= 50:
                                    interpretation, color = "Accettabile", "orange"
                                else:
                                    interpretation, color = "Scarso", "red"
                            elif metric_name.lower() == 'r2':
                                if metric_value >= 0.8:
                                    interpretation, color = "Eccellente", "green"
                                elif metric_value >= 0.6:
                                    interpretation, color = "Buono", "blue"
                                elif metric_value >= 0.4:
                                    interpretation, color = "Moderato", "orange"
                                else:
                                    interpretation, color = "Scarso", "red"
                            elif metric_name.lower() in ['mae', 'rmse']:
                                interpretation = "Errore assoluto"
                            
                            value_display = f"{metric_value:.3f}" + ("%" if metric_name.lower() in ['mape', 'smape'] else "")
                            st.metric(
                                label=f"{metric_name.upper()}",
                                value=value_display,
                                help=f"Interpretazione: {interpretation}"
                            )
                
                # Detailed interpretation guide
                st.markdown("""
                **📚 Guida all'Interpretazione delle Metriche:**
                
                - **MAPE (Mean Absolute Percentage Error)**: Percentuale di errore medio. Più basso = migliore.
                - **R² (Coefficient of Determination)**: Quanto bene il modello spiega i dati (0-1). Più alto = migliore.
                - **MAE (Mean Absolute Error)**: Errore medio in unità originali. Più basso = migliore.
                - **RMSE (Root Mean Square Error)**: Penalizza errori grandi. Più basso = migliore.
                """)
            else:
                st.info("🔄 Esegui un forecast per vedere le metriche di performance")
            
            st.markdown("---")
            
            # === RESIDUAL ANALYSIS SECTION ===
            st.markdown("### 🔍 **2. Analisi dei Residui (Errori del Modello)**")
            st.markdown("I residui mostrano la differenza tra valori reali e predetti. Residui ben distribuiti indicano un buon modello.")
            
            if (hasattr(st.session_state, 'last_prophet_result') and 
                hasattr(st.session_state, 'last_prophet_data')):
                
                try:
                    from src.modules.forecasting.prophet_diagnostics import ProphetDiagnosticAnalyzer
                    
                    # Run advanced diagnostics
                    analyzer = ProphetDiagnosticAnalyzer()
                    diagnostic_results = analyzer.analyze_forecast_quality(
                        st.session_state.last_prophet_result,
                        st.session_state.last_prophet_data['df'],
                        st.session_state.last_prophet_data['date_col'],
                        st.session_state.last_prophet_data['target_col']
                    )
                    
                    # Display diagnostic results
                    if 'residual_analysis' in diagnostic_results:
                        res_analysis = diagnostic_results['residual_analysis']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            mean_val = res_analysis.get('mean', 0)
                            st.metric("Media Residui", f"{mean_val:.4f}")
                            if abs(mean_val) < 0.01:
                                st.success("✅ Media vicina a zero (ottimo)")
                            else:
                                st.warning("⚠️ Media non zero (possibile bias)")
                        
                        with col2:
                            std_val = res_analysis.get('std', 0)
                            st.metric("Deviazione Standard", f"{std_val:.4f}")
                            st.info("💡 Minore = errori più consistenti")
                        
                        with col3:
                            skew_val = res_analysis.get('skewness', 0)
                            st.metric("Asimmetria", f"{skew_val:.4f}")
                            if abs(skew_val) < 0.5:
                                st.success("✅ Distribuzione simmetrica")
                            else:
                                st.warning("⚠️ Distribuzione asimmetrica")
                        
                        with col4:
                            kurt_val = res_analysis.get('kurtosis', 0)
                            st.metric("Curtosi", f"{kurt_val:.4f}")
                            if abs(kurt_val) < 1:
                                st.success("✅ Distribuzione normale")
                            else:
                                st.info("📊 Code spesse/sottili")
                        
                        # Actionable insights for residuals
                        st.markdown("""
                        **🎯 Cosa significano questi risultati:**
                        
                        - **Media vicina a zero**: Il modello non ha bias sistematici
                        - **Bassa deviazione standard**: Errori consistenti e prevedibili
                        - **Distribuzione simmetrica**: Il modello non sovra/sottostima sistematicamente
                        - **Curtosi normale**: Gli errori seguono una distribuzione gaussiana
                        
                        **💡 Se i residui non sono ottimali:**
                        - Bias sistematico → Aggiungi regressori esterni o festività
                        - Alta variabilità → Considera modelli più complessi o più dati
                        - Asimmetria → Verifica outliers o trasformazioni dei dati
                        """)
                
                except Exception as e:
                    st.warning(f"⚠️ Analisi residui non disponibile: {str(e)}")
                    st.info("💡 L'analisi dettagliata dei residui richiede un modello Prophet")
            else:
                st.info("🔄 Esegui un forecast Prophet per vedere l'analisi dei residui")
            
            st.markdown("---")
            
            # === FORECAST QUALITY ASSESSMENT ===
            st.markdown("### 🎯 **3. Valutazione Qualità delle Previsioni**")
            st.markdown("Analisi dell'affidabilità e stabilità delle previsioni future generate dal modello.")
            
            if hasattr(st.session_state, 'last_forecast_df'):
                forecast_df = st.session_state.last_forecast_df
                
                if not forecast_df.empty:
                    # Forecast statistics
                    forecast_col = 'yhat' if 'yhat' in forecast_df.columns else forecast_df.columns[-1]
                    forecast_values = forecast_df[forecast_col]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Media Previsioni", f"{forecast_values.mean():.2f}")
                    with col2:
                        st.metric("Deviazione Standard", f"{forecast_values.std():.2f}")
                    with col3:
                        cv = (forecast_values.std() / forecast_values.mean()) * 100 if forecast_values.mean() != 0 else 0
                        st.metric("Coefficiente Variazione", f"{cv:.2f}%")
                        if cv < 20:
                            st.success("✅ Previsioni stabili")
                        elif cv < 50:
                            st.warning("⚠️ Variabilità moderata")
                        else:
                            st.error("❌ Alta variabilità")
                    with col4:
                        # Trend analysis
                        if len(forecast_values) > 1:
                            trend_change = forecast_values.iloc[-1] - forecast_values.iloc[0]
                            trend_pct = (trend_change / forecast_values.iloc[0]) * 100 if forecast_values.iloc[0] != 0 else 0
                            st.metric("Trend Previsioni", f"{trend_pct:+.2f}%")
                            
                            if abs(trend_pct) < 5:
                                st.success("✅ Trend stabile")
                            elif abs(trend_pct) < 20:
                                st.warning("⚠️ Trend moderato")
                            else:
                                st.error("❌ Trend forte")
                    
                    # Confidence interval analysis (if available)
                    if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                        avg_interval_width = (forecast_df['yhat_upper'] - forecast_df['yhat_lower']).mean()
                        avg_forecast = forecast_df['yhat'].mean()
                        interval_ratio = (avg_interval_width / avg_forecast) * 100 if avg_forecast != 0 else 0
                        
                        st.markdown("#### 🎯 **Analisi Intervalli di Confidenza**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Larghezza Media Intervallo", f"{avg_interval_width:.2f}")
                        with col2:
                            st.metric("Rapporto Intervallo/Previsione", f"{interval_ratio:.2f}%")
                        
                        if interval_ratio < 20:
                            st.success("✅ **Intervalli stretti**: Previsioni ad alta precisione")
                            st.markdown("💡 Il modello è molto sicuro delle sue previsioni")
                        elif interval_ratio < 50:
                            st.warning("⚠️ **Intervalli moderati**: Previsioni con incertezza normale")
                            st.markdown("💡 Considera più dati storici per ridurre l'incertezza")
                        else:
                            st.error("❌ **Intervalli larghi**: Previsioni con alta incertezza")
                            st.markdown("🚨 Il modello ha poca confidenza - rivedi i parametri o i dati")
                    
                    # Practical forecast interpretation
                    st.markdown("""
                    **📊 Interpretazione Pratica delle Previsioni:**
                    
                    - **Coefficiente di Variazione basso (<20%)**: Previsioni consistenti e affidabili
                    - **Trend stabile (<5%)**: Andamento futuro prevedibile, buono per pianificazione
                    - **Intervalli di confidenza stretti**: Il modello è sicuro delle sue previsioni
                    
                    **⚠️ Segnali di Attenzione:**
                    - **Alta variabilità**: Le previsioni variano molto - considera più dati o parametri diversi
                    - **Trend forte (>20%)**: Cambiamenti drastici previsti - verifica la plausibilità
                    - **Intervalli larghi**: Alta incertezza - usa le previsioni con cautela
                    """)
                else:
                    st.info("🔄 Esegui un forecast per vedere la valutazione qualità")
            
            st.markdown("---")
            
            # === TIME SERIES DECOMPOSITION ===
            st.markdown("### 🔄 **4. Decomposizione della Serie Temporale**")
            st.markdown("Analisi delle componenti: trend (direzione), stagionalità (pattern ricorrenti) e residui (rumore casuale).")
            
            # Get cleaned data for decomposition
            if hasattr(st.session_state, 'cleaned_data') and st.session_state.cleaned_data is not None:
                df_clean = st.session_state.cleaned_data
                date_col = st.session_state.date_col
                target_col = st.session_state.target_col
                
                try:
                    # Prepare the series for decomposition
                    df_ts = df_clean.copy()
                    df_ts[date_col] = pd.to_datetime(df_ts[date_col])
                    df_ts = df_ts.sort_values(date_col)
                    df_ts = df_ts.set_index(date_col)
                    series = df_ts[target_col].dropna()
                    
                    # Determine seasonal period
                    seasonal_period = 7  # Default to weekly
                    min_periods = 2 * seasonal_period
                    
                    if len(series) >= min_periods:
                        from statsmodels.tsa.seasonal import seasonal_decompose
                        decomposition = seasonal_decompose(series, model='additive', period=seasonal_period)
                        
                        # Component analysis
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Trend analysis
                            trend_start = decomposition.trend.dropna().iloc[0]
                            trend_end = decomposition.trend.dropna().iloc[-1]
                            trend_change = trend_end - trend_start
                            trend_direction = "📈 Crescente" if trend_change > 0 else "📉 Decrescente" if trend_change < 0 else "➡️ Stabile"
                            
                            st.metric("Direzione Trend", trend_direction)
                            st.markdown(f"**Variazione**: {trend_change:+.2f}")
                            
                            if abs(trend_change) < series.std() * 0.1:
                                st.success("✅ Trend stabile")
                                st.markdown("💡 Serie senza grandi cambiamenti direzionali")
                            else:
                                st.info("📊 Trend significativo")
                                st.markdown("💡 Considera modelli che catturano bene i trend")
                        
                        with col2:
                            # Seasonal analysis
                            seasonal_strength = np.std(decomposition.seasonal.dropna()) / np.std(decomposition.observed.dropna()) * 100
                            st.metric("Forza Stagionalità", f"{seasonal_strength:.1f}%")
                            
                            if seasonal_strength > 30:
                                st.success("✅ Stagionalità forte")
                                st.markdown("💡 Pattern stagionali evidenti - importante includerli nel modello")
                            elif seasonal_strength > 15:
                                st.warning("⚠️ Stagionalità moderata")
                                st.markdown("💡 Pattern stagionali presenti ma non dominanti")
                            else:
                                st.info("📊 Stagionalità debole")
                                st.markdown("💡 Pochi pattern stagionali - focus su trend e regressori")
                        
                        with col3:
                            # Residual analysis
                            residual_variance = np.var(decomposition.resid.dropna())
                            st.metric("Varianza Residui", f"{residual_variance:.2f}")
                            
                            noise_ratio = residual_variance / np.var(series.dropna()) * 100
                            if noise_ratio < 20:
                                st.success("✅ Poco rumore")
                                st.markdown("💡 Dati puliti, buona qualità")
                            elif noise_ratio < 50:
                                st.warning("⚠️ Rumore moderato")
                                st.markdown("💡 Considera smoothing o preprocessing")
                            else:
                                st.error("❌ Molto rumore")
                                st.markdown("🚨 Dati molto rumorosi - migliora la qualità")
                        
                        # Decomposition insights
                        st.markdown("""
                        **🔍 Insights dalla Decomposizione:**
                        
                        **Trend (Direzione a lungo termine):**
                        - Crescente: Aumento costante nel tempo - buono per business in crescita
                        - Decrescente: Diminuzione costante - potrebbe richiedere interventi
                        - Stabile: Nessuna direzione chiara - focus su stagionalità e eventi
                        
                        **Stagionalità (Pattern ricorrenti):**
                        - Forte (>30%): Pattern molto regolari - sfrutta la prevedibilità
                        - Moderata (15-30%): Pattern presenti ma variabili
                        - Debole (<15%): Pochi pattern - considera fattori esterni
                        
                        **Residui (Variazioni casuali):**
                        - Bassi: Dati puliti, modello può essere preciso
                        - Alti: Molto rumore, difficile predire con precisione
                        """)
                    else:
                        st.warning(f"⚠️ Servono almeno {min_periods} osservazioni per la decomposizione")
                        st.info("💡 Raccogli più dati storici per un'analisi completa")
                
                except Exception as e:
                    st.error(f"❌ Errore nella decomposizione: {str(e)}")
                    st.info("💡 Potrebbe essere dovuto a pattern irregolari nei dati")
            else:
                st.info("🔄 Carica dati per vedere la decomposizione")
            
            st.markdown("---")
            
            # === SEASONALITY ANALYSIS ===
            st.markdown("### 📅 **5. Analisi Stagionalità Dettagliata**")
            st.markdown("Identificazione di pattern ricorrenti per ottimizzare le previsioni.")
            
            if hasattr(st.session_state, 'cleaned_data') and st.session_state.cleaned_data is not None:
                df_seasonal = st.session_state.cleaned_data.copy()
                df_seasonal[date_col] = pd.to_datetime(df_seasonal[date_col])
                df_seasonal = df_seasonal.sort_values(date_col)
                
                # Monthly seasonality analysis
                if len(df_seasonal) >= 24:
                    try:
                        df_seasonal['month'] = df_seasonal[date_col].dt.month
                        monthly_stats = df_seasonal.groupby('month')[target_col].agg(['mean', 'std', 'count']).reset_index()
                        
                        st.markdown("#### 📅 **Pattern Mensili**")
                        
                        # Find peak and low months
                        peak_month = monthly_stats.loc[monthly_stats['mean'].idxmax(), 'month']
                        low_month = monthly_stats.loc[monthly_stats['mean'].idxmin(), 'month']
                        month_names = ['Gen', 'Feb', 'Mar', 'Apr', 'Mag', 'Giu', 'Jul', 'Ago', 'Set', 'Ott', 'Nov', 'Dic']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mese di Picco", month_names[peak_month-1])
                            st.success(f"✅ Massimo: {monthly_stats.loc[monthly_stats['month']==peak_month, 'mean'].iloc[0]:.2f}")
                        with col2:
                            st.metric("Mese Minimo", month_names[low_month-1])
                            st.info(f"📊 Minimo: {monthly_stats.loc[monthly_stats['month']==low_month, 'mean'].iloc[0]:.2f}")
                        with col3:
                            seasonal_range = monthly_stats['mean'].max() - monthly_stats['mean'].min()
                            avg_value = monthly_stats['mean'].mean()
                            seasonality_impact = (seasonal_range / avg_value) * 100
                            st.metric("Impatto Stagionalità", f"{seasonality_impact:.1f}%")
                            
                            if seasonality_impact > 50:
                                st.success("✅ Stagionalità molto forte")
                            elif seasonality_impact > 25:
                                st.warning("⚠️ Stagionalità moderata")
                            else:
                                st.info("📊 Stagionalità debole")
                        
                        # Practical monthly insights
                        st.markdown(f"""
                        **💡 Insights Mensili Pratici:**
                        
                        - **Periodo di picco**: {month_names[peak_month-1]} - Pianifica risorse extra
                        - **Periodo minimo**: {month_names[low_month-1]} - Ottimizza costi operativi
                        - **Variazione stagionale**: {seasonality_impact:.1f}% - {"Molto significativa" if seasonality_impact > 50 else "Moderata" if seasonality_impact > 25 else "Limitata"}
                        
                        **🎯 Raccomandazioni:**
                        - Abilita la stagionalità annuale nel modello Prophet
                        - Considera eventi specifici nei mesi di picco
                        - Pianifica budget basandoti sui pattern identificati
                        """)
                    
                    except Exception as e:
                        st.warning("⚠️ Analisi mensile non disponibile")
                else:
                    st.info("📊 Servono almeno 24 mesi di dati per l'analisi mensile")
                
                # Weekly seasonality analysis
                if len(df_seasonal) >= 14:
                    try:
                        df_seasonal['dayofweek'] = df_seasonal[date_col].dt.dayofweek
                        weekly_stats = df_seasonal.groupby('dayofweek')[target_col].agg(['mean', 'std', 'count']).reset_index()
                        
                        st.markdown("#### 📊 **Pattern Settimanali**")
                        
                        # Find peak and low days
                        peak_day = weekly_stats.loc[weekly_stats['mean'].idxmax(), 'dayofweek']
                        low_day = weekly_stats.loc[weekly_stats['mean'].idxmin(), 'dayofweek']
                        day_names = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Giorno di Picco", day_names[peak_day])
                            st.success(f"✅ Massimo: {weekly_stats.loc[weekly_stats['dayofweek']==peak_day, 'mean'].iloc[0]:.2f}")
                        with col2:
                            st.metric("Giorno Minimo", day_names[low_day])
                            st.info(f"📊 Minimo: {weekly_stats.loc[weekly_stats['dayofweek']==low_day, 'mean'].iloc[0]:.2f}")
                        with col3:
                            weekly_range = weekly_stats['mean'].max() - weekly_stats['mean'].min()
                            weekly_avg = weekly_stats['mean'].mean()
                            weekly_impact = (weekly_range / weekly_avg) * 100 if weekly_avg != 0 else 0
                            st.metric("Variazione Settimanale", f"{weekly_impact:.1f}%")
                            
                            if weekly_impact > 30:
                                st.success("✅ Pattern settimanale forte")
                            elif weekly_impact > 15:
                                st.warning("⚠️ Pattern settimanale moderato")
                            else:
                                st.info("📊 Pattern settimanale debole")
                        
                        # Business insights for weekly patterns
                        st.markdown(f"""
                        **💼 Insights Settimanali per il Business:**
                        
                        - **Giorno più intenso**: {day_names[peak_day]} - Pianifica staffing massimo
                        - **Giorno più leggero**: {day_names[low_day]} - Riduci risorse operative
                        - **Impatto settimanale**: {weekly_impact:.1f}% - {"Molto rilevante" if weekly_impact > 30 else "Moderatamente rilevante" if weekly_impact > 15 else "Poco rilevante"}
                        
                        **📋 Azioni Consigliate:**
                        - Abilita stagionalità settimanale nel modello
                        - Adatta gli orari di lavoro ai pattern identificati
                        - Considera promozioni nei giorni meno intensi
                        """)
                    
                    except Exception as e:
                        st.warning("⚠️ Analisi settimanale non disponibile")
                else:
                    st.info("📊 Servono almeno 14 giorni di dati per l'analisi settimanale")
                
                # Clean up temporary columns
                temp_cols = ['month', 'dayofweek']
                for col in temp_cols:
                    if col in df_seasonal.columns:
                        df_seasonal.drop(col, axis=1, inplace=True)
            
            st.markdown("---")
            
            # === STATISTICAL LOG & ADVANCED METRICS ===
            st.markdown("### 📋 **6. Log Statistico e Metriche Avanzate**")
            st.markdown("Documentazione completa delle configurazioni e risultati per tracciabilità e debugging.")
            
            if hasattr(st.session_state, 'last_forecast_metrics'):
                # Current session info
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                model_name = st.session_state.get('selected_model', 'Unknown')
                
                st.markdown(f"""
                **📊 Sessione di Forecasting**
                - **Timestamp**: {current_time}
                - **Modello utilizzato**: {model_name}
                - **Dati**: {len(st.session_state.get('cleaned_data', []))} osservazioni
                """)
                
                # Performance summary
                metrics = st.session_state.last_forecast_metrics
                st.markdown("**🎯 Riassunto Performance:**")
                
                performance_summary = []
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        performance_summary.append(f"- **{metric_name.upper()}**: {metric_value:.4f}")
                
                for summary in performance_summary[:6]:  # Show first 6 metrics
                    st.markdown(summary)
                
                # Model configuration summary
                if hasattr(st.session_state, 'model_configs'):
                    model_configs = st.session_state.model_configs
                    if model_name in model_configs:
                        config = model_configs[model_name]
                        st.markdown("**⚙️ Configurazione Modello:**")
                        
                        config_summary = []
                        # Safeguard: only iterate if config is a dictionary
                        if isinstance(config, dict):
                            for param, value in config.items():
                                config_summary.append(f"- **{param}**: {value}")
                        elif hasattr(config, "__dict__"):
                            # Fallback: iterate over attributes of objects with __dict__
                            for param, value in vars(config).items():
                                config_summary.append(f"- **{param}**: {value}")
                        else:
                            config_summary.append("- Nessuna configurazione disponibile")
                        
                        for conf in config_summary[:8]:  # Show first 8 parameters
                            st.markdown(conf)
                
                # Data quality summary
                if hasattr(st.session_state, 'cleaned_data'):
                    df = st.session_state.cleaned_data
                    missing_values = df.isnull().sum().sum()
                    date_range_days = (df[date_col].max() - df[date_col].min()).days
                    
                    st.markdown(f"""
                    **📈 Qualità Dati:**
                    - **Record totali**: {len(df)}
                    - **Valori mancanti**: {missing_values}
                    - **Periodo coperto**: {date_range_days} giorni
                    - **Frequenza media**: {len(df)/max(date_range_days, 1):.2f} osservazioni/giorno
                    """)
                
                # Export options for documentation
                st.markdown("**📋 Esportazione Risultati:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Scarica Report Completo"):
                        # Create a comprehensive report
                        report_data = {
                            'timestamp': current_time,
                            'model': model_name,
                            'metrics': metrics,
                            'data_points': len(st.session_state.get('cleaned_data', [])),
                            'recommendations': recommendations if 'recommendations' in locals() else []
                        }
                        
                        import json
                        report_json = json.dumps(report_data, indent=2, default=str)
                        st.download_button(
                            label="💾 Download JSON Report",
                            data=report_json,
                            file_name=f"forecast_report_{current_time.replace(':', '-')}.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("📈 Salva Configurazione"):
                        # Save current configuration for reuse
                        config_data = {
                            'model': model_name,
                            'parameters': st.session_state.get('model_configs', {}),
                            'forecast_config': st.session_state.get('forecast_config', {}),
                            'timestamp': current_time
                        }
                        
                        import json
                        config_json = json.dumps(config_data, indent=2, default=str)
                        st.download_button(
                            label="💾 Download Config",
                            data=config_json,
                            file_name=f"forecast_config_{current_time.replace(':', '-')}.json",
                            mime="application/json"
                        )
            else:
                st.info("🔄 Esegui un forecast per vedere il log statistico")
            
            # Final recommendations section
            st.markdown("---")
            st.markdown("## 🚀 **Prossimi Passi Consigliati**")
            
            # Generate dynamic recommendations based on analysis
            final_recommendations = []
            
            if hasattr(st.session_state, 'last_forecast_metrics'):
                metrics = st.session_state.last_forecast_metrics
                
                if 'mape' in metrics and metrics['mape'] > 20:
                    final_recommendations.append("🎯 **Migliora MAPE**: Aggiungi festività, regressori esterni o ottimizza parametri stagionalità")
                
                if 'r2' in metrics and metrics['r2'] < 0.6:
                    final_recommendations.append("📊 **Aumenta R²**: Considera modelli più complessi o includi più variabili esplicative")
                
                final_recommendations.append("📅 **Monitora regolarmente**: Riallena il modello con nuovi dati ogni 1-3 mesi")
                final_recommendations.append("🔄 **Test A/B**: Confronta diversi modelli per trovare il migliore per i tuoi dati")
                final_recommendations.append("📈 **Validazione continua**: Monitora le performance su dati reali vs previsioni")
            else:
                final_recommendations = [
                    "🔄 **Esegui un forecast**: Inizia con il modello Prophet per una prima analisi",
                    "📊 **Analizza i pattern**: Usa la decomposizione per capire trend e stagionalità",
                    "🎯 **Configura parametri**: Ottimizza le impostazioni basandoti sui pattern identificati"
                ]
            
            for i, rec in enumerate(final_recommendations, 1):
                st.markdown(f"{i}. {rec}")
            
            st.success("🎉 **Analisi completata!** Usa questi insights per migliorare le tue previsioni e ottimizzare le decisioni di business.")
        
        else:
            st.info("⚠️ **Advanced Diagnostic disponibili dopo l'esecuzione del forecasting.** Esegui un modello di forecasting per vedere le diagnostiche avanzate.")
            
            st.markdown("""
            ### 🔬 Cosa troverai qui dopo l'esecuzione del forecast:
            
            - **🎯 Insights e Raccomandazioni**: Suggerimenti pratici per migliorare le previsioni
            - **📈 Metriche di Performance**: Analisi dettagliata di MAPE, R², MAE, RMSE
            - **🔍 Analisi Residui**: Valutazione degli errori del modello
            - **📊 Qualità Previsioni**: Affidabilità e stabilità delle previsioni future
            - **🔄 Decomposizione Serie**: Trend, stagionalità e componenti casuali
            - **📅 Stagionalità Dettagliata**: Pattern mensili e settimanali
            - **📋 Log Statistico**: Documentazione completa per tracciabilità
            
            **💡 Ogni sezione include suggerimenti pratici per utenti non esperti di forecasting!**
            """)
    
    # ==================================================================================
    # TAB 4: SUMMARY & EXPORT
    # ==================================================================================
    with tab4:
        if st.session_state.get('forecast_results_available', False) and hasattr(st.session_state, 'last_forecast_df'):
            st.markdown("## 📊 Summary & Export")
            st.markdown("Aggregazione dei risultati per mese e settimana con opzioni di download per analisi successive.")
            
            # Import aggregation functions
            from src.modules.utils.data_utils import (
                aggregate_forecast_by_month, aggregate_forecast_by_week,
                create_forecast_export_file
            )
            
            # Helper function: Filter out incomplete periods and zero values
            def filter_complete_periods(agg_df, period_col):
                """
                Filtra periodi parziali basandosi su colonna 'is_complete' calcolata in data_utils.
                Mantiene solo righe dove:
                - is_complete == True (periodi completi al 90%+ dei giorni attesi)
                - Oppure è forecast futuro (Actual=0 ma Forecast>0)
                Inoltre sostituisce 0 con NaN per non plottarli.
                """
                filtered_df = agg_df.copy()
                
                forecast_col = 'Forecast'
                actual_col = 'Actual'
                complete_col = 'is_complete'
                
                if forecast_col not in filtered_df.columns or actual_col not in filtered_df.columns:
                    return filtered_df
                
                # Se colonna is_complete esiste, usala per filtrare
                if complete_col in filtered_df.columns:
                    # Mantieni righe dove:
                    # 1. is_complete == True (periodo storico completo)
                    # 2. Oppure Actual <= 0 e Forecast > 0 (forecast futuro legittimo)
                    keep_mask = (filtered_df[complete_col] == True) | \
                                ((filtered_df[actual_col] <= 0) & (filtered_df[forecast_col] > 0))
                    filtered_df = filtered_df[keep_mask].copy()
                    
                    # Rimuovi colonne di supporto
                    if 'DayCount' in filtered_df.columns:
                        filtered_df = filtered_df.drop(columns=['DayCount', complete_col])
                else:
                    # Fallback: mantieni righe dove (Forecast > 0) OR (Actual > 0)
                    keep_mask = (filtered_df[forecast_col] > 0) | (filtered_df[actual_col] > 0)
                    filtered_df = filtered_df[keep_mask].copy()
                
                # Sostituisci 0 con NaN nelle colonne numeriche per non plottarle
                numeric_cols = ['Forecast', 'Actual', 'Lower Bound', 'Upper Bound']
                for col in numeric_cols:
                    if col in filtered_df.columns:
                        filtered_df[col] = filtered_df[col].replace(0, np.nan)
                
                return filtered_df if not filtered_df.empty else agg_df
            
            try:
                # Get data from session state
                forecast_df = st.session_state.last_forecast_df
                cleaned_data = st.session_state.cleaned_data
                date_col = st.session_state.date_col
                target_col = st.session_state.target_col
                
                # Pre-compute all exports and cache in session state to prevent data loss on download
                if 'summary_exports_cache' not in st.session_state:
                    st.session_state.summary_exports_cache = {
                        'monthly_agg': aggregate_forecast_by_month(forecast_df, cleaned_data, date_col, target_col),
                        'weekly_agg': aggregate_forecast_by_week(forecast_df, cleaned_data, date_col, target_col),
                        'forecast_csv': None
                    }
                
                # Get cached exports
                monthly_agg = st.session_state.summary_exports_cache['monthly_agg']
                weekly_agg = st.session_state.summary_exports_cache['weekly_agg']
                
                # Apply filtering to remove incomplete periods and zeros
                monthly_agg_filtered = filter_complete_periods(monthly_agg, 'Year-Month')
                weekly_agg_filtered = filter_complete_periods(weekly_agg, 'Year-Week')
                
                # ===== SECTION 1: MONTHLY & WEEKLY AGGREGATION =====
                
                
                # Limit periods for better readability
                monthly_agg_display = monthly_agg_filtered.tail(24)  # Max 24 months
                weekly_agg_display = weekly_agg_filtered.tail(56)    # Max 56 weeks
                
                # ===== MONTHLY CHART & TABLE =====
                
                
                # Monthly Chart
                fig_monthly = go.Figure()
                fig_monthly.add_trace(go.Scatter(
                    x=monthly_agg_display['Year-Month'],
                    y=monthly_agg_display['Forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#1f77b4', width=2.5, shape='spline'),
                ))
                
                # Add actual if available
                if 'Actual' in monthly_agg_display.columns and monthly_agg_display['Actual'].notna().any():
                    fig_monthly.add_trace(go.Scatter(
                        x=monthly_agg_display['Year-Month'],
                        y=monthly_agg_display['Actual'],
                        mode='lines',
                        name='Actual',
                        line=dict(color='#ff7f0e', width=2.5, shape='spline', dash='dash'),
                    ))
                
                # Add confidence interval
                fig_monthly.add_trace(go.Scatter(
                    x=monthly_agg_display['Year-Month'],
                    y=monthly_agg_display['Upper Bound'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                fig_monthly.add_trace(go.Scatter(
                    x=monthly_agg_display['Year-Month'],
                    y=monthly_agg_display['Lower Bound'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='95% Confidence Interval',
                    fillcolor='rgba(31, 119, 180, 0.2)'
                ))
                
                fig_monthly.update_layout(
                    title="Monthly Forecast vs Actual (Last 24 months)",
                    xaxis_title="Month",
                    yaxis_title="Volume",
                    height=450,
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Monthly Table
                st.markdown("**Dati Mensili Completi**")
                st.dataframe(monthly_agg, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # ===== WEEKLY CHART & TABLE =====
                
                
                # Weekly Chart
                fig_weekly = go.Figure()
                fig_weekly.add_trace(go.Scatter(
                    x=weekly_agg_display['Year-Week'],
                    y=weekly_agg_display['Forecast'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#2ca02c', width=2.5, shape='spline'),
                ))
                
                # Add actual if available
                if 'Actual' in weekly_agg_display.columns and weekly_agg_display['Actual'].notna().any():
                    fig_weekly.add_trace(go.Scatter(
                        x=weekly_agg_display['Year-Week'],
                        y=weekly_agg_display['Actual'],
                        mode='lines',
                        name='Actual',
                        line=dict(color='#d62728', width=2.5, shape='spline', dash='dash'),
                    ))
                
                # Add confidence interval
                fig_weekly.add_trace(go.Scatter(
                    x=weekly_agg_display['Year-Week'],
                    y=weekly_agg_display['Upper Bound'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False
                ))
                fig_weekly.add_trace(go.Scatter(
                    x=weekly_agg_display['Year-Week'],
                    y=weekly_agg_display['Lower Bound'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    name='95% Confidence Interval',
                    fillcolor='rgba(44, 160, 44, 0.2)'
                ))
                
                fig_weekly.update_layout(
                    title="Weekly Forecast vs Actual (Last 56 weeks)",
                    xaxis_title="Week",
                    yaxis_title="Volume",
                    height=450,
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_weekly, use_container_width=True)
                
                # Weekly Table
                st.markdown("**Dati Settimanali Completi**")
                st.dataframe(weekly_agg, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # ===== SECTION 2: DOWNLOAD OPTIONS =====
                st.markdown("### 📥 Download Dati")
                
                st.markdown("#### 📊 Serie Storica Completa")
                st.write("Scarica la serie di forecast completa con intervalli di confidenza.")
                
                try:
                    # Compute forecast export if not cached
                    if st.session_state.summary_exports_cache['forecast_csv'] is None:
                        st.session_state.summary_exports_cache['forecast_csv'] = create_forecast_export_file(forecast_df)
                    
                    st.download_button(
                        label="📥 Download Forecast (CSV)",
                        data=st.session_state.summary_exports_cache['forecast_csv'],
                        file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_forecast"
                    )
                except Exception as e:
                    st.error(f"Errore nel download forecast: {str(e)}")
                
                st.markdown("---")
                
                col_agg1, col_agg2 = st.columns(2)
                
                with col_agg1:
                    st.markdown("#### 📆 Aggregazione Mensile (CSV)")
                    monthly_csv = monthly_agg.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Dati Mensili (CSV)",
                        data=monthly_csv,
                        file_name=f"aggregation_monthly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_monthly"
                    )
                
                with col_agg2:
                    st.markdown("#### 📊 Aggregazione Settimanale (CSV)")
                    weekly_csv = weekly_agg.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Dati Settimanali (CSV)",
                        data=weekly_csv,
                        file_name=f"aggregation_weekly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_weekly"
                    )
                
                st.success("✅ Tutti i dati sono pronti per il download!")
                
            except Exception as e:
                st.error(f"❌ Errore nel tab Summary & Export: {str(e)}")
                st.info("💡 Assicurati di aver completato un forecast prima di accedere a questa sezione.")
        
        else:
            st.info("🔄 Summary & Export disponibile dopo l'esecuzione del forecasting.")
            st.markdown("""
            ### 📊 Cosa troverai qui dopo l'esecuzione del forecast:
            
            - **📆 Aggregazione Mensile**: Grafico e tabella con forecast vs consuntivo mensile
            - **📊 Aggregazione Settimanale**: Grafico e tabella con forecast vs consuntivo settimanale
            - **⏰ Matrice Slot 30min**: Media volumi per slot orario e giorno della settimana
            - **📥 Download Dati**: 
              - Serie storica completa con upper/lower bounds
              - Matrice slot 30min x giorni della settimana
              - Aggregazioni mensili e settimanali
            
            **💡 Tutte le tabelle e i grafici sono visualizzati contemporaneamente senza sovrascritture!**
            """)
if st.session_state.data_loaded and st.session_state.get('forecast_results_available', False):
    # Reset options and navigation - ENHANCED USER CONTROL
    st.markdown("---")
    st.subheader("🔄 Next Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run Another Forecast", type="secondary", key="run_another"):
            # Clear forecast results flag to return to data analysis view
            if 'forecast_results_available' in st.session_state:
                del st.session_state.forecast_results_available
            st.rerun()
    
    with col2:
        if st.button("📊 Back to Data Analysis", type="secondary", key="back_to_data"):
            # Clear forecast results flag to return to data analysis view
            if 'forecast_results_available' in st.session_state:
                del st.session_state.forecast_results_available
            st.rerun()
    
    with col3:
        if st.button("🆕 Load New Data", type="secondary", key="load_new_data"):
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