"""
Componenti UI riutilizzabili per l'applicazione di forecasting
Include widgets avanzati, sidebar, expander e controlli per parametri dei modelli
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

try:
    from .config import *
    from .data_utils import *
except ImportError:
    from config import *
    from data_utils import *

def render_data_upload_section() -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str], Dict[str, Any]]:
    """
    Renderizza la sezione di upload e configurazione dei dati
    
    Returns:
        Tuple: (dataframe, date_column, target_column, upload_config)
    """
    st.header("1. 📂 Data Source")
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ["Sample Dataset", "Upload CSV/Excel File"],
        help="Choose between using a generated sample dataset or uploading your own file"
    )
    
    df = None
    date_col = None
    target_col = None
    upload_config = {}
    
    if data_source == "Sample Dataset":
        st.info("📊 Using automatically generated sample dataset with trend and seasonality")
        df = generate_sample_data()
        date_col = 'date'
        target_col = 'volume'
        upload_config = {
            'source': 'sample',
            'delimiter': ',',
            'date_format': '%Y-%m-%d'
        }
        
    else:  # Upload file
        with st.expander("📂 File Upload Configuration", expanded=False):
            # File format detection
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your time series data file"
            )
            if uploaded_file:
                file_format = detect_file_format(uploaded_file)
                st.success(f"✅ Detected format: {file_format.upper()}")
            
            # Format-specific options
            if uploaded_file:
                file_format = detect_file_format(uploaded_file)
                
                if file_format == 'csv':
                    delimiter = st.selectbox(
                        "CSV Delimiter",
                        [",", ";", "|", "\t"],
                        help="Character that separates columns in your CSV file"
                    )
                    
                    encoding = st.selectbox(
                        "File Encoding",
                        ["utf-8", "latin-1", "cp1252"],
                        help="Text encoding of your CSV file"
                    )
                else:
                    delimiter = ","
                    encoding = "utf-8"
                
                # Date format selection
                user_friendly_format = st.selectbox(
                    "Date Format",
                    list(DATE_FORMATS.keys()),
                    help="Select the date format used in your file"
                )
                date_format = DATE_FORMATS[user_friendly_format]
                
                upload_config = {
                    'source': 'upload',
                    'delimiter': delimiter,
                    'encoding': encoding,
                    'date_format': date_format,
                    'file_format': file_format
                }
                
                # Load and preview data
                try:
                    if file_format == 'csv':
                        df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
                    
                    # Auto-detect date and target columns
                    detected_date, detected_target = auto_detect_columns(df)
                    
                    # Column selection
                    col1, col2 = st.columns(2)
                    with col1:
                        date_col = st.selectbox(
                            "📅 Date Column",
                            options=df.columns.tolist(),
                            index=df.columns.tolist().index(detected_date) if detected_date in df.columns else 0,
                            help="Column containing date/timestamp values"
                        )
                    with col2:
                        target_col = st.selectbox(
                            "🎯 Target Column", 
                            options=df.columns.tolist(),
                            index=df.columns.tolist().index(detected_target) if detected_target in df.columns else 1,
                            help="Column containing the values to forecast"
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
                    return None, None, None, upload_config
    
    return df, date_col, target_col, upload_config

def render_data_preview_section(df: pd.DataFrame, date_col: str, target_col: str, 
                               upload_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Renderizza la sezione di anteprima e statistiche dei dati
    
    Args:
        df: DataFrame dei dati
        date_col: Nome colonna data
        target_col: Nome colonna target
        upload_config: Configurazione upload
        
    Returns:
        pd.DataFrame: DataFrame processato
    """
    st.header("2. 📊 Data Preview & Statistics")
    
    # Convert date column
    if upload_config['source'] == 'upload':
        df[date_col] = pd.to_datetime(df[date_col], format=upload_config['date_format'], errors='coerce')
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Remove rows with invalid dates
    initial_rows = len(df)
    df = df.dropna(subset=[date_col])
    if len(df) < initial_rows:
        st.warning(f"⚠️ Removed {initial_rows - len(df)} rows with invalid dates")
    
    # Data statistics
    stats = get_data_statistics(df, date_col, target_col)
    
    # Display statistics - organizziamo in gruppi di 2 per la sidebar
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📈 Records", stats['total_records'])
        st.metric("� Mean", f"{stats['mean_value']:.2f}")
        st.metric("� Min", f"{stats['min_value']:.2f}")
    with col2:
        st.metric("� Days", f"{stats['date_range_days']}")
        st.metric("� Std", f"{stats['std_value']:.2f}")
        st.metric("📈 Max", f"{stats['max_value']:.2f}")
    
    # Missing values e duplicati in verticale
    if stats['missing_values'] > 0:
        st.metric("❌ Missing", f"{stats['missing_values']} ({stats['missing_percentage']:.1f}%)")
    if stats['duplicate_dates'] > 0:
        st.metric("🔄 Duplicates", stats['duplicate_dates'])
    
    # Data preview
    with st.expander("🔍 Data Preview", expanded=False):
        st.subheader("First 5 rows")
        st.dataframe(df.head(5), height=150)
        
        st.subheader("Last 5 rows") 
        st.dataframe(df.tail(5), height=150)
        
        # Basic time series plot
        st.subheader("📈 Quick Time Series Plot")
        fig = px.line(df.sort_values(date_col), x=date_col, y=target_col, 
                     title=f"Time Series: {target_col}")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    return df

def render_data_cleaning_section(df: pd.DataFrame, date_col: str, target_col: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Renderizza la sezione di pulizia e pre-processing dei dati
    
    Returns:
        Tuple: (cleaned_dataframe, cleaning_config)
    """
    st.header("3. 🧹 Data Cleaning & Preprocessing")
    
    cleaning_config = {}
    
    with st.expander("📅 Time Range Filter", expanded=False):
        # Date range selection
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date()
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                help="Filter data starting from this date"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=max_date,
                min_value=start_date,
                max_value=max_date,
                help="Filter data ending at this date"
            )
        
        # Apply date filter
        df_filtered = df[
            (df[date_col] >= pd.to_datetime(start_date)) & 
            (df[date_col] <= pd.to_datetime(end_date))
        ].copy()
        
        if len(df_filtered) != len(df):
            st.info(f"📊 Filtered from {len(df)} to {len(df_filtered)} records")
        
        cleaning_config['date_range'] = {'start': start_date, 'end': end_date}
    
    with st.expander("⏱️ Frequency & Aggregation", expanded=False):
        # Detect current frequency
        try:
            df_sorted = df_filtered.sort_values(date_col)
            detected_freq = pd.infer_freq(df_sorted[date_col])
            if detected_freq is None:
                detected_freq = "D"
        except:
            detected_freq = "D"
        
        current_freq_label = {v: k for k, v in FREQUENCY_MAP.items()}.get(detected_freq, "Daily")
        st.info(f"🔍 Detected frequency: {current_freq_label} ({detected_freq})")
        
        # Frequency selection
        target_freq_label = st.selectbox(
            "Target Frequency",
            list(FREQUENCY_MAP.keys()),
            index=list(FREQUENCY_MAP.keys()).index(current_freq_label) if current_freq_label in FREQUENCY_MAP else 0,
            help="Aggregate data to this frequency"
        )
        target_freq = FREQUENCY_MAP[target_freq_label]
        
        # Aggregation method
        aggregation_method = st.selectbox(
            "Aggregation Method",
            AGGREGATION_METHODS,
            help="How to combine multiple values when aggregating"
        )
        
        # Apply aggregation if frequency changed
        if target_freq != detected_freq:
            df_agg = aggregate_data(df_filtered, date_col, target_col, target_freq, aggregation_method)
            st.success(f"✅ Aggregated to {target_freq_label} frequency: {len(df_agg)} periods")
        else:
            df_agg = df_filtered.copy()
        
        cleaning_config['frequency'] = {
            'original': detected_freq,
            'target': target_freq,
            'aggregation_method': aggregation_method
        }
    
    with st.expander("❌ Missing Values", expanded=False):
        # Check for missing values
        missing_stats = get_missing_value_stats(df_agg, target_col)
        
        if missing_stats['count'] > 0:
            st.warning(f"⚠️ Found {missing_stats['count']} missing values ({missing_stats['percentage']:.1f}%)")
            
            missing_method = st.selectbox(
                "Missing Value Handling",
                MISSING_HANDLING_OPTIONS,
                help="How to handle missing values in the target column"
            )
            
            # Apply missing value handling
            df_agg = handle_missing_values(df_agg, target_col, missing_method)
            st.success(f"✅ Applied {missing_method} for missing values")
        else:
            st.success("✅ No missing values found")
            missing_method = "None"
        
        cleaning_config['missing_values'] = {
            'method': missing_method,
            'original_count': missing_stats['count']
        }
    
    with st.expander("📊 Outlier Detection & Handling", expanded=False):
        # Outlier detection
        outlier_stats = detect_outliers(df_agg, target_col)
        
        if outlier_stats['count'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.warning(f"⚠️ Found {outlier_stats['count']} outliers ({outlier_stats['percentage']:.1f}%)")
                
                # Show outlier boxplot
                fig = create_outlier_boxplot(df_agg, target_col)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                handle_outliers = st.checkbox("Handle Outliers", value=False)
                
                if handle_outliers:
                    outlier_method = st.selectbox(
                        "Outlier Handling Method",
                        ["Replace with median", "Replace with mean", "Remove outliers", "Winsorize (clip)"],
                        help="How to handle detected outliers"
                    )
                    
                    # Apply outlier handling
                    df_agg = handle_outliers_data(df_agg, target_col, outlier_method)
                    st.success(f"✅ Applied {outlier_method} for outliers")
                else:
                    outlier_method = "None"
        else:
            st.success("✅ No outliers detected")
            outlier_method = "None"
        
        cleaning_config['outliers'] = {
            'method': outlier_method,
            'original_count': outlier_stats.get('count', 0)
        }
    
    with st.expander("🔧 Additional Cleaning Options", expanded=False):
        remove_zeros = st.checkbox(
            "Remove Zero Values",
            value=False,
            help="Remove records where target value is exactly zero"
        )
        
        clip_negatives = st.checkbox(
            "Clip Negative Values",
            value=True,
            help="Convert negative values to zero"
        )
        
        remove_duplicates = st.checkbox(
            "Remove Duplicate Dates",
            value=True,
            help="Keep only the last value for duplicate dates"
        )
        
        validate_data = st.checkbox(
            "Data Validation",
            value=True,
            help="Perform additional data quality checks"
        )
        
        # Apply additional cleaning
        if remove_zeros:
            initial_len = len(df_agg)
            df_agg = df_agg[df_agg[target_col] != 0]
            if len(df_agg) < initial_len:
                st.info(f"📊 Removed {initial_len - len(df_agg)} zero values")
        
        if clip_negatives:
            negative_count = (df_agg[target_col] < 0).sum()
            if negative_count > 0:
                df_agg[target_col] = df_agg[target_col].clip(lower=0)
                st.info(f"📊 Clipped {negative_count} negative values to zero")
        
        if remove_duplicates:
            initial_len = len(df_agg)
            df_agg = df_agg.drop_duplicates(subset=[date_col], keep='last')
            if len(df_agg) < initial_len:
                st.info(f"📊 Removed {initial_len - len(df_agg)} duplicate dates")
        
        if validate_data:
            validation_results = validate_data_quality(df_agg, date_col, target_col)
            if validation_results['errors']:
                for error in validation_results['errors']:
                    st.error(f"❌ {error}")
            if validation_results['warnings']:
                for warning in validation_results['warnings']:
                    st.warning(f"⚠️ {warning}")
        
        cleaning_config['additional'] = {
            'remove_zeros': remove_zeros,
            'clip_negatives': clip_negatives,
            'remove_duplicates': remove_duplicates,
            'validate_data': validate_data
        }
    
    # Final data summary
    st.subheader("📋 Cleaned Data Summary")
    final_stats = get_data_statistics(df_agg, date_col, target_col)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Records", final_stats['total_records'])
        st.metric("📈 Mean", f"{final_stats['mean_value']:.2f}")
    with col2:
        st.metric("📅 Days", f"{final_stats['date_range_days']}")
        st.metric("� Std", f"{final_stats['std_value']:.2f}")
    
    return df_agg, cleaning_config

def render_model_selection_section() -> Tuple[str, Dict[str, Any]]:
    """
    Renderizza la sezione di selezione del modello e configurazione
    
    Returns:
        Tuple: (selected_model, model_configs)
    """
    st.header("4. 🤖 Model Selection & Configuration")
    
    # Model selection
    model = st.selectbox(
        "Select Forecasting Model",
        list(MODEL_DESCRIPTIONS.keys()),
        help="Choose the forecasting algorithm to use"
    )
    
    # Show model description
    st.info(f"ℹ️ {MODEL_DESCRIPTIONS[model]}")
    
    model_configs = {}
    
    # Model-specific configurations
    if model == "Prophet":
        model_configs["Prophet"] = render_prophet_config()
    elif model == "ARIMA":
        model_configs["ARIMA"] = render_arima_config()
    elif model == "SARIMA":
        model_configs["SARIMA"] = render_sarima_config()
    elif model == "Holt-Winters":
        model_configs["Holt-Winters"] = render_holtwinters_config()
    elif model == "Auto-Select":
        st.info("🤖 Auto-Select will test multiple models and choose the best performer")
        model_configs["Prophet"] = render_prophet_config()
        model_configs["ARIMA"] = render_arima_config() 
        model_configs["SARIMA"] = render_sarima_config()
        model_configs["Holt-Winters"] = render_holtwinters_config()
    
    return model, model_configs

def render_prophet_config() -> Dict[str, Any]:
    """Renderizza i parametri di configurazione per Prophet"""
    with st.expander("⚙️ Prophet Configuration", expanded=False):
        config = {}
        
        # Core parameters
        st.subheader("🔧 Core Parameters")
        
        config['seasonality_mode'] = st.selectbox(
            "Seasonality Mode",
            ['additive', 'multiplicative'],
            index=0,
            help=PARAMETER_TOOLTIPS['prophet']['seasonality_mode']
        )
        
        config['changepoint_prior_scale'] = st.slider(
            "Trend Flexibility",
            min_value=0.001,
            max_value=0.5,
            value=PROPHET_DEFAULTS['changepoint_prior_scale'],
            step=0.001,
            format="%.3f",
            help=PARAMETER_TOOLTIPS['prophet']['changepoint_prior_scale']
        )
        
        config['seasonality_prior_scale'] = st.slider(
            "Seasonality Strength",
            min_value=0.01,
            max_value=10.0,
            value=PROPHET_DEFAULTS['seasonality_prior_scale'],
            step=0.01,
            help=PARAMETER_TOOLTIPS['prophet']['seasonality_prior_scale']
        )
        
        config['uncertainty_samples'] = st.number_input(
            "Uncertainty Samples",
            min_value=100,
            max_value=2000,
            value=PROPHET_DEFAULTS['uncertainty_samples'],
            step=100,
            help=PARAMETER_TOOLTIPS['prophet']['uncertainty_samples']
        )
        
        # Seasonality configuration
        st.subheader("📊 Seasonality Configuration")
        
        config['yearly_seasonality'] = st.selectbox(
            "Yearly Seasonality",
            ['auto', True, False],
            index=0,
            help="Automatically detect or manually set yearly patterns"
        )
        
        config['weekly_seasonality'] = st.selectbox(
            "Weekly Seasonality", 
            ['auto', True, False],
            index=0,
            help="Automatically detect or manually set weekly patterns"
        )
        
        config['daily_seasonality'] = st.selectbox(
            "Daily Seasonality",
            ['auto', True, False],
            index=0,
            help="Automatically detect or manually set daily patterns"
        )
        
        # Custom seasonalities
        st.subheader("🔄 Custom Seasonalities")
        add_custom_seasonality = st.checkbox("Add Custom Seasonality", value=False)
        
        if add_custom_seasonality:
            custom_name = st.text_input("Seasonality Name", value="custom")
            custom_period = st.number_input("Period (days)", min_value=1, value=30)
            custom_fourier = st.number_input("Fourier Order", min_value=1, max_value=20, value=5)
            
            config['custom_seasonalities'] = [{
                'name': custom_name,
                'period': custom_period,
                'fourier_order': custom_fourier
            }]
        else:
            config['custom_seasonalities'] = []
        
        # Holidays
        st.subheader("🎉 Holidays Configuration")
        config['holidays_country'] = st.selectbox(
            "Country Holidays",
            [None, 'IT', 'US', 'UK', 'DE', 'FR', 'ES', 'CA'],
            help="Include national holidays for the selected country"
        )
        
        # Advanced options
        with st.expander("🔬 Advanced Options", expanded=False):
            config['growth'] = st.selectbox(
                "Growth Model",
                ['linear', 'logistic'],
                help="Linear for unlimited growth, logistic for growth with ceiling"
            )
            
            if config['growth'] == 'logistic':
                config['cap'] = st.number_input(
                    "Growth Ceiling",
                    min_value=1.0,
                    value=1000.0,
                    help="Maximum value that the series can reach"
                )
            
            config['mcmc_samples'] = st.number_input(
                "MCMC Samples",
                min_value=0,
                max_value=1000,
                value=0,
                help="Use MCMC for uncertainty estimation (0 = disabled)"
            )
        
        return config

def render_arima_config() -> Dict[str, Any]:
    """Renderizza i parametri di configurazione per ARIMA"""
    with st.expander("⚙️ ARIMA Configuration", expanded=False):
        config = {}
        
        # Auto-ARIMA
        config['auto_arima'] = st.checkbox(
            "Auto-ARIMA",
            value=True,
            help="Automatically find optimal parameters using statistical tests"
        )
        
        if not config['auto_arima']:
            st.subheader("📊 Manual ARIMA Parameters")
            
            config['p'] = st.number_input(
                "AR Order (p)",
                min_value=0,
                max_value=10,
                value=ARIMA_DEFAULTS['p'],
                help=PARAMETER_TOOLTIPS['arima']['p']
            )
            
            config['d'] = st.number_input(
                "Differencing (d)",
                min_value=0,
                max_value=5,
                value=ARIMA_DEFAULTS['d'],
                help=PARAMETER_TOOLTIPS['arima']['d']
            )
            
            config['q'] = st.number_input(
                "MA Order (q)",
                min_value=0,
                max_value=10,
                value=ARIMA_DEFAULTS['q'],
                help=PARAMETER_TOOLTIPS['arima']['q']
            )
        else:
            st.subheader("🔍 Auto-ARIMA Configuration")
            
            config['max_p'] = st.number_input("Max AR Order", min_value=1, max_value=10, value=5)
            config['max_d'] = st.number_input("Max Differencing", min_value=1, max_value=3, value=2)
            config['max_q'] = st.number_input("Max MA Order", min_value=1, max_value=10, value=5)
            
            config['information_criterion'] = st.selectbox(
                "Information Criterion",
                ['aic', 'bic', 'hqic'],
                help="Criterion for model selection"
            )
            config['stepwise'] = st.checkbox("Stepwise Search", value=True)
            config['suppress_warnings'] = st.checkbox("Suppress Warnings", value=True)
        
        # Advanced options
        with st.expander("🔬 Advanced Options", expanded=False):
            config['test'] = st.selectbox(
                "Stationarity Test",
                ['adf', 'kpss'],
                help="Test to check if data is stationary"
            )
            config['seasonal_test'] = st.selectbox(
                "Seasonality Test",
                ['ocsb', 'ch'],
                help="Test to detect seasonal patterns"
            )
            config['error_action'] = st.selectbox(
                "Error Handling",
                ['warn', 'ignore', 'raise'],
                index=1
            )
        
        return config

def render_sarima_config() -> Dict[str, Any]:
    """Renderizza i parametri di configurazione per SARIMA"""
    with st.expander("⚙️ SARIMA Configuration", expanded=False):
        config = {}
        
        # Auto-SARIMA
        config['auto_sarima'] = st.checkbox(
            "Auto-SARIMA",
            value=True,
            help="Automatically find optimal parameters"
        )
        
        if not config['auto_sarima']:
            # Non-seasonal parameters
            st.subheader("📊 Non-Seasonal Parameters")
            
            config['p'] = st.number_input("AR (p)", 0, 10, SARIMA_DEFAULTS['p'], key="sarima_p")
            config['d'] = st.number_input("Diff (d)", 0, 5, SARIMA_DEFAULTS['d'], key="sarima_d")
            config['q'] = st.number_input("MA (q)", 0, 10, SARIMA_DEFAULTS['q'], key="sarima_q")
            
            # Seasonal parameters
            st.subheader("🔄 Seasonal Parameters")
            
            config['P'] = st.number_input("Seasonal AR (P)", 0, 10, SARIMA_DEFAULTS['P'], key="sarima_P")
            config['D'] = st.number_input("Seasonal Diff (D)", 0, 5, SARIMA_DEFAULTS['D'], key="sarima_D")
            config['Q'] = st.number_input("Seasonal MA (Q)", 0, 10, SARIMA_DEFAULTS['Q'], key="sarima_Q")
            config['s'] = st.number_input("Season Length (s)", 1, 365, SARIMA_DEFAULTS['s'], key="sarima_s")
        else:
            st.subheader("🔍 Auto-SARIMA Configuration")
            
            config['max_p'] = st.number_input("Max p", 1, 5, 3, key="auto_sarima_max_p")
            config['max_d'] = st.number_input("Max d", 1, 3, 2, key="auto_sarima_max_d")
            config['max_q'] = st.number_input("Max q", 1, 5, 3, key="auto_sarima_max_q")
            
            config['max_P'] = st.number_input("Max P", 1, 3, 2, key="auto_sarima_max_P")
            config['max_D'] = st.number_input("Max D", 1, 2, 1, key="auto_sarima_max_D")
            config['max_Q'] = st.number_input("Max Q", 1, 3, 2, key="auto_sarima_max_Q")
            
            config['seasonal_period'] = st.number_input(
                "Seasonal Period",
                min_value=2,
                max_value=365,
                value=12,
                help="Length of seasonal cycle"
            )
        
        return config

def render_holtwinters_config() -> Dict[str, Any]:
    """Renderizza i parametri di configurazione per Holt-Winters"""
    with st.expander("⚙️ Holt-Winters Configuration", expanded=False):
        config = {}
        
        # Core parameters
        st.subheader("🔧 Core Parameters")
        
        config['trend'] = st.selectbox(
            "Trend Type",
            ['add', 'mul', None],
            index=0,
            help=PARAMETER_TOOLTIPS['holt_winters']['trend']
        )
        
        config['seasonal'] = st.selectbox(
            "Seasonal Type",
            ['add', 'mul', None],
            index=0,
            help=PARAMETER_TOOLTIPS['holt_winters']['seasonal']
        )
        
        config['damped_trend'] = st.checkbox(
            "Damped Trend",
            value=HOLTWINTERS_DEFAULTS['damped_trend'],
            help=PARAMETER_TOOLTIPS['holt_winters']['damped_trend']
        )
        
        config['seasonal_periods'] = st.number_input(
            "Seasonal Periods",
            min_value=2,
            max_value=365,
            value=HOLTWINTERS_DEFAULTS['seasonal_periods'],
            help=PARAMETER_TOOLTIPS['holt_winters']['seasonal_periods']
        )
        
        # Smoothing parameters
        st.subheader("📊 Smoothing Parameters")
        use_custom_smoothing = st.checkbox("Custom Smoothing Parameters", value=False)
        
        if use_custom_smoothing:
            config['smoothing_level'] = st.slider(
                "Alpha (Level)",
                0.0, 1.0,
                HOLTWINTERS_DEFAULTS['smoothing_level'],
                0.01,
                help="Smoothing parameter for level"
            )
            
            if config['trend'] is not None:
                config['smoothing_trend'] = st.slider(
                    "Beta (Trend)",
                    0.0, 1.0,
                    HOLTWINTERS_DEFAULTS['smoothing_trend'],
                    0.01,
                    help="Smoothing parameter for trend"
                )
            else:
                config['smoothing_trend'] = None
            
            if config['seasonal'] is not None:
                config['smoothing_seasonal'] = st.slider(
                    "Gamma (Seasonal)",
                    0.0, 1.0,
                    HOLTWINTERS_DEFAULTS['smoothing_seasonal'],
                    0.01,
                    help="Smoothing parameter for seasonal"
                )
            else:
                config['smoothing_seasonal'] = None
                
        else:
            config['smoothing_level'] = None
            config['smoothing_trend'] = None
            config['smoothing_seasonal'] = None
        
        # Advanced options
        with st.expander("🔬 Advanced Options", expanded=False):
            config['use_boxcox'] = st.checkbox(
                "Box-Cox Transformation",
                value=False,
                help="Apply Box-Cox transformation to stabilize variance"
            )
            
            if config['use_boxcox']:
                config['boxcox_lambda'] = st.slider(
                    "Box-Cox Lambda",
                    -2.0, 2.0, 0.0, 0.1,
                    help="Lambda parameter for Box-Cox (None = auto)"
                )
            else:
                config['boxcox_lambda'] = None
            
            config['remove_bias'] = st.checkbox(
                "Remove Bias",
                value=True,
                help="Remove bias from forecast"
            )
        
        return config

def render_external_regressors_section(df: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
    """
    Renderizza la sezione per la configurazione dei regressori esterni
    """
    st.header("5. 📈 External Regressors & Holidays")

    regressor_config = {
        'holidays_df': None,
        'selected_regressors': []
    }
    holidays_df = None

    with st.expander("📅 Holiday Effects", expanded=False):
        add_holidays = st.checkbox(
            "Add Holiday Effects",
            value=False,
            help="Include holiday effects in the forecast"
        )

        if add_holidays:
            holiday_source = st.radio(
                "Select holiday source",
                ["Select Country", "Manual Input"],
                horizontal=True,
                key="holiday_source"
            )

            if holiday_source == "Select Country":
                country_code = st.selectbox(
                    "Select Country for Holidays",
                    options=[None] + list(SUPPORTED_HOLIDAY_COUNTRIES.keys()),
                    format_func=lambda x: SUPPORTED_HOLIDAY_COUNTRIES.get(x, "None"),
                    help="Automatically include national holidays for the selected country."
                )
                if country_code:
                    try:
                        holidays_df = get_holidays_for_country(country_code, df[date_col])
                        st.success(f"✅ Loaded {len(holidays_df)} holidays for {SUPPORTED_HOLIDAY_COUNTRIES[country_code]}")
                        if not holidays_df.empty:
                            st.dataframe(holidays_df.head(), height=150)
                    except Exception as e:
                        st.error(f"Could not load holidays: {e}")
            
            elif holiday_source == "Manual Input":
                st.info("Enter holiday names and dates. Format: YYYY-MM-DD, Holiday Name")
                manual_holidays_text = st.text_area(
                    "Manual Holidays (one per line)",
                    "2024-01-01, New Year\n2024-12-25, Christmas",
                    height=150,
                    help="Provide holidays in 'YYYY-MM-DD, Holiday Name' format."
                )
                if st.button("Parse Manual Holidays"):
                    try:
                        holidays_df = parse_manual_holidays(manual_holidays_text)
                        st.success(f"✅ Parsed {len(holidays_df)} manual holidays.")
                        if not holidays_df.empty:
                            st.dataframe(holidays_df.head(), height=150)
                    except Exception as e:
                        st.error(f"❌ Error parsing holidays: {e}")

    if holidays_df is not None and not holidays_df.empty:
        regressor_config['holidays_df'] = holidays_df

    with st.expander("📊 External Regressors", expanded=False):
        # Get potential regressor columns
        potential_regressors = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in potential_regressors:
            potential_regressors.remove(target_col)
        
        if not potential_regressors:
            st.info("No numerical columns available to use as external regressors.")
            selected_regressors = []
        else:
            selected_regressors = st.multiselect(
                "Select External Regressors",
                options=potential_regressors,
                help="Select additional numerical columns to include in the model."
            )
        
        if selected_regressors:
            st.success(f"✅ Selected {len(selected_regressors)} regressors.")
            regressor_config['selected_regressors'] = selected_regressors

    return regressor_config

def render_forecast_horizon_section(df: pd.DataFrame, date_col: str, 
                                    freq: str) -> Tuple[int, pd.Timestamp, pd.Timestamp]:
    """
    Renderizza la sezione per la configurazione dell'orizzonte di previsione
    """
    st.header("6. 🎯 Forecast Horizon Configuration")
    
    # Default values
    default_horizon = 30
    min_horizon = 1
    max_horizon = 365
    
    with st.expander("⚙️ Horizon Parameters", expanded=False):
        # Forecast horizon
        horizon = st.number_input(
            "Forecast Horizon (days)",
            min_value=min_horizon,
            max_value=max_horizon,
            value=default_horizon,
            help="Number of days to forecast into the future"
        )
        
        # Frequency
        frequency = st.selectbox(
            "Forecast Frequency",
            ['D', 'W', 'M', 'Q', 'Y'],
            index=0,
            help="Frequency of the forecasted data"
        )
        
        # Start date
        min_date = df[date_col].min().date()
        max_date = df[date_col].max().date() + timedelta(days=horizon)
        start_date = st.date_input(
            "Forecast Start Date",
            value=max_date - timedelta(days=default_horizon),
            min_value=min_date,
            max_value=max_date,
            help="Start date for the forecast"
        )
    
    # Calculate end date based on horizon
    end_date = start_date + timedelta(days=horizon)
    
    st.info(f"📅 Forecasting from {start_date} to {end_date} ({horizon} days)")
    
    return horizon, start_date, end_date

def render_forecast_config_section() -> Dict[str, Any]:
    """
    Renderizza la sezione per la configurazione del forecast
    """
    st.header("7. 🎯 Forecast Settings")
    
    with st.expander("⚙️ Forecast Parameters", expanded=False):
        config = {}
        
        # Forecast horizon
        config['horizon'] = st.number_input(
            "Forecast Horizon",
            min_value=1,
            max_value=365,
            value=DEFAULT_HORIZON,
            help="Number of future periods to forecast"
        )
        
        # Frequency
        config['frequency'] = st.selectbox(
            "Forecast Frequency",
            ['D', 'W', 'M', 'Q', 'Y'],
            index=0,
            help="Frequency of the forecasted data"
        )
        
        # Aggregation method
        config['aggregation'] = st.selectbox(
            "Aggregation Method",
            AGGREGATION_METHODS,
            help="Method to aggregate data at the desired frequency"
        )
        
        # Confidence intervals
        config['include_confidence'] = st.checkbox(
            "Include Confidence Intervals",
            value=True,
            help="Generate prediction intervals for uncertainty quantification"
        )
        
        if config['include_confidence']:
            config['confidence_level'] = st.slider(
                "Confidence Level",
                min_value=0.5,
                max_value=0.99,
                value=DEFAULT_CI_LEVEL,
                step=0.01,
                format="%.2f",
                help="Confidence level for prediction intervals"
            )
            
            config['interval_width'] = st.selectbox(
                "Interval Width",
                [0.80, 0.90, 0.95, 0.99],
                index=0,
                help="Width of confidence intervals"
            )
    
    # Backtesting configuration
    with st.expander("📈 Backtesting & Validation", expanded=False):
        config['enable_backtesting'] = st.checkbox(
            "Enable Backtesting",
            value=True,
            help="Test model performance on historical data"
        )
        
        if config['enable_backtesting']:
            backtest_method = st.radio(
                "Validation Method",
                ["Simple Split", "Cross-Validation", "Rolling Window"],
                help="Method for splitting data into train/test sets"
            )
            
            config['backtest_method'] = backtest_method
            
            if backtest_method == "Simple Split":
                config['train_size'] = st.slider(
                    "Training Set Size",
                    min_value=0.5,
                    max_value=0.9,
                    value=0.8,
                    step=0.05,
                    format="%.2f",
                    help="Proportion of data to use for training"
                )
            
            elif backtest_method == "Cross-Validation":
                col1, col2 = st.columns(2)
                with col1:
                with col1:
                    config['cv_folds'] = st.number_input(
                        "Number of Folds",
                        min_value=3,
                        max_value=20,
                        value=5,
                        help="Number of cross-validation folds"
                    )
                with col2:
                    config['cv_horizon'] = st.number_input(
                        "CV Horizon",
                        min_value=1,
                        max_value=90,
                        value=30,
                        help="Forecast horizon for each fold"
                    )
            
            elif backtest_method == "Rolling Window":
                col1, col2 = st.columns(2)
                with col1:
                    config['window_size'] = st.number_input(
                        "Window Size",
                        min_value=50,
                        max_value=1000,
                        value=200,
                        help="Size of rolling training window"
                    )
                with col2:
                    config['step_size'] = st.number_input(
                        "Step Size",
                        min_value=1,
                        max_value=30,
                        value=7,
                        help="Number of periods to roll forward"
                    )
    
    # Metrics selection
    with st.expander("📏 Evaluation Metrics", expanded=False):
        available_metrics = list(METRICS_DEFINITIONS.keys())
        config['selected_metrics'] = st.multiselect(
            "Select Evaluation Metrics",
            options=available_metrics,
            default=['MAPE', 'MAE', 'RMSE'],
            help="Metrics to calculate for model evaluation"
        )
        
        # Show metric descriptions
        for metric in config['selected_metrics']:
            if metric in METRICS_DEFINITIONS:
                st.info(f"**{metric}**: {METRICS_DEFINITIONS[metric]}")
    
    return config

def render_output_config_section() -> Dict[str, Any]:
    """
    Renderizza la sezione per la configurazione dell'output
    """
    st.header("8. 💾 Output & Export")
    
    with st.expander("⚙️ Output Configuration", expanded=False):
        config = {}
        
        # Metrics selection
        config['show_metrics'] = st.checkbox(
            "Show Evaluation Metrics",
            value=True,
            help="Display evaluation metrics for the forecast"
        )
        
        if config['show_metrics']:
            config['metrics_list'] = st.multiselect(
                "Select Metrics",
                options=list(METRICS_DEFINITIONS.keys()),
                default=['MAPE', 'MAE', 'RMSE'],
                help="Metrics to display in the output"
            )
        
        # Visualization options
        config['show_forecast'] = st.checkbox(
            "Show Forecast Plot",
            value=True,
            help="Display the forecasted values plot"
        )
        
        config['show_components'] = st.checkbox(
            "Show Forecast Components",
            value=True,
            help="Display trend, seasonality, and other components"
        )
        
        config['show_residuals'] = st.checkbox(
            "Show Residual Analysis",
            value=True,
            help="Display residual plots for model diagnostics"
        )
        
        config['interactive_plots'] = st.checkbox(
            "Interactive Plots",
            value=True,
            help="Enable zoom, pan, and hover features in plots"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            config['plot_height'] = st.number_input(
                "Plot Height",
                min_value=300,
                max_value=800,
                value=500,
                help="Height of forecast plots in pixels"
            )
        
        with col2:
            config['color_scheme'] = st.selectbox(
                "Color Scheme",
                ['plotly', 'seaborn', 'custom'],
                help="Color palette for plots"
            )
    
    # Export options
    with st.expander("💾 Export Options", expanded=False):
        config['enable_export'] = st.checkbox(
            "Enable Export Features",
            value=True,
            help="Allow downloading results in various formats"
        )
        
        if config['enable_export']:
            config['export_formats'] = st.multiselect(
                "Export Formats",
                options=['CSV', 'Excel', 'JSON', 'PDF Report'],
                default=['CSV', 'Excel'],
                help="Available download formats for results"
            )
            
            if 'PDF Report' in config['export_formats']:
                config['pdf_options'] = {
                    'include_plots': st.checkbox("Include Plots in PDF", value=True),
                    'include_metrics': st.checkbox("Include Metrics Table", value=True),
                    'include_summary': st.checkbox("Include Model Summary", value=True)
                }
    
    # Real-time monitoring
    with st.expander("🔄 Real-time Features", expanded=False):
        config['auto_refresh'] = st.checkbox(
            "Auto Refresh Results",
            value=False,
            help="Automatically refresh forecast when parameters change"
        )
        
        if config['auto_refresh']:
            config['refresh_interval'] = st.selectbox(
                "Refresh Interval",
                [5, 10, 30, 60],
                index=1,
                help="Seconds between automatic refreshes"
            )
    
    return config
