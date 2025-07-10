"""
Enhanced Prophet module with advanced features:
- External regressors support
- Auto-tuning capabilities  
- Advanced seasonality configuration
- Holiday integration
- Comprehensive diagnostics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import warnings
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

try:
    import holidays as holiday_lib
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False
    st.warning("⚠️ Install 'holidays' package for holiday support: pip install holidays")

from .config import *
from .data_utils import *

def prepare_prophet_data(df: pd.DataFrame, date_col: str, target_col: str, 
                        external_regressors: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepara i dati per Prophet con colonne ds/y e regressori esterni
    
    Args:
        df: DataFrame dei dati
        date_col: Nome colonna data
        target_col: Nome colonna target
        external_regressors: Lista regressori esterni
        
    Returns:
        pd.DataFrame: Dati formattati per Prophet
    """
    prophet_df = df.copy()
    
    # Rinomina colonne per Prophet
    prophet_df = prophet_df.rename(columns={
        date_col: 'ds',
        target_col: 'y'
    })
    
    # Assicurati che ds sia datetime
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # Valida colonna y
    prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
    
    # Rimuovi NaN
    initial_len = len(prophet_df)
    prophet_df = prophet_df.dropna(subset=['ds', 'y'])
    
    if len(prophet_df) < initial_len:
        st.warning(f"⚠️ Removed {initial_len - len(prophet_df)} rows with missing values")
    
    # Aggiungi regressori esterni se specificati
    if external_regressors:
        for regressor in external_regressors:
            if regressor in df.columns:
                prophet_df[regressor] = pd.to_numeric(df[regressor], errors='coerce')
    
    return prophet_df

def create_holiday_dataframe(country: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Crea DataFrame delle festività per il paese specificato
    
    Args:
        country: Codice paese (IT, US, UK, etc.)
        df: DataFrame per determinare range date
        
    Returns:
        Optional[pd.DataFrame]: DataFrame festività o None
    """
    if not HOLIDAYS_AVAILABLE or country is None:
        return None
    
    try:
        # Determina range anni
        min_year = df['ds'].dt.year.min()
        max_year = df['ds'].dt.year.max() + 2  # Include 2 anni futuri
        
        # Mappa paesi
        country_map = {
            'IT': 'Italy',
            'US': 'UnitedStates', 
            'UK': 'UnitedKingdom',
            'DE': 'Germany',
            'FR': 'France',
            'ES': 'Spain',
            'CA': 'Canada'
        }
        
        if country not in country_map:
            st.warning(f"⚠️ Holiday calendar not available for {country}")
            return None
        
        # Ottieni festività
        country_holidays = getattr(holiday_lib, country_map[country])()
        
        holiday_dates = []
        holiday_names = []
        
        for year in range(min_year, max_year + 1):
            year_holidays = country_holidays[f'{year}-01-01':f'{year}-12-31']
            for date, name in year_holidays.items():
                holiday_dates.append(date)
                holiday_names.append(name)
        
        if holiday_dates:
            holidays_df = pd.DataFrame({
                'ds': pd.to_datetime(holiday_dates),
                'holiday': holiday_names,
                'lower_window': 0,
                'upper_window': 0
            })
            
            st.success(f"✅ Added {len(holidays_df)} holidays for {country}")
            return holidays_df
        
    except Exception as e:
        st.warning(f"⚠️ Error creating holiday calendar: {str(e)}")
    
    return None

def auto_tune_prophet_parameters(df: pd.DataFrame, 
                                initial_params: Dict[str, Any],
                                cv_horizon: int = 30) -> Dict[str, Any]:
    """
    Auto-tuning dei parametri Prophet usando cross-validation
    
    Args:
        df: Dati formattati per Prophet
        initial_params: Parametri iniziali
        cv_horizon: Orizzonte per CV
        
    Returns:
        Dict: Parametri ottimizzati
    """
    st.info("🔍 Running Prophet auto-tuning...")
    
    # Parametri da testare
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0, 100.0]
    }
    
    best_params = initial_params.copy()
    best_score = float('inf')
    
    progress_bar = st.progress(0)
    total_combinations = len(param_grid['changepoint_prior_scale']) * len(param_grid['seasonality_prior_scale'])
    current_combination = 0
    
    for cps in param_grid['changepoint_prior_scale']:
        for sps in param_grid['seasonality_prior_scale']:
            try:
                # Test parameters
                test_params = initial_params.copy()
                test_params['changepoint_prior_scale'] = cps
                test_params['seasonality_prior_scale'] = sps
                
                # Create and train model
                model = Prophet(**test_params)
                
                # Add regressors if any
                if 'external_regressors' in initial_params:
                    for regressor in initial_params['external_regressors']:
                        if regressor in df.columns:
                            model.add_regressor(regressor)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(df)
                
                # Cross-validation
                df_cv = cross_validation(
                    model, 
                    horizon=f'{cv_horizon} days',
                    initial=f'{len(df)//3} days',
                    period=f'{cv_horizon//2} days'
                )
                
                # Calculate performance metric
                df_metrics = performance_metrics(df_cv)
                score = df_metrics['rmse'].mean()
                
                if score < best_score:
                    best_score = score
                    best_params.update({
                        'changepoint_prior_scale': cps,
                        'seasonality_prior_scale': sps
                    })
                
            except Exception as e:
                # Skip this combination if it fails
                pass
            
            current_combination += 1
            progress_bar.progress(current_combination / total_combinations)
    
    progress_bar.empty()
    
    if best_score < float('inf'):
        st.success(f"✅ Auto-tuning completed! Best RMSE: {best_score:.2f}")
        st.info(f"🎯 Optimal parameters: changepoint_prior_scale={best_params['changepoint_prior_scale']}, seasonality_prior_scale={best_params['seasonality_prior_scale']}")
    else:
        st.warning("⚠️ Auto-tuning failed, using default parameters")
    
    return best_params

def build_enhanced_prophet_model(df: pd.DataFrame, params: Dict[str, Any], 
                                external_regressors: Optional[List[str]] = None) -> Prophet:
    """
    Costruisce un modello Prophet con tutte le funzionalità avanzate
    
    Args:
        df: Dati formattati per Prophet  
        params: Parametri modello
        external_regressors: Lista regressori esterni
        
    Returns:
        Prophet: Modello addestrato
    """
    # Extract parameters with defaults
    seasonality_mode = params.get('seasonality_mode', 'additive')
    changepoint_prior_scale = params.get('changepoint_prior_scale', 0.05)
    seasonality_prior_scale = params.get('seasonality_prior_scale', 10.0)
    uncertainty_samples = params.get('uncertainty_samples', 1000)
    
    # Seasonality settings
    yearly_seasonality = params.get('yearly_seasonality', 'auto')
    weekly_seasonality = params.get('weekly_seasonality', 'auto') 
    daily_seasonality = params.get('daily_seasonality', 'auto')
    
    # Growth settings
    growth = params.get('growth', 'linear')
    
    # Create holidays dataframe
    holidays_df = None
    if params.get('holidays_country'):
        holidays_df = create_holiday_dataframe(params['holidays_country'], df)
    
    # Initialize Prophet model
    model = Prophet(
        growth=growth,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        holidays=holidays_df,
        uncertainty_samples=uncertainty_samples,
        mcmc_samples=params.get('mcmc_samples', 0)
    )
    
    # Add custom seasonalities
    custom_seasonalities = params.get('custom_seasonalities', [])
    for seasonality in custom_seasonalities:
        model.add_seasonality(
            name=seasonality['name'],
            period=seasonality['period'],
            fourier_order=seasonality['fourier_order']
        )
    
    # Add external regressors
    if external_regressors:
        for regressor in external_regressors:
            if regressor in df.columns:
                model.add_regressor(regressor)
                st.info(f"📈 Added external regressor: {regressor}")
    
    # Add cap for logistic growth
    if growth == 'logistic':
        if 'cap' not in df.columns:
            cap_value = params.get('cap', df['y'].max() * 1.2)
            df['cap'] = cap_value
            st.info(f"📊 Set growth cap at {cap_value}")
    
    return model

def create_future_dataframe_with_regressors(model: Prophet, periods: int, freq: str,
                                          df: pd.DataFrame, 
                                          external_regressors: Optional[List[str]] = None,
                                          regressor_configs: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Crea future dataframe con valori per regressori esterni
    
    Args:
        model: Modello Prophet addestrato
        periods: Numero periodi futuri
        freq: Frequenza
        df: DataFrame originale
        external_regressors: Lista regressori
        regressor_configs: Configurazioni regressori
        
    Returns:
        pd.DataFrame: Future dataframe con regressori
    """
    # Create basic future dataframe
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Add cap for logistic growth
    if 'cap' in df.columns:
        future['cap'] = df['cap'].iloc[-1]  # Use last cap value
    
    # Add external regressors
    if external_regressors and regressor_configs:
        for regressor in external_regressors:
            if regressor in df.columns:
                config = regressor_configs.get(regressor, {})
                method = config.get('future_method', 'last_value')
                
                # Historical values
                historical_values = df[regressor].values
                future.loc[:len(df)-1, regressor] = historical_values
                
                # Future values
                if method == 'last_value':
                    future[regressor] = future[regressor].fillna(historical_values[-1])
                elif method == 'mean':
                    future[regressor] = future[regressor].fillna(df[regressor].mean())
                elif method == 'trend':
                    # Simple linear trend
                    trend_slope = (historical_values[-1] - historical_values[0]) / len(historical_values)
                    for i in range(len(df), len(future)):
                        future.loc[i, regressor] = historical_values[-1] + trend_slope * (i - len(df) + 1)
                elif method == 'manual':
                    future[regressor] = future[regressor].fillna(config.get('future_value', 0))
    
    return future

def run_prophet_cross_validation(model: Prophet, df: pd.DataFrame, 
                                cv_config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue cross-validation per Prophet
    
    Args:
        model: Modello Prophet addestrato
        df: Dati di training
        cv_config: Configurazione CV
        
    Returns:
        Tuple: (cv_results, metrics)
    """
    st.info("🔄 Running Prophet cross-validation...")
    
    # Configuration
    horizon = cv_config.get('cv_horizon', 30)
    n_folds = cv_config.get('cv_folds', 5)
    
    # Calculate periods
    initial_periods = len(df) // (n_folds + 1)
    period_length = len(df) // n_folds // 2
    
    try:
        # Run cross-validation
        df_cv = cross_validation(
            model,
            initial=f'{initial_periods} days',
            period=f'{period_length} days',
            horizon=f'{horizon} days',
            parallel='processes'
        )
        
        # Calculate metrics
        df_metrics = performance_metrics(df_cv)
        
        st.success(f"✅ Cross-validation completed: {len(df_cv)} predictions across {n_folds} folds")
        
        return df_cv, df_metrics
        
    except Exception as e:
        st.error(f"❌ Cross-validation failed: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def create_prophet_visualizations(model: Prophet, forecast: pd.DataFrame, 
                                 df: pd.DataFrame, target_col: str,
                                 output_config: Dict[str, Any]) -> None:
    """
    Crea visualizzazioni avanzate per Prophet
    
    Args:
        model: Modello Prophet
        forecast: Risultati forecast
        df: Dati originali 
        target_col: Nome colonna target
        output_config: Configurazione output
    """
    # Main forecast plot
    st.subheader("📈 Prophet Forecast")
    
    # Create interactive plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'], 
        mode='markers+lines',
        name='Historical Data',
        line=dict(color=PLOT_CONFIG['colors']['historical'])
    ))
    
    # Forecast
    forecast_future = forecast[forecast['ds'] > df['ds'].max()]
    
    if not forecast_future.empty:
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color=PLOT_CONFIG['colors']['forecast'], dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor=PLOT_CONFIG['colors']['confidence']
        ))
        
        # Add vertical line at forecast start
        fig.add_vline(
            x=df['ds'].max(),
            line_dash="dash",
            line_color="gray",
            annotation_text="Forecast Start"
        )
    
    fig.update_layout(
        title=f"Prophet Forecast - {target_col}",
        xaxis_title="Date",
        yaxis_title=target_col,
        height=output_config.get('plot_height', 500),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Components plot
    if output_config.get('show_components', True):
        st.subheader("📊 Forecast Components")
        
        components_fig = plot_components_plotly(model, forecast)
        components_fig.update_layout(height=600)
        st.plotly_chart(components_fig, use_container_width=True)
    
    # Residuals analysis
    if output_config.get('show_residuals', True):
        st.subheader("📉 Residual Analysis")
        
        # Calculate residuals
        forecast_historical = forecast[forecast['ds'] <= df['ds'].max()]
        residuals = df.set_index('ds')['y'] - forecast_historical.set_index('ds')['yhat']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals time series
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Scatter(
                x=residuals.index,
                y=residuals.values,
                mode='lines+markers',
                name='Residuals'
            ))
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            fig_residuals.update_layout(
                title="Residuals Over Time",
                xaxis_title="Date",
                yaxis_title="Residual"
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        with col2:
            # Residuals histogram
            fig_hist = px.histogram(
                x=residuals.values,
                nbins=30,
                title="Residuals Distribution"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

def run_prophet_model(df: pd.DataFrame, date_col: str, target_col: str,
                     horizon: int, selected_metrics: List[str],
                     params: Dict[str, Any], **kwargs) -> None:
    """
    Funzione principale per eseguire il modello Prophet con tutte le funzionalità
    
    Args:
        df: DataFrame dei dati
        date_col: Nome colonna data
        target_col: Nome colonna target  
        horizon: Orizzonte forecast
        selected_metrics: Metriche di valutazione
        params: Parametri modello
        **kwargs: Parametri aggiuntivi
    """
    st.markdown("## 🔮 Prophet Forecasting")
    
    try:
        # Prepare data
        prophet_df = prepare_prophet_data(df, date_col, target_col)
        
        if len(prophet_df) < 10:
            st.error("❌ Not enough data for Prophet forecasting (minimum 10 points required)")
            return
        
        # Auto-tuning if enabled
        if params.get('auto_tune', False):
            params = auto_tune_prophet_parameters(prophet_df, params)
        
        # External regressors
        external_regressors = kwargs.get('external_regressors', [])
        regressor_configs = kwargs.get('regressor_configs', {})
        
        # Build model
        model = build_enhanced_prophet_model(prophet_df, params, external_regressors)
        
        # Train model
        with st.spinner("🔄 Training Prophet model..."):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df)
        
        st.success("✅ Prophet model trained successfully!")
        
        # Create future dataframe
        future = create_future_dataframe_with_regressors(
            model, horizon, kwargs.get('freq', 'D'), 
            prophet_df, external_regressors, regressor_configs
        )
        
        # Generate forecast
        with st.spinner("🔮 Generating forecast..."):
            forecast = model.predict(future)
        
        # Cross-validation if enabled
        if kwargs.get('use_cv', False):
            cv_config = {
                'cv_horizon': kwargs.get('cv_horizon', 30),
                'cv_folds': kwargs.get('cv_folds', 5)
            }
            df_cv, df_metrics = run_prophet_cross_validation(model, prophet_df, cv_config)
            
            if not df_cv.empty:
                st.subheader("📊 Cross-Validation Results")
                st.dataframe(df_metrics)
        
        # Calculate metrics on historical data
        if len(selected_metrics) > 0:
            st.subheader("📏 Model Performance")
            
            # Get historical predictions
            forecast_hist = forecast[forecast['ds'] <= prophet_df['ds'].max()].set_index('ds')
            actual_hist = prophet_df.set_index('ds')
            
            # Calculate metrics
            metrics_results = {}
            for metric in selected_metrics:
                if metric == 'MAPE':
                    mape = np.mean(np.abs((actual_hist['y'] - forecast_hist['yhat']) / actual_hist['y'])) * 100
                    metrics_results[metric] = f"{mape:.2f}%"
                elif metric == 'MAE':
                    mae = mean_absolute_error(actual_hist['y'], forecast_hist['yhat'])
                    metrics_results[metric] = f"{mae:.2f}"
                elif metric == 'RMSE':
                    rmse = np.sqrt(mean_squared_error(actual_hist['y'], forecast_hist['yhat']))
                    metrics_results[metric] = f"{rmse:.2f}"
            
            # Display metrics
            cols = st.columns(len(metrics_results))
            for i, (metric, value) in enumerate(metrics_results.items()):
                with cols[i]:
                    st.metric(metric, value)
        
        # Visualizations
        output_config = kwargs.get('output_config', {
            'show_components': True,
            'show_residuals': True,
            'plot_height': 500
        })
        
        create_prophet_visualizations(model, forecast, prophet_df, target_col, output_config)
        
        # Forecast summary
        st.subheader("📋 Forecast Summary")
        
        future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
        
        if not future_forecast.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📅 Forecast Periods", len(future_forecast))
            with col2:
                st.metric("📈 Average Forecast", f"{future_forecast['yhat'].mean():.2f}")
            with col3:
                trend = "📈 Increasing" if future_forecast['yhat'].iloc[-1] > future_forecast['yhat'].iloc[0] else "📉 Decreasing"
                st.metric("📊 Trend", trend)
            
            # Forecast data table
            with st.expander("🔍 Detailed Forecast Data"):
                display_forecast = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                display_forecast.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
                display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m-%d')
                st.dataframe(display_forecast, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error in Prophet forecasting: {str(e)}")
        st.exception(e)
