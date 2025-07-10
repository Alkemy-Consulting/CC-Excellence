"""
Enhanced ARIMA module with advanced features:
- Auto-ARIMA parameter optimization
- Seasonal period auto-detection
- Advanced diagnostics and residual analysis
- Backtesting capabilities
- Interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any, Tuple
import warnings
from datetime import datetime, timedelta

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
            warnings.warn(f"{caller_name} requires matplotlib which is not installed.")
            return False
    
    # Monkey patch for pmdarima compatibility
    try:
        import sklearn.utils
        sklearn.utils.check_matplotlib_support = check_matplotlib_support
    except:
        pass

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
    try:
        import sklearn.utils.validation
        sklearn.utils.validation._check_fit_params = _check_fit_params
    except:
        pass

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.error("❌ Install statsmodels: pip install statsmodels")

try:
    import pmdarima as pm
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    st.warning("⚠️ Install pmdarima for auto-ARIMA: pip install pmdarima")

from .config import *
from .data_utils import *

def check_stationarity(series: pd.Series, significance_level: float = 0.05) -> Dict[str, Any]:
    """
    Verifica la stazionarietà della serie temporale usando test ADF e KPSS
    
    Args:
        series: Serie temporale
        significance_level: Livello di significatività
        
    Returns:
        Dict: Risultati test stazionarietà
    """
    results = {
        'is_stationary': False,
        'adf_statistic': None,
        'adf_pvalue': None,
        'kpss_statistic': None,
        'kpss_pvalue': None,
        'recommendation': ''
    }
    
    if not STATSMODELS_AVAILABLE:
        return results
    
    try:
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())
        results['adf_statistic'] = adf_result[0]
        results['adf_pvalue'] = adf_result[1]
        
        # KPSS test
        kpss_result = kpss(series.dropna(), regression='c')
        results['kpss_statistic'] = kpss_result[0]
        results['kpss_pvalue'] = kpss_result[1]
        
        # Determine stationarity
        adf_stationary = results['adf_pvalue'] < significance_level
        kpss_stationary = results['kpss_pvalue'] > significance_level
        
        if adf_stationary and kpss_stationary:
            results['is_stationary'] = True
            results['recommendation'] = "✅ Series is stationary"
        elif not adf_stationary and not kpss_stationary:
            results['is_stationary'] = False  
            results['recommendation'] = "❌ Series is non-stationary, consider differencing"
        else:
            results['recommendation'] = "⚠️ Tests disagree, manual inspection needed"
        
    except Exception as e:
        results['recommendation'] = f"❌ Error in stationarity tests: {str(e)}"
    
    return results

def suggest_differencing_order(series: pd.Series, max_d: int = 3) -> int:
    """
    Suggerisce l'ordine di differenziazione ottimale
    
    Args:
        series: Serie temporale
        max_d: Massimo ordine da testare
        
    Returns:
        int: Ordine di differenziazione suggerito
    """
    if not STATSMODELS_AVAILABLE:
        return 1
    
    for d in range(max_d + 1):
        if d == 0:
            test_series = series
        else:
            test_series = series.diff(d).dropna()
        
        if len(test_series) < 10:
            continue
            
        stationarity = check_stationarity(test_series)
        if stationarity['is_stationary']:
            return d
    
    return 1  # Default fallback

def detect_seasonal_periods(df: pd.DataFrame, date_col: str, target_col: str, 
                          freq: str) -> int:
    """
    Rileva automaticamente i periodi stagionali basati sulla frequenza
    
    Args:
        df: DataFrame
        date_col: Colonna data
        target_col: Colonna target
        freq: Frequenza dati
        
    Returns:
        int: Numero periodi stagionali
    """
    # Mappings based on frequency
    freq_to_seasonal = {
        'D': 7,    # Daily -> weekly seasonality
        'W': 52,   # Weekly -> yearly seasonality
        'M': 12,   # Monthly -> yearly seasonality
        'Q': 4,    # Quarterly -> yearly seasonality
        'Y': 1     # Yearly -> no seasonality
    }
    
    base_seasonal = freq_to_seasonal.get(freq, 12)
    
    # Try to detect from autocorrelation if enough data
    if len(df) >= base_seasonal * 3:
        try:
            # Calculate autocorrelation
            series = df[target_col].dropna()
            autocorr = [series.autocorr(lag) for lag in range(1, min(len(series)//2, base_seasonal*2))]
            
            # Find peaks in autocorrelation
            if autocorr:
                max_corr_idx = np.argmax(autocorr) + 1
                if autocorr[max_corr_idx-1] > 0.3:  # Significant correlation
                    return max_corr_idx
            
        except Exception:
            pass
    
    return base_seasonal

def run_auto_arima(series: pd.Series, seasonal_periods: int, 
                   max_p: int = 5, max_d: int = 2, max_q: int = 5,
                   max_P: int = 2, max_D: int = 1, max_Q: int = 2) -> Dict[str, Any]:
    """
    Esegue auto-ARIMA per trovare parametri ottimali
    
    Args:
        series: Serie temporale
        seasonal_periods: Periodi stagionali
        max_p, max_d, max_q: Limiti parametri non-stagionali
        max_P, max_D, max_Q: Limiti parametri stagionali
        
    Returns:
        Dict: Parametri ottimali e risultati
    """
    if not PMDARIMA_AVAILABLE:
        st.warning("⚠️ pmdarima not available, using default parameters")
        return {
            'p': 1, 'd': 1, 'q': 0,
            'P': 0, 'D': 0, 'Q': 0, 's': seasonal_periods,
            'aic': None, 'bic': None,
            'auto_arima_used': False
        }
    
    try:
        with st.spinner("🔍 Running auto-ARIMA optimization..."):
            # Run auto-ARIMA
            model = auto_arima(
                series.dropna(),
                start_p=0, start_q=0,
                max_p=max_p, max_d=max_d, max_q=max_q,
                start_P=0, start_Q=0,
                max_P=max_P, max_D=max_D, max_Q=max_Q,
                seasonal=seasonal_periods > 1,
                m=seasonal_periods if seasonal_periods > 1 else 1,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )
        
        # Extract parameters
        order = model.order
        seasonal_order = model.seasonal_order
        
        results = {
            'p': order[0], 'd': order[1], 'q': order[2],
            'P': seasonal_order[0], 'D': seasonal_order[1], 'Q': seasonal_order[2],
            's': seasonal_order[3],
            'aic': model.aic(),
            'bic': model.bic(),
            'auto_arima_used': True,
            'model': model
        }
        
        st.success(f"✅ Auto-ARIMA completed: ARIMA({order[0]},{order[1]},{order[2]}) x ({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})")
        st.info(f"📊 AIC: {results['aic']:.2f}, BIC: {results['bic']:.2f}")
        
        return results
        
    except Exception as e:
        st.error(f"❌ Auto-ARIMA failed: {str(e)}")
        return {
            'p': 1, 'd': 1, 'q': 0,
            'P': 0, 'D': 0, 'Q': 0, 's': seasonal_periods,
            'aic': None, 'bic': None,
            'auto_arima_used': False
        }

def fit_arima_model(series: pd.Series, order: Tuple[int, int, int], 
                   seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Any:
    """
    Fit del modello ARIMA/SARIMA
    
    Args:
        series: Serie temporale
        order: Ordine ARIMA (p,d,q)
        seasonal_order: Ordine stagionale (P,D,Q,s)
        
    Returns:
        Modello fitted
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for ARIMA modeling")
    
    try:
        if seasonal_order and seasonal_order[3] > 1:
            # SARIMA model
            model = ARIMA(series, order=order, seasonal_order=seasonal_order)
        else:
            # ARIMA model
            model = ARIMA(series, order=order)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = model.fit()
        
        return fitted_model
        
    except Exception as e:
        st.error(f"❌ Error fitting ARIMA model: {str(e)}")
        raise

def create_arima_diagnostics(fitted_model: Any, series: pd.Series) -> None:
    """
    Crea diagnostici per il modello ARIMA
    
    Args:
        fitted_model: Modello ARIMA fitted
        series: Serie temporale originale
    """
    st.subheader("🔬 Model Diagnostics")
    
    try:
        # Get residuals
        residuals = fitted_model.resid
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Residuals plot
            fig_resid = go.Figure()
            fig_resid.add_trace(go.Scatter(
                x=residuals.index,
                y=residuals.values,
                mode='lines+markers',
                name='Residuals',
                line=dict(width=1)
            ))
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            fig_resid.update_layout(
                title="Residuals Over Time",
                xaxis_title="Time",
                yaxis_title="Residuals",
                height=300
            )
            st.plotly_chart(fig_resid, use_container_width=True)
            
            # Q-Q plot approximation (histogram)
            fig_qq = px.histogram(
                x=residuals,
                nbins=30,
                title="Residuals Distribution"
            )
            fig_qq.update_layout(height=300)
            st.plotly_chart(fig_qq, use_container_width=True)
        
        with col2:
            # ACF plot approximation
            lags = range(1, min(21, len(residuals)//4))
            acf_values = [residuals.autocorr(lag) for lag in lags]
            
            fig_acf = go.Figure()
            fig_acf.add_trace(go.Bar(
                x=list(lags),
                y=acf_values,
                name='ACF'
            ))
            fig_acf.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Upper threshold")
            fig_acf.add_hline(y=-0.2, line_dash="dash", line_color="red", annotation_text="Lower threshold")
            fig_acf.update_layout(
                title="Autocorrelation Function of Residuals",
                xaxis_title="Lag",
                yaxis_title="ACF",
                height=300
            )
            st.plotly_chart(fig_acf, use_container_width=True)
            
            # Ljung-Box test
            try:
                lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
                significant_lags = (lb_test['lb_pvalue'] < 0.05).sum()
                
                if significant_lags == 0:
                    st.success("✅ Ljung-Box test: No significant autocorrelation in residuals")
                else:
                    st.warning(f"⚠️ Ljung-Box test: {significant_lags} significant lags detected")
                    
            except:
                st.info("ℹ️ Ljung-Box test not available")
        
        # Model summary
        st.subheader("📊 Model Summary")
        
        summary_stats = {
            'AIC': fitted_model.aic,
            'BIC': fitted_model.bic,
            'Log-Likelihood': fitted_model.llf,
            'Parameters': len(fitted_model.params)
        }
        
        cols = st.columns(len(summary_stats))
        for i, (stat, value) in enumerate(summary_stats.items()):
            with cols[i]:
                st.metric(stat, f"{value:.2f}")
        
    except Exception as e:
        st.warning(f"⚠️ Error creating diagnostics: {str(e)}")

def create_arima_forecast_plot(df: pd.DataFrame, fitted_model: Any, 
                              forecast_values: np.ndarray, 
                              conf_int: np.ndarray,
                              horizon: int, freq: str,
                              date_col: str, target_col: str) -> None:
    """
    Crea il plot del forecast ARIMA
    
    Args:
        df: DataFrame originale
        fitted_model: Modello fitted
        forecast_values: Valori forecast
        conf_int: Intervalli di confidenza
        horizon: Orizzonte forecast
        freq: Frequenza
        date_col: Colonna data
        target_col: Colonna target
    """
    st.subheader("📈 ARIMA Forecast")
    
    # Create future dates
    last_date = df[date_col].max()
    if freq == 'D':
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
    elif freq == 'W':
        future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=horizon, freq='W')
    elif freq == 'M':
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='M')
    else:
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
    
    # Create plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[target_col],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color=PLOT_CONFIG['colors']['historical'])
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color=PLOT_CONFIG['colors']['forecast'], dash='dash')
    ))
    
    # Confidence intervals
    if conf_int is not None and len(conf_int) > 0:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=conf_int[:, 1],  # Upper bound
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=conf_int[:, 0],  # Lower bound
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='Confidence Interval',
            fillcolor=PLOT_CONFIG['colors']['confidence']
        ))
    
    # Add vertical line at forecast start
    fig.add_vline(
        x=last_date,
        line_dash="dash",
        line_color="gray",
        annotation_text="Forecast Start"
    )
    
    fig.update_layout(
        title=f"ARIMA Forecast - {target_col}",
        xaxis_title="Date",
        yaxis_title=target_col,
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def perform_arima_backtesting(df: pd.DataFrame, date_col: str, target_col: str,
                             params: Dict[str, Any], train_size: float = 0.8) -> Dict[str, float]:
    """
    Esegue backtesting per ARIMA
    
    Args:
        df: DataFrame
        date_col: Colonna data
        target_col: Colonna target
        params: Parametri modello
        train_size: Proporzione dati di training
        
    Returns:
        Dict: Metriche di performance
    """
    if not STATSMODELS_AVAILABLE:
        return {}
    
    # Split data
    split_idx = int(len(df) * train_size)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    
    if len(test_data) == 0:
        st.warning("⚠️ Not enough data for backtesting")
        return {}
    
    try:
        # Prepare series
        train_series = train_data[target_col]
        test_series = test_data[target_col]
        
        # Fit model on training data
        order = (params['p'], params['d'], params['q'])
        
        if params.get('s', 0) > 1:
            seasonal_order = (params['P'], params['D'], params['Q'], params['s'])
            fitted_model = fit_arima_model(train_series, order, seasonal_order)
        else:
            fitted_model = fit_arima_model(train_series, order)
        
        # Generate forecast
        forecast_result = fitted_model.forecast(steps=len(test_data))
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(test_series, forecast_result)
        rmse = np.sqrt(mean_squared_error(test_series, forecast_result))
        mape = np.mean(np.abs((test_series - forecast_result) / test_series)) * 100
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
        
        st.subheader("📊 Backtesting Results")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{mae:.2f}")
        with col2:
            st.metric("RMSE", f"{rmse:.2f}")
        with col3:
            st.metric("MAPE", f"{mape:.2f}%")
        
        # Plot actual vs predicted
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_data[date_col],
            y=test_series,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=test_data[date_col],
            y=forecast_result,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Backtesting: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title=target_col,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return metrics
        
    except Exception as e:
        st.error(f"❌ Backtesting failed: {str(e)}")
        return {}

def run_arima_model(df: pd.DataFrame, date_col: str, target_col: str,
                   horizon: int, selected_metrics: List[str],
                   params: Dict[str, Any], **kwargs) -> None:
    """
    Funzione principale per eseguire il modello ARIMA/SARIMA
    
    Args:
        df: DataFrame dei dati
        date_col: Nome colonna data
        target_col: Nome colonna target
        horizon: Orizzonte forecast
        selected_metrics: Metriche di valutazione
        params: Parametri modello
        **kwargs: Parametri aggiuntivi
    """
    st.markdown("## 📊 ARIMA/SARIMA Forecasting")
    
    if not STATSMODELS_AVAILABLE:
        st.error("❌ statsmodels package is required for ARIMA modeling")
        return
    
    try:
        # Prepare time series
        series = df[target_col].copy()
        
        if len(series) < 20:
            st.error("❌ Not enough data for ARIMA modeling (minimum 20 points required)")
            return
        
        # Detect frequency
        freq = kwargs.get('freq', 'D')
        
        # Stationarity analysis
        st.subheader("🔍 Stationarity Analysis")
        stationarity_results = check_stationarity(series)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ADF p-value", f"{stationarity_results['adf_pvalue']:.4f}" if stationarity_results['adf_pvalue'] else "N/A")
        with col2:
            st.metric("KPSS p-value", f"{stationarity_results['kpss_pvalue']:.4f}" if stationarity_results['kpss_pvalue'] else "N/A")
        
        st.info(stationarity_results['recommendation'])
        
        # Auto-ARIMA or manual parameters
        if params.get('auto_arima', True):
            # Detect seasonal periods
            seasonal_periods = detect_seasonal_periods(df, date_col, target_col, freq)
            st.info(f"🔄 Detected seasonal periods: {seasonal_periods}")
            
            # Run auto-ARIMA
            auto_results = run_auto_arima(
                series, seasonal_periods,
                max_p=params.get('max_p', 5),
                max_d=params.get('max_d', 2), 
                max_q=params.get('max_q', 5),
                max_P=params.get('max_P', 2),
                max_D=params.get('max_D', 1),
                max_Q=params.get('max_Q', 2)
            )
            
            # Use auto-ARIMA results
            order = (auto_results['p'], auto_results['d'], auto_results['q'])
            seasonal_order = (auto_results['P'], auto_results['D'], auto_results['Q'], auto_results['s'])
            
        else:
            # Manual parameters
            order = (params['p'], params['d'], params['q'])
            
            if 's' in params and params['s'] > 1:
                seasonal_order = (params['P'], params['D'], params['Q'], params['s'])
            else:
                seasonal_order = None
        
        # Display model configuration
        st.subheader("⚙️ Model Configuration")
        if seasonal_order and seasonal_order[3] > 1:
            st.info(f"📊 SARIMA({order[0]},{order[1]},{order[2]}) x ({seasonal_order[0]},{seasonal_order[1]},{seasonal_order[2]},{seasonal_order[3]})")
        else:
            st.info(f"📊 ARIMA({order[0]},{order[1]},{order[2]})")
        
        # Fit model
        with st.spinner("🔄 Training ARIMA model..."):
            fitted_model = fit_arima_model(series, order, seasonal_order)
        
        st.success("✅ ARIMA model trained successfully!")
        
        # Model diagnostics
        create_arima_diagnostics(fitted_model, series)
        
        # Generate forecast
        with st.spinner("🔮 Generating forecast..."):
            forecast_result = fitted_model.forecast(steps=horizon)
            forecast_conf_int = fitted_model.get_forecast(steps=horizon).conf_int()
        
        # Create forecast visualization
        create_arima_forecast_plot(
            df, fitted_model, forecast_result, forecast_conf_int.values,
            horizon, freq, date_col, target_col
        )
        
        # Backtesting if enabled
        if kwargs.get('enable_backtesting', True):
            backtest_metrics = perform_arima_backtesting(df, date_col, target_col, {
                'p': order[0], 'd': order[1], 'q': order[2],
                'P': seasonal_order[0] if seasonal_order else 0,
                'D': seasonal_order[1] if seasonal_order else 0, 
                'Q': seasonal_order[2] if seasonal_order else 0,
                's': seasonal_order[3] if seasonal_order else 0
            })
        
        # Forecast summary
        st.subheader("📋 Forecast Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📅 Forecast Periods", horizon)
        with col2:
            st.metric("📈 Average Forecast", f"{forecast_result.mean():.2f}")
        with col3:
            trend = "📈 Increasing" if forecast_result.iloc[-1] > forecast_result.iloc[0] else "📉 Decreasing"
            st.metric("📊 Trend", trend)
        
        # Forecast data table
        with st.expander("🔍 Detailed Forecast Data"):
            # Create future dates
            last_date = df[date_col].max()
            if freq == 'D':
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
            elif freq == 'W':
                future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=horizon, freq='W')
            elif freq == 'M':
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='M')
            else:
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq='D')
            
            forecast_df = pd.DataFrame({
                'Date': future_dates.strftime('%Y-%m-%d'),
                'Forecast': forecast_result.round(2),
                'Lower Bound': forecast_conf_int.iloc[:, 0].round(2),
                'Upper Bound': forecast_conf_int.iloc[:, 1].round(2)
            })
            
            st.dataframe(forecast_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error in ARIMA forecasting: {str(e)}")
        st.exception(e)
