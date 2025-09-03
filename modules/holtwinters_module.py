import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import io
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.modules.utils.metrics_module import compute_all_metrics

def run_holt_winters_model(df: pd.DataFrame, date_col: str, target_col: str, horizon: int, selected_metrics: list, params: dict, return_metrics=False):
    """
    Runs the Holt-Winters model using parameters passed from the UI.
    """
    if not return_metrics:
        st.subheader("Holt-Winters Forecast")

    # Estrai i parametri dal dizionario
    trend_type = params.get('trend_type', 'add')
    seasonal_type = params.get('seasonal_type', 'add')
    damped_trend = params.get('damped_trend', True)
    seasonal_periods = params.get('seasonal_periods', 12)
    use_custom = params.get('use_custom', False)

    smoothing_level, smoothing_trend, smoothing_seasonal, optimized = None, None, None, True
    if use_custom:
        optimized = False
        smoothing_level = params.get('smoothing_level')
        smoothing_trend = params.get('smoothing_trend')
        smoothing_seasonal = params.get('smoothing_seasonal')

    try:
        series = df.set_index(date_col)[target_col]
        fitted, forecast, model_params = holt_winters_forecast(
            series,
            forecast_periods=horizon,
            seasonal_periods=seasonal_periods,
            trend=trend_type,
            seasonal=seasonal_type,
            damped_trend=damped_trend,
            smoothing_level=smoothing_level,
            smoothing_trend=smoothing_trend,
            smoothing_seasonal=smoothing_seasonal,
            optimized=optimized
        )

        if not return_metrics:
            st.success("Modello Holt-Winters addestrato con successo.")
            with st.expander("Vedi parametri del modello ottimizzati"):
                st.json({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in model_params.items() if v is not None})

        # Calcolo e visualizzazione metriche
        if not return_metrics:
            st.write("### Evaluation Metrics (su dati storici)")
        metrics_results = compute_all_metrics(series, fitted)
        if not selected_metrics: selected_metrics = ["MAE", "RMSE", "MAPE"]
        
        if not return_metrics:
            cols = st.columns(len(selected_metrics))
            for i, metric in enumerate(selected_metrics):
                value = metrics_results.get(metric)
                if value is not None:
                    format_str = "{:.0f}%" if metric in ["MAPE", "SMAPE"] else "{:.3f}"
                    cols[i].metric(metric, format_str.format(value))

            # Grafico con Plotly
            st.write("### Grafico Forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df[date_col], y=df[target_col], mode='lines', name='Storico', line=dict(color='#1f77b4')))
            fig.add_trace(go.Scatter(x=fitted.index, y=fitted, mode='lines', name='Fitted', line=dict(color='#ff7f0e', dash='dash')))
            fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast', line=dict(color='#d62728')))
            st.plotly_chart(fig, width='stretch')

            # Bottone di download
            if st.button("📥 Scarica Forecast in Excel", key="holtwinters_download_btn"):
                forecast_df = pd.DataFrame({'ds': forecast.index, 'yhat': forecast.values})
                buffer = io.BytesIO()
                forecast_df.to_excel(buffer, index=False, engine='openpyxl')
                buffer.seek(0)
                st.download_button(
                    label="Download .xlsx",
                    data=buffer,
                    file_name="holtwinters_forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:
        st.error(f"Errore durante l'esecuzione del modello Holt-Winters: {e}")
        if return_metrics:
            return {}
    
    # Restituisce le metriche se richiesto (per il modulo Exploratory)
    if return_metrics:
        return metrics_results if 'metrics_results' in locals() else {}
    
    return None  # Default per mantenere compatibilità


def holt_winters_forecast(
    series: pd.Series,
    forecast_periods: int = 12,
    seasonal_periods: int = 12,
    trend: str = 'add',
    seasonal: str = 'add',
    damped_trend: bool = True,
    initialization_method: str = 'estimated',
    smoothing_level: Optional[float] = None,
    smoothing_trend: Optional[float] = None,
    smoothing_seasonal: Optional[float] = None,
    optimized: bool = True
) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Apply the Holt-Winters model to a time series and return fitted values,
    forecasts and model parameters.
    """
    series = series.sort_index()
    series = series.asfreq(pd.infer_freq(series.index))

    model = ExponentialSmoothing(
        series,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method=initialization_method,
    )

    fit = model.fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal,
        optimized=optimized,
    )

    fitted_values = fit.fittedvalues
    forecast_values = fit.forecast(forecast_periods)

    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((series - fitted_values) / series)) * 100
    rmse = np.sqrt(np.mean(np.square(series - fitted_values)))

    model_params = {
        'smoothing_level (α)': fit.params.get('smoothing_level'),
        'smoothing_trend (β)': fit.params.get('smoothing_trend'),
        'smoothing_seasonal (γ)': fit.params.get('smoothing_seasonal'),
        'damping_trend (ϕ)': fit.params.get('damping_trend'),
        'initial_level': fit.params.get('initial_level'),
        'initial_trend': fit.params.get('initial_trend'),
        'initial_seasonal': fit.params.get('initial_seasonal'),
        'aic': fit.aic,
        'bic': fit.bic,
        'mape': mape,
        'rmse': rmse,
    }

    return fitted_values, forecast_values, model_params
