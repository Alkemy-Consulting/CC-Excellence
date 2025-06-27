import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

def run_holt_winters_model(df: pd.DataFrame, horizon: int = 6, default_seasonal_periods: int = 12):
    with st.form(key="forecast_form", clear_on_submit=False):
    """
    Streamlit-friendly function to run Holt-Winters forecasting with full UI, plotting, and parameter selection.
    Args:
        df: DataFrame with columns ['ds', 'y'] (ds: datetime, y: target)
        horizon: forecast steps
        default_seasonal_periods: default value for seasonal periods
    """
    st.subheader("Parametri Holt-Winters")
    model_type = st.radio(
        "Seleziona il tipo di modello",
        options=[
            "Triple Exponential Default",
            "Triple Exponential Fitted",
            "Straight Line",
            "Quadratic",
            "Cubic"
        ],
        key="model_type"
    )

    seasonal_periods = st.number_input("Periodi stagionali", min_value=2, max_value=24, value=default_seasonal_periods, key="hw_seasonal_periods")
    use_custom = st.checkbox("Utilizza parametri custom", value=False, key="hw_custom")

    if use_custom:
        smoothing_level = st.number_input("Alpha (livello)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="hw_alpha")
        smoothing_trend = st.number_input("Beta (trend)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="hw_beta")
        smoothing_seasonal = st.number_input("Gamma (stagionalità)", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="hw_gamma")
        optimized = False
    else:
        smoothing_level = None
        smoothing_trend = None
        smoothing_seasonal = None
        optimized = True

    # Esegui il modello selezionato
    if model_type == "Triple Exponential Default" or model_type == "Triple Exponential Fitted":
        fitted, forecast, params = holt_winters_forecast(
            df.set_index("ds")["y"],
            forecast_periods=horizon,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add',
            damped_trend=True,
            smoothing_level=smoothing_level,
            smoothing_trend=smoothing_trend,
            smoothing_seasonal=smoothing_seasonal,
            optimized=optimized
        )
    elif model_type == "Straight Line":
        x = np.arange(len(df))
        coeffs = np.polyfit(x, df['y'], 1)
        fitted = pd.Series(np.polyval(coeffs, x), index=df['ds'])
        forecast_index = pd.date_range(df['ds'].iloc[-1], periods=horizon+1, freq=pd.infer_freq(df['ds']))[1:]
        forecast = pd.Series(np.polyval(coeffs, np.arange(len(df), len(df)+horizon)), index=forecast_index)
        params = {'model': 'Straight Line', 'coefficients': coeffs.tolist()}
    elif model_type == "Quadratic":
        x = np.arange(len(df))
        coeffs = np.polyfit(x, df['y'], 2)
        fitted = pd.Series(np.polyval(coeffs, x), index=df['ds'])
        forecast_index = pd.date_range(df['ds'].iloc[-1], periods=horizon+1, freq=pd.infer_freq(df['ds']))[1:]
        forecast = pd.Series(np.polyval(coeffs, np.arange(len(df), len(df)+horizon)), index=forecast_index)
        params = {'model': 'Quadratic', 'coefficients': coeffs.tolist()}
    elif model_type == "Cubic":
        x = np.arange(len(df))
        coeffs = np.polyfit(x, df['y'], 3)
        fitted = pd.Series(np.polyval(coeffs, x), index=df['ds'])
        forecast_index = pd.date_range(df['ds'].iloc[-1], periods=horizon+1, freq=pd.infer_freq(df['ds']))[1:]
        forecast = pd.Series(np.polyval(coeffs, np.arange(len(df), len(df)+horizon)), index=forecast_index)
        params = {'model': 'Cubic', 'coefficients': coeffs.tolist()}

    st.success("Modello di previsione addestrato.")

    st.subheader("🔏 Metriche di errore")
    mae, rmse, mape, df_combined = evaluate_forecast(df.set_index("ds")['y'], forecast)
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

    st.write("**Parametri del modello:**")

    # Mostra metriche di errore
    if 'mape' in params and 'rmse' in params:
        col1, col2 = st.columns(2)
        col1.metric("MAPE", f"{params['mape']:.2f}%")
        col2.metric("RMSE", f"{params['rmse']:.2f}")
    st.json(params)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fitted.index, y=fitted, mode='lines', name='Storico (fitted)'))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast, mode='lines', name='Previsione', line=dict(color='red')))
    fig.update_layout(title="Previsione Modello", xaxis_title="Data", yaxis_title="Valore")
    st.plotly_chart(fig, use_container_width=True)

    # Trigger automatic submission to simulate reactive behavior
    st.form_submit_button("Aggiorna modello", on_click=lambda: None, disabled=True)

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
