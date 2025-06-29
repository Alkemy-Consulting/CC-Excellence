import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go

def run_holt_winters_model(df: pd.DataFrame, horizon: int = 6, default_seasonal_periods: int = 12):
    """
    Streamlit-friendly function to run Holt-Winters forecasting with full UI, plotting, and parameter selection.
    Args:
        df: DataFrame with columns ['ds', 'y'] (ds: datetime, y: target)
        horizon: forecast steps
        default_seasonal_periods: default value for seasonal periods
    """
    st.subheader("Parametri Holt-Winters")
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

    # Esecuzione automatica del modello senza bottone
    fitted, forecast, params = holt_winters_forecast(
        df.set_index("ds")['y'],
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

    st.success("Modello Holt-Winters addestrato.")
    st.write("**Parametri del modello:**")
    st.json(params)

    # Calcolo metriche di errore selezionate
    y_true = df.set_index("ds")['y']
    y_pred = fitted
    metrics = st.session_state.get('selected_metrics', ["MAE", "RMSE", "MAPE"])
    st.subheader("🔏 Metriche di errore")
    if "MAE" in metrics:
        mae = np.mean(np.abs(y_true - y_pred))
        st.write(f"**MAE:** {mae:.2f}")
    if "MSE" in metrics:
        mse = np.mean((y_true - y_pred) ** 2)
        st.write(f"**MSE:** {mse:.2f}")
    if "RMSE" in metrics:
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        st.write(f"**RMSE:** {rmse:.2f}")
    if "MAPE" in metrics:
        mape = np.mean(np.abs((y_true - y_pred) / y_true.replace(0, np.nan))) * 100
        st.write(f"**MAPE:** {mape:.2f}%")
    if "SMAPE" in metrics:
        smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
        st.write(f"**SMAPE:** {smape:.2f}%")

    fig, ax = plt.subplots()
    fitted.plot(ax=ax, label="Storico (fitted)")
    forecast.plot(ax=ax, label="Previsione", color="red")
    ax.legend()
    st.pyplot(fig)

    # Modelli alternativi (Straight Line, Quadratic, Cubic) sono lasciati come esempio/commento.

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
