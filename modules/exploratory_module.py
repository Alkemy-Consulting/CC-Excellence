import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from typing import Optional

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def run_exploratory_analysis(df):
    st.subheader("📊 Analisi Esplorativa")

    st.markdown("**📌 Dimensioni del dataset:**")
    st.write(f"{df.shape[0]} righe × {df.shape[1]} colonne")

    st.markdown("**🔍 Prime righe del dataset:**")
    st.dataframe(df.head())

    st.markdown("**📈 Statistiche descrittive:**")
    st.dataframe(df.describe())

    st.markdown("**📅 Andamento nel tempo:**")
    if "ds" in df.columns and "y" in df.columns:
        fig = px.line(df, x="ds", y="y", title="Serie temporale")
        st.plotly_chart(fig)
    else:
        st.warning("Il dataset non contiene le colonne 'ds' e 'y' per visualizzare la serie storica.")
        return

    # Prepara la serie
    series = df.set_index("ds")["y"].dropna()
    train = series.iloc[:-6] if len(series) > 12 else series.iloc[:-2]
    test = series.iloc[-6:] if len(series) > 12 else series.iloc[-2:]

    results = {}

    # Holt-Winters
    try:
        hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12, damped_trend=True, initialization_method='estimated')
        hw_fit = hw_model.fit(optimized=True)
        hw_forecast = hw_fit.forecast(len(test))
        hw_mae = np.mean(np.abs(test - hw_forecast))
        hw_rmse = np.sqrt(np.mean((test - hw_forecast) ** 2))
        hw_mape = mean_absolute_percentage_error(test, hw_forecast)
        hw_smape = symmetric_mean_absolute_percentage_error(test, hw_forecast)
        results['Holt-Winters'] = {"MAE": hw_mae, "RMSE": hw_rmse, "MAPE": hw_mape, "SMAPE": hw_smape}
    except Exception as e:
        results['Holt-Winters'] = {"error": str(e)}

    # ARIMA
    try:
        arima_model = ARIMA(train, order=(1,1,1))
        arima_fit = arima_model.fit()
        arima_forecast = arima_fit.forecast(steps=len(test))
        arima_mae = np.mean(np.abs(test - arima_forecast))
        arima_rmse = np.sqrt(np.mean((test - arima_forecast) ** 2))
        arima_mape = mean_absolute_percentage_error(test, arima_forecast)
        arima_smape = symmetric_mean_absolute_percentage_error(test, arima_forecast)
        results['ARIMA'] = {"MAE": arima_mae, "RMSE": arima_rmse, "MAPE": arima_mape, "SMAPE": arima_smape}
    except Exception as e:
        results['ARIMA'] = {"error": str(e)}

    # Prophet (se disponibile)
    try:
        from prophet import Prophet
        prophet_df = train.reset_index().rename(columns={"ds": "ds", "y": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=len(test), freq='D')
        forecast = m.predict(future)
        prophet_pred = forecast['yhat'].iloc[-len(test):].values
        prophet_mae = np.mean(np.abs(test.values - prophet_pred))
        prophet_rmse = np.sqrt(np.mean((test.values - prophet_pred) ** 2))
        prophet_mape = mean_absolute_percentage_error(test.values, prophet_pred)
        prophet_smape = symmetric_mean_absolute_percentage_error(test.values, prophet_pred)
        results['Prophet'] = {"MAE": prophet_mae, "RMSE": prophet_rmse, "MAPE": prophet_mape, "SMAPE": prophet_smape}
    except Exception as e:
        results['Prophet'] = {"error": str(e)}

    st.markdown("**📊 Confronto modelli (backtest):**")
    st.write(pd.DataFrame(results).T)

    # Suggerisci il migliore
    best_model = None
    best_metric = float('inf')
    for model, metrics in results.items():
        if isinstance(metrics, dict) and "MAPE" in metrics:
            if metrics["MAPE"] < best_metric:
                best_metric = metrics["MAPE"]
                best_model = model
    if best_model:
        st.success(f"Il modello suggerito per questa serie è: **{best_model}** (MAPE={best_metric:.2f}%)")
    else:
        st.warning("Nessun modello ha prodotto risultati validi.")
