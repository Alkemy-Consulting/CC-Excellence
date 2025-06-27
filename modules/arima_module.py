import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def run_arima_model(df, p=1, d=1, q=0, forecast_steps=6, target_col='y'):
    """
    Esegue ARIMA sul DataFrame fornito e mostra risultati in Streamlit.
    Args:
        df (pd.DataFrame): DataFrame con almeno la colonna target_col.
        p, d, q (int): Parametri ARIMA.
        forecast_steps (int): Numero di passi di previsione.
        target_col (str): Nome della colonna target.
    """
    st.header("ARIMA Forecasting")
    st.write(f"Colonna target: **{target_col}** | Parametri: (p={p}, d={d}, q={q}) | Previsione: {forecast_steps} passi")
    st.write("Anteprima dati:")
    st.dataframe(df.head())

    try:
        # Se esiste la colonna 'ds', usala come indice datetime
        if 'ds' in df.columns:
            df = df.copy()
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.set_index('ds')
        series = df[target_col]

        if isinstance(df.index, pd.DatetimeIndex):
            last_date = df.index[-1]
            freq = pd.infer_freq(df.index)
            if freq is None:
                freq = 'D'  # fallback
            forecast_index = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=forecast_steps, freq=freq)
        else:
            forecast_index = range(len(series), len(series)+forecast_steps)

        model = ARIMA(series, order=(p, d, q))
        model_fit = model.fit()
        st.success("Modello ARIMA addestrato.")
        st.write(model_fit.summary())

        forecast = model_fit.forecast(steps=forecast_steps)
        forecast = pd.Series(forecast.values, index=forecast_index)

        fig, ax = plt.subplots()
        series.plot(ax=ax, label="Storico")
        forecast.plot(ax=ax, label="Previsione", color="red")
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Errore nell'esecuzione del modello: {e}")
