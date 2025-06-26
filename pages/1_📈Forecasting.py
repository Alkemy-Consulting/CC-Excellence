import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Aggiungi il path al modulo personalizzato
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))

from prophet_module import (
    build_prophet_model,
    evaluate_forecast,
    plot_forecast,
    plot_components
)
from exploratory_module import run_exploratory_analysis

st.title("📈 Contact Center Forecasting Tool")

# Sidebar comune
with st.sidebar:
    st.header("1. Data")
    with st.expander("📂 Dataset"):
        delimiter = st.selectbox("Delimitatore CSV", [",", ";", "|", "	"], index=0)
        user_friendly_format = st.selectbox("Formato data", [
            "gg/mm/aaaa", "gg/mm/aa", "aaaa-mm-gg",
            "mm/gg/aaaa", "gg.mm.aaaa", "aaaa/mm/gg"
        ], index=1)
        format_map = {
            "gg/mm/aaaa": "%d/%m/%Y",
            "gg/mm/aa": "%d/%m/%y",
            "aaaa-mm-gg": "%Y-%m-%d",
            "mm/gg/aaaa": "%m/%d/%Y",
            "gg.mm.aaaa": "%d.%m.%Y",
            "aaaa/mm/gg": "%Y/%m/%d"
        }
        date_format = format_map[user_friendly_format]
        file = st.file_uploader("Carica un file CSV", type=["csv"])

    df, date_col, target_col, freq, aggregation_method = None, None, None, "D", "sum"
    clip_negatives = replace_outliers = clean_zeros = False

    if file:
        df = pd.read_csv(file, delimiter=delimiter)
        columns = df.columns.tolist()

        with st.expander("🧩 Columns"):
            date_col = st.selectbox("Colonna data", options=columns)
            target_col = st.selectbox("Colonna target", options=columns, index=1 if len(columns) > 1 else 0)

        with st.expander("⏱️ Granularità"):
            try:
                df[date_col] = pd.to_datetime(df[date_col], format=date_format)
                df_sorted = df.sort_values(by=date_col)
                inferred = pd.infer_freq(df_sorted[date_col])
                detected_freq = inferred if inferred else "D"
            except:
                detected_freq = "D"
            st.text(f"Granularità rilevata: {detected_freq}")
            freq = st.selectbox("Seleziona una nuova granularità", ["D", "W", "M"], index=["D", "W", "M"].index(detected_freq) if detected_freq in ["D", "W", "M"] else 0)
            aggregation_method = st.selectbox("Metodo di aggregazione", ["sum", "mean", "max", "min"])

        with st.expander("🧹 Data Cleaning"):
            clean_zeros = st.checkbox("Rimuovi righe con zero nel target", value=True)
            replace_outliers = st.checkbox("Sostituisci outlier (z-score > 3) con mediana", value=True)
            clip_negatives = st.checkbox("Trasforma valori negativi in zero", value=True)

        st.header("2. Parametri Forecast")
        model_tab = st.selectbox("Seleziona il modello", ["Prophet", "ARIMA", "Holt-Winters", "Exploratory"])
        launch_forecast = st.button("🚀 Avvia il forecast")

# Esecuzione modello e visualizzazione risultati
if file and launch_forecast:
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)
    df = df[[date_col, target_col]].dropna()

    if clean_zeros:
        df = df[df[target_col] != 0]
    if clip_negatives:
        df[target_col] = df[target_col].clip(lower=0)
    if replace_outliers:
        y = df[target_col]
        z = (y - y.mean()) / y.std()
        df.loc[z.abs() > 3, target_col] = y.median()

    if aggregation_method == "sum":
        df = df.groupby(date_col).sum().reset_index()
    elif aggregation_method == "mean":
        df = df.groupby(date_col).mean().reset_index()
    elif aggregation_method == "max":
        df = df.groupby(date_col).max().reset_index()
    elif aggregation_method == "min":
        df = df.groupby(date_col).min().reset_index()

    df = df.rename(columns={date_col: "ds", target_col: "y"})
    df = df.set_index("ds").asfreq(freq).reset_index()

    st.subheader(f"🔎 Risultati del modello: {model_tab}")

    if model_tab == "Prophet":
        # Parametri fittizi (da espandere con la sezione Modelling)
        yearly_seasonality = True
        weekly_seasonality = True
        daily_seasonality = False
        seasonality_mode = "additive"
        changepoint_prior_scale = 0.05
        periods_input = 30
        use_holidays = False

        model, forecast = build_prophet_model(
            df=df,
            freq=freq,
            periods_input=periods_input,
            use_holidays=use_holidays,
            yearly=yearly_seasonality,
            weekly=weekly_seasonality,
            daily=daily_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale
        )

        st.subheader("📊 Previsioni")
        st.plotly_chart(plot_forecast(model, forecast))

        st.subheader("📈 Componenti del modello")
        st.plotly_chart(plot_components(model, forecast))

        st.subheader("📏 Metriche di errore")
        mae, rmse, mape, df_combined = evaluate_forecast(df, forecast)
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAPE:** {mape:.2f}%")

        st.subheader("📁 Esporta i risultati")
        csv_export = forecast.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📅 Scarica il forecast in CSV",
            data=csv_export,
            file_name='forecast_prophet.csv',
            mime='text/csv'
        )

    elif model_tab == "ARIMA":
        st.info("🔧 Modulo ARIMA in sviluppo. Presto disponibile.")

    elif model_tab == "Holt-Winters":
        st.info("🔧 Modulo Holt-Winters in sviluppo. Presto disponibile.")

    elif model_tab == "Exploratory":
        run_exploratory_analysis(df)
