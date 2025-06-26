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

st.title("📈 Contact Center Forecasting Tool")

# Sidebar comune
with st.sidebar:
    st.header("1. Data")
    with st.expander("📂 Dataset"):
        delimiter = st.selectbox("Delimitatore CSV", [",", ";", "|", "\t"], index=0)
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
        model_tab = st.selectbox("Seleziona il modello", ["Prophet", "ARIMA", "Holt-Winters"])
        launch_forecast = st.button("🚀 Avvia il forecast")
