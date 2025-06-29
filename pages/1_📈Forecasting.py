import streamlit as st
import pandas as pd
import numpy as np

from modules.prophet_module import run_prophet_model
from modules.arima_module import run_arima_model
from modules.holtwinters_module import run_holt_winters_model
from modules.exploratory_module import run_exploratory_analysis

st.title("📈 Contact Center Forecasting Tool")

# Sidebar comune
with st.sidebar:
    st.header("1. Dataset")
    with st.expander("📂 File import"):
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

        with st.expander("🧹 Colonne"):
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

        st.header("2. Modello")
        model_tab = st.selectbox("Seleziona il modello", ["Prophet", "ARIMA", "Holt-Winters", "Exploratory"])

        st.header("3. Backtesting")
        with st.expander("📊 Split"):
            use_cv = st.checkbox("Usa Cross-Validation")
            if use_cv:
                col1, col2 = st.columns(2)
                with col1:
                    cv_start_date = st.date_input("Data inizio CV")
                with col2:
                    cv_end_date = st.date_input("Data fine CV")
                n_folds = st.number_input("Numero di folds", min_value=2, max_value=20, value=5)
                fold_horizon = st.number_input("Orizzonte per fold (in periodi)", min_value=1, value=30)

            if df is not None and not df.empty and not use_cv:
                test_start = df[date_col].iloc[int(len(df) * 0.8)]
                test_end = df[date_col].iloc[-1]
                train_pct = round(len(df[df[date_col] < test_start]) / len(df) * 100, 2)
                st.success(f"Il test set va da **{test_start.date()}** a **{test_end.date()}** – il training usa il {train_pct}% dei dati")

        with st.expander("📏 Metriche"):
            selected_metrics = st.multiselect(
                "Seleziona le metriche di valutazione",
                options=["MAPE", "MAE", "MSE", "RMSE", "SMAPE"],
                default=["MAPE", "MAE", "RMSE"]
            )

        with st.expander("🗓️ Intervalli"):
            st.write("(Periodo o finestra di validazione)")
            aggregate_scope = st.checkbox("Valuta le performance su valori aggregati")

        st.header("4. Forecast")
        with st.expander("📅 Parametri Forecast"):
            make_forecast = st.checkbox("Make forecast on future dates")
            horizon = 30  # Default value for horizon
            if make_forecast:
                horizon = st.number_input("Orizzonte (numero di periodi)", min_value=1, value=30)
                if df is not None and not df.empty:
                    start_date = df[date_col].max() + pd.tseries.frequencies.to_offset(freq)
                    end_date = start_date + pd.tseries.frequencies.to_offset(freq) * (horizon - 1)
                    st.success(f"Il forecast coprirà il periodo da **{start_date.date()}** a **{end_date.date()}**")

        # Bottone per lanciare il forecast
        forecast_button = st.button("🚀 Avvia il forecast")

# Chiamata al modello selezionato (fuori dalla sidebar)
if file and forecast_button:
    st.header("Risultati del Forecast")
    if model_tab == "Prophet":
        run_prophet_model(df, date_col, target_col, freq, horizon, make_forecast)
    elif model_tab == "ARIMA":
        run_arima_model(df, p=1, d=1, q=0, forecast_steps=horizon, target_col=target_col)
    elif model_tab == "Holt-Winters":
        run_holt_winters_model(df, date_col=date_col, target_col=target_col, horizon=horizon, default_seasonal_periods=12)
    elif model_tab == "Exploratory":
        run_exploratory_analysis(df)
