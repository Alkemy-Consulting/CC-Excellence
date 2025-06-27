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
from arima_module import run_arima_model
from holtwinters_module import run_holt_winters_model

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
                test_start = df[date_col].iloc[int(len(df)*0.8)]
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
            if make_forecast:
                horizon = st.number_input("Orizzonte (numero di periodi)", min_value=1, value=30)
                if df is not None and not df.empty:
                    start_date = df[date_col].max() + pd.tseries.frequencies.to_offset(freq)
                    end_date = start_date + pd.tseries.frequencies.to_offset(freq) * (horizon - 1)
                    st.success(f"Il forecast coprirà il periodo da **{start_date.date()}** a **{end_date.date()}**")

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
        st.subheader("Parametri Prophet")
        yearly_seasonality = st.checkbox("Yearly seasonality", value=True, key="prophet_yearly")
        weekly_seasonality = st.checkbox("Weekly seasonality", value=True, key="prophet_weekly")
        daily_seasonality = st.checkbox("Daily seasonality", value=False, key="prophet_daily")
        seasonality_mode = st.selectbox("Seasonality mode", ["additive", "multiplicative"], index=0, key="prophet_mode")
        changepoint_prior_scale = st.number_input("Changepoint prior scale", min_value=0.001, max_value=1.0, value=0.05, step=0.01, key="prophet_changepoint")
        periods_input = st.number_input("Orizzonte forecast (periodi)", min_value=1, value=30, key="prophet_periods")
        use_holidays = st.checkbox("Usa festività italiane", value=False, key="prophet_holidays")
        if st.button("Esegui Prophet"):
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
            st.subheader("📈 Previsioni")
            st.plotly_chart(plot_forecast(model, forecast))
            st.subheader("📁 Componenti del modello")
            st.plotly_chart(plot_components(model, forecast))
            st.subheader("🔏 Metriche di errore")
            mae, rmse, mape, df_combined = evaluate_forecast(df, forecast)
            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**RMSE:** {rmse:.2f}")
            st.write(f"**MAPE:** {mape:.2f}%")
            st.subheader("📅 Esporta i risultati")
            csv_export = forecast.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📆 Scarica il forecast in CSV",
                data=csv_export,
                file_name='forecast_prophet.csv',
                mime='text/csv'
            )

    elif model_tab == "ARIMA":
        st.subheader("Parametri ARIMA")
        p = st.number_input("p (AR)", min_value=0, max_value=10, value=1, key="arima_p")
        d = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1, key="arima_d")
        q = st.number_input("q (MA)", min_value=0, max_value=10, value=0, key="arima_q")
        forecast_steps = st.slider("Passi di previsione", 1, 24, 6, key="arima_steps")
        if st.button("Esegui ARIMA"):
            run_arima_model(df, p=p, d=d, q=q, forecast_steps=forecast_steps, target_col="y")

    elif model_tab == "Holt-Winters":
        # Usa horizon se definito, altrimenti default 6
        forecast_steps = horizon if 'horizon' in locals() else 6
        run_holt_winters_model(df, horizon=forecast_steps, default_seasonal_periods=12)

    elif model_tab == "Exploratory":
        st.subheader("Analisi Esplorativa")
        if st.button("Esegui Analisi Esplorativa"):
            run_exploratory_analysis(df)
