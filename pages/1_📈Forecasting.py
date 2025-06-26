import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.title("📈 Contact Center Forecasting Tool")

# Tabs per le sezioni
tabs = st.tabs(["Exploratory", "Prophet", "ARIMA", "Holt-Winters"])

# ===============================
# TAB 1: EXPLORATORY (placeholder)
# ===============================
with tabs[0]:
    st.write("Prossimamente...")

# ===============================
# TAB 2: PROPHET
# ===============================
with tabs[1]:
    st.subheader("🔮 Forecasting con Prophet")

    df = None
    columns = []
    date_col = target_col = freq = fillna_method = None
    detected_freq = "D"
    aggregation_method = "sum"
    yearly_seasonality = weekly_seasonality = daily_seasonality = False
    seasonality_mode = "additive"
    changepoint_prior_scale = 0.05
    periods_input = 30
    use_holidays = False
    launch_forecast = False
    delimiter = ","
    date_format = "%Y-%m-%d"

    # Sidebar completa
    with st.sidebar:
        st.header("1. Data")
        with st.expander("📂 Dataset"):
            delimiter = st.selectbox("Delimitatore CSV", [",", ";", "|", "\t"], index=0)
            date_format = st.text_input("Formato data (es. %Y-%m-%d)", value="%Y-%m-%d")
            file = st.file_uploader("Carica un file CSV", type=["csv"])

        if file:
            df = pd.read_csv(file, delimiter=delimiter)
            columns = df.columns.tolist()

            with st.expander("🧩 Columns"):
                date_col = st.selectbox("Colonna data", options=columns)
                target_col = st.selectbox("Colonna target", options=columns)

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

            st.header("2. Parametri Prophet")
            yearly_seasonality = st.checkbox("Stagionalità annuale", value=True)
            weekly_seasonality = st.checkbox("Stagionalità settimanale", value=True)
            daily_seasonality = st.checkbox("Stagionalità giornaliera", value=False)
            seasonality_mode = st.selectbox("Seasonality mode", ["additive", "multiplicative"])
            changepoint_prior_scale = st.slider("Changepoint prior scale", 0.001, 0.5, 0.05)

            st.header("3. Orizzonte di forecast")
            periods_input = st.number_input("Inserisci il numero di periodi da prevedere", min_value=1, max_value=365, value=30)

            st.header("4. Opzioni avanzate")
            use_holidays = st.checkbox("Includi festività italiane", value=False)

            st.header("5. Esegui")
            launch_forecast = st.button("🚀 Avvia il forecast")

    if df is not None:
        st.write("Anteprima dei dati:", df.head())

        df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        df = df[[date_col, target_col]].dropna()

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

        if launch_forecast:
            if use_holidays:
                years = df['ds'].dt.year.unique()
                holiday_dates = []
                for year in years:
                    holiday_dates.extend([
                        f"{year}-01-01", f"{year}-01-06", f"{year}-04-25",
                        f"{year}-05-01", f"{year}-06-02", f"{year}-08-15",
                        f"{year}-11-01", f"{year}-12-08", f"{year}-12-25",
                        f"{year}-12-26"
                    ])
                holidays = pd.DataFrame({
                    'ds': pd.to_datetime(holiday_dates),
                    'holiday': 'festività_italiane'
                })
                model = Prophet(
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale,
                    holidays=holidays
                )
            else:
                model = Prophet(
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale
                )

            model.fit(df)
            future = model.make_future_dataframe(periods=periods_input, freq=freq)
            forecast = model.predict(future)

            st.subheader("📊 Previsioni")
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            st.subheader("📈 Componenti del modello")
            fig2 = plot_components_plotly(model, forecast)
            st.plotly_chart(fig2)

            st.subheader("📏 Metriche di errore")
            df_forecast = forecast[['ds', 'yhat']].set_index('ds')
            df_actual = df.set_index('ds')
            df_combined = df_actual.join(df_forecast, how='left').dropna()

            mae = mean_absolute_error(df_combined['y'], df_combined['yhat'])
            mse = mean_squared_error(df_combined['y'], df_combined['yhat'])
            rmse = mse ** 0.5
            mape = np.mean(np.abs((df_combined['y'] - df_combined['yhat']) / df_combined['y'])) * 100

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

# ===============================
# TAB 3 e 4 (Placeholder)
# ===============================
with tabs[2]:
    st.write("Prossimamente...")

with tabs[3]:
    st.write("Prossimamente...")
