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

    # Sidebar completa
    with st.sidebar:
        st.header("1. Carica i dati")
        file = st.file_uploader("Carica un file CSV", type=["csv"])

        if file:
            df = pd.read_csv(file)
            columns = df.columns.tolist()

            st.header("2. Preparazione dati")
            date_col = st.selectbox("Colonna data", options=columns)
            target_col = st.selectbox("Colonna target", options=columns)

            st.header("3. Parametri Prophet")
            yearly_seasonality = st.checkbox("Stagionalità annuale", value=True)
            weekly_seasonality = st.checkbox("Stagionalità settimanale", value=True)
            daily_seasonality = st.checkbox("Stagionalità giornaliera", value=False)
            seasonality_mode = st.selectbox("Seasonality mode", ["additive", "multiplicative"])
            changepoint_prior_scale = st.slider("Changepoint prior scale", 0.001, 0.5, 0.05)

            st.header("4. Orizzonte di forecast")
            periods_input = st.number_input("Inserisci il numero di giorni di forecast", min_value=1, max_value=365, value=30)

            st.header("5. Esegui")
            launch_forecast = st.button("🚀 Avvia il forecast")

    if file:
        df = pd.read_csv(file)
        st.write("Anteprima dei dati:", df.head())

        df[date_col] = pd.to_datetime(df[date_col])
        df = df[[date_col, target_col]].dropna()
        df = df.rename(columns={date_col: "ds", target_col: "y"})

        if launch_forecast:
            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=periods_input)
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
