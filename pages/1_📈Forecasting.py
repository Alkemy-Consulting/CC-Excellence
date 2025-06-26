import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import datetime

st.title("📈 Contact Center Forecasting Tool")

# Tabs
tabs = st.tabs(["Exploratory", "Prophet", "ARIMA", "Holt-Winters"])

with tabs[0]:
    st.markdown("""
    Questo tool consente di:
    1. Caricare un file CSV contenente dati storici (es. volume chiamate).
    2. Analizzare e selezionare la miglior granularità.
    3. Eseguire modelli di forecasting multipli (Prophet, ARIMA, Holt-Winters).
    4. Valutare gli errori e confrontare i modelli.
    5. Proiettare il miglior forecast su grafico.
    6. Scaricare il forecast in CSV.
    """)

    # 1. UPLOAD FILE
    file = st.file_uploader("Carica il file CSV", type="csv")
    holidays_file = st.file_uploader("Carica file festività opzionale (colonna 'ds')", type="csv")

    if file:
        df = pd.read_csv(file, parse_dates=True)
        st.write("Anteprima dei dati:", df.head())

        # 2. IDENTIFICAZIONE DATA E TARGET
        all_cols = df.columns.tolist()
        date_col = st.selectbox("Seleziona la colonna temporale", all_cols)
        value_col = st.selectbox("Seleziona la colonna di volume (target)", all_cols)

        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df = df[[date_col, value_col]].dropna()
        df = df.sort_values(by=date_col)
        df = df.rename(columns={date_col: "ds", value_col: "y"})

        # 3. RILEVAMENTO GRANULARITÀ
        df['diff'] = df['ds'].diff().dt.total_seconds()
        granularity_sec = df['diff'].mode()[0]
        if granularity_sec <= 3600:
            granularity = 'H'
        elif granularity_sec <= 86400:
            granularity = 'D'
        elif granularity_sec <= 604800:
            granularity = 'W'
        else:
            granularity = 'M'
        df = df.drop(columns=['diff'])
        st.success(f"Granularità rilevata: {granularity}")

        # 4. VISUALIZZAZIONE DATI
        st.subheader("Visualizzazione della serie storica")
        fig = px.line(df, x='ds', y='y', title='Serie Temporale')
        st.plotly_chart(fig)

        # 5. PREVISIONI MULTIPLE
        st.subheader("Forecasting con modelli multipli")
        horizon = st.slider("Seleziona orizzonte di forecast (periodi)", 7, 90, 30)

        results = {}
        future_dates = pd.date_range(start=df['ds'].max(), periods=horizon + 1, freq=granularity)[1:]

        # Prophet
        try:
            if holidays_file:
                holidays = pd.read_csv(holidays_file)
                holidays['ds'] = pd.to_datetime(holidays['ds'], dayfirst=True)
            else:
                # principali festività italiane
                years = df['ds'].dt.year.unique()
                holiday_dates = []
                for year in years:
                    holiday_dates.extend([
                        f"{year}-01-01",  # Capodanno
                        f"{year}-01-06",  # Epifania
                        f"{year}-04-25",  # Festa della Liberazione
                        f"{year}-05-01",  # Festa del Lavoro
                        f"{year}-06-02",  # Festa della Repubblica
                        f"{year}-08-15",  # Ferragosto
                        f"{year}-11-01",  # Ognissanti
                        f"{year}-12-08",  # Immacolata Concezione
                        f"{year}-12-25",  # Natale
                        f"{year}-12-26",  # Santo Stefano
                    ])
                holidays = pd.DataFrame({'ds': pd.to_datetime(holiday_dates), 'holiday': 'festività_italiane'})

            model_prophet = Prophet(yearly_seasonality=True, holidays=holidays)
            model_prophet.fit(df)
            future = model_prophet.make_future_dataframe(periods=horizon, freq=granularity)
            forecast_prophet = model_prophet.predict(future)
            prophet_pred = forecast_prophet.set_index('ds').loc[future_dates]['yhat']
            results['Prophet'] = prophet_pred.values
        except:
            results['Prophet'] = [np.nan] * horizon

        # Holt-Winters
        try:
            model_hw = ExponentialSmoothing(df['y'], trend='add', seasonal='add', seasonal_periods=7).fit()
            hw_forecast = model_hw.forecast(horizon)
            results['Holt-Winters'] = hw_forecast.values
        except:
            results['Holt-Winters'] = [np.nan] * horizon

        # ARIMA
        try:
            model_arima = ARIMA(df['y'], order=(1,1,1)).fit()
            arima_forecast = model_arima.forecast(steps=horizon)
            results['ARIMA'] = arima_forecast.values
        except:
            results['ARIMA'] = [np.nan] * horizon

        # 6. VALUTAZIONE ERRORI
        st.subheader("Valutazione degli errori")
        cutoff = len(df) - horizon
        train = df.iloc[:cutoff]
        test = df.iloc[cutoff:]
        y_true = test['y'].values

        error_table = []
        for model, preds in results.items():
            if len(preds) == len(y_true):
                mae = mean_absolute_error(y_true, preds)
                rmse = mean_squared_error(y_true, preds) ** 0.5
                mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
                error_table.append({"Modello": model, "MAE": mae, "RMSE": rmse, "MAPE": mape})

        df_errors = pd.DataFrame(error_table).sort_values(by="MAPE")
        st.dataframe(df_errors)

        # 7. MIGLIOR MODELLO E GRAFICO FORECAST
        if not df_errors.empty:
            best_model = df_errors.iloc[0]['Modello']
            st.success(f"Il modello più accurato è: {best_model}")

            future_df = pd.DataFrame({
                "ds": future_dates,
                "Forecast": results[best_model]
            })

            df_plot = pd.concat([df, future_df], axis=0)
            fig3 = px.line(df_plot, x="ds", y="y" if 'y' in df_plot else "Forecast", title="Forecast finale")
            st.plotly_chart(fig3)

            # Download CSV
            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Scarica il forecast in CSV",
                data=csv,
                file_name='forecast_output.csv',
                mime='text/csv'
            )
