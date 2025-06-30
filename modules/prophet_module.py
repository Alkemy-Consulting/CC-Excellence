import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io

def build_and_forecast_prophet(df, freq='D', periods=30, use_holidays=False, yearly=True, weekly=False, daily=False, seasonality_mode='additive', changepoint_prior_scale=0.05):
    holidays = None
    if use_holidays:
        years = df['ds'].dt.year.unique()
        holiday_dates = [f"{year}-12-25" for year in years]  # Example: Christmas
        holidays = pd.DataFrame({'ds': pd.to_datetime(holiday_dates), 'holiday': 'holiday'})

    model = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        holidays=holidays
    )

    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return model, forecast

def evaluate_forecast(df, forecast):
    df = df.copy()
    forecast = forecast.copy()

    df['ds'] = pd.to_datetime(df['ds'])
    forecast['ds'] = pd.to_datetime(forecast['ds'])

    eval_df = df[df['ds'].isin(forecast['ds'])].set_index('ds')
    pred_df = forecast.set_index('ds').loc[eval_df.index]

    df_combined = eval_df.join(pred_df[['yhat']], how='inner')

    mae = mean_absolute_error(df_combined['y'], df_combined['yhat'])
    rmse = mean_squared_error(df_combined['y'], df_combined['yhat']) ** 0.5
    mape = np.mean(np.abs((df_combined['y'] - df_combined['yhat']) / df_combined['y'])) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'combined': df_combined
    }

def plot_forecast(model, forecast):
    return plot_plotly(model, forecast)

def plot_components(model, forecast):
    return plot_components_plotly(model, forecast)

def run_prophet_model(df, date_col, target_col, freq, horizon, make_forecast, use_cv=False, cv_start_date=None, cv_end_date=None, n_folds=5, fold_horizon=30, test_start_date=None, test_end_date=None):
    st.subheader("Prophet Forecast")

    prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    def download_forecast_excel(forecast_df):
        forecast_out = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        csv = forecast_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Scarica Forecast in CSV",
            data=csv,
            file_name="forecast.csv",
            mime="text/csv"
        )

    if use_cv:
        st.markdown("### Cross-Validation Prophet")
        try:
            total_days = (cv_end_date - cv_start_date).days
            initial_days = total_days - (n_folds - 1) * fold_horizon

            if initial_days <= 0:
                st.error("Intervallo troppo corto per il numero di fold selezionato.")
                return

            df_cv_range = prophet_df[
                (prophet_df['ds'] >= cv_start_date) & (prophet_df['ds'] <= cv_end_date)
            ].copy()

            st.info(f"Dati usati per CV: {df_cv_range['ds'].min().date()} → {df_cv_range['ds'].max().date()}")

            model = Prophet()
            model.fit(df_cv_range)

            df_cv = cross_validation(
                model,
                initial=f"{initial_days} days",
                period=f"{fold_horizon} days",
                horizon=f"{fold_horizon} days"
            )
            df_perf = performance_metrics(df_cv)
            st.dataframe(df_perf[['horizon', 'mae', 'rmse', 'mape']].round(2))

            st.write("### CV Metrics (media su tutti i fold)")
            st.write({
                'MAE': df_perf['mae'].mean(),
                'RMSE': df_perf['rmse'].mean(),
                'MAPE': df_perf['mape'].mean()
            })

        except Exception as e:
            st.error(f"Errore durante la cross-validation: {e}")

    else:
        if not make_forecast:
            st.markdown("### Backtesting personalizzato")

            if test_start_date and test_end_date:
                test_mask = (prophet_df['ds'] >= test_start_date) & (prophet_df['ds'] <= test_end_date)
                test_df = prophet_df[test_mask]
                train_df = prophet_df[prophet_df['ds'] < test_start_date]
            else:
                split_point = int(len(prophet_df) * 0.8)
                train_df = prophet_df.iloc[:split_point]
                test_df = prophet_df.iloc[split_point:]

            st.info(f"Training: {train_df['ds'].min().date()} → {train_df['ds'].max().date()}\nTest: {test_df['ds'].min().date()} → {test_df['ds'].max().date()}")

            model = Prophet()
            model.fit(train_df)

            future = model.make_future_dataframe(periods=len(test_df), freq=freq)
            forecast = model.predict(future)
            forecast = forecast[forecast['ds'].isin(test_df['ds'])]

            results = evaluate_forecast(test_df, forecast)
            st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)
            st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)
            st.write("### Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{results['MAE']:.2f}")
            col2.metric("RMSE", f"{results['RMSE']:.2f}")
            col3.metric("MAPE", f"{results['MAPE']:.2f}%")

            download_forecast_excel(forecast)

        else:
            model, forecast = build_and_forecast_prophet(
                prophet_df,
                freq=freq,
                periods=horizon
            )
            st.markdown("### Forecast futuro")
            st.plotly_chart(plot_forecast(model, forecast), use_container_width=True)
            st.plotly_chart(plot_components(model, forecast), use_container_width=True)
            metrics = evaluate_forecast(prophet_df, forecast)
            st.write("### Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"{metrics['MAE']:.2f}")
            col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")

            download_forecast_excel(forecast)
