import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
from modules.metrics_module import compute_metrics, compute_all_metrics


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

    metrics = compute_all_metrics(df_combined['y'], df_combined['yhat'])
    metrics['combined'] = df_combined
    return metrics

def plot_forecast(model, forecast):
    return plot_plotly(model, forecast)

def plot_components(model, forecast):
    return plot_components_plotly(model, forecast)

def run_prophet_model(df, date_col, target_col, freq, horizon, make_forecast, use_cv=False, cv_start_date=None, cv_end_date=None, n_folds=5, fold_horizon=30, test_start_date=None, test_end_date=None, params=None, selected_metrics=None, return_metrics=False):
    import prophet
    # st.info(f"Versione di Prophet in uso: {prophet.__version__}")  # <-- RIMOSSO QUESTO MESSAGGIO
    if not return_metrics:
        st.subheader("Prophet Forecast")

    # Preprocessing
    prophet_df = df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # Imposta i parametri del modello, con valori di default
    if params is None:
        params = {}
    seasonality_mode = params.get('seasonality_mode', 'additive')
    changepoint_prior_scale = params.get('changepoint_prior_scale', 0.05)
    holidays_country = params.get('holidays_country')

    holidays = None
    if holidays_country:
        try:
            # Questa è una semplificazione, per un'implementazione robusta
            # si dovrebbe usare una libreria come `holidays`
            from prophet.make_holidays import make_holidays_df
            years = prophet_df['ds'].dt.year.unique()
            holidays = make_holidays_df(year_list=years, country=holidays_country)
        except Exception as e:
            st.warning(f"Non è stato possibile caricare le festività per {holidays_country}: {e}")

    if use_cv:
        st.markdown("### Cross-Validation Prophet")

        try:
            total_days = (cv_end_date - cv_start_date).days
            initial_days = total_days - (n_folds - 1) * fold_horizon

            if initial_days <= 0:
                st.error("Intervallo troppo corto per il numero di fold selezionato.")
                return

            df_cv_range = prophet_df[
                (prophet_df['ds'] >= pd.to_datetime(cv_start_date)) & (prophet_df['ds'] <= pd.to_datetime(cv_end_date))
            ].copy()

            if not return_metrics:
                st.info(f"Dati usati per CV: {df_cv_range['ds'].min().date()} → {df_cv_range['ds'].max().date()}")

            model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                holidays=holidays
            )

            # Fallback automatico per la cross-validation
            try:
                if not return_metrics:
                    st.write("Esecuzione cross-validation con sintassi moderna (Prophet >= 1.1)...")
                df_cv = cross_validation(
                    model,
                    df=df_cv_range,
                    initial=f'{initial_days} days',
                    period=f'{fold_horizon} days',
                    horizon=f'{fold_horizon} days'
                )
            except TypeError as e:
                if "unexpected keyword argument 'df'" in str(e):
                    if not return_metrics:
                        st.warning("Sintassi moderna fallita. Tentativo con la sintassi legacy (Prophet < 1.1)...")
                    model.fit(df_cv_range)
                    df_cv = cross_validation(
                        model,
                        initial=f'{initial_days} days',
                        period=f'{fold_horizon} days',
                        horizon=f'{fold_horizon} days'
                    )
                else:
                    # Se l'errore è diverso, lo sollevo comunque
                    raise e
            
            df_perf = performance_metrics(df_cv)
            if not return_metrics:
                st.dataframe(df_perf[['horizon', 'mae', 'rmse', 'mape']].round(2))

                # Sintesi delle metriche
                st.write("### CV Metrics (media su tutti i fold)")
            
            # Mostra solo le metriche selezionate
            avg_metrics = {
                'MAE': df_perf['mae'].mean(),
                'RMSE': df_perf['rmse'].mean(),
                'MAPE': df_perf['mape'].mean(),
                'MSE': df_perf['mse'].mean(),
                'SMAPE': df_perf['smape'].mean() if 'smape' in df_perf else np.nan
            }

            if not return_metrics:
                cols = st.columns(len(selected_metrics))
                for i, metric in enumerate(selected_metrics):
                    metric_key = metric.upper()
                    value = avg_metrics.get(metric_key, np.nan)
                    if metric_key in ["MAPE", "SMAPE"]:
                        cols[i].metric(metric, f"{value:.0f}%")
                    else:
                        cols[i].metric(metric, f"{value:.2f}")

                # Plot del forecast dell'ultimo fold come esempio
                st.write("### Grafico dell'ultimo fold di Cross-Validation")
                last_cutoff = df_cv['cutoff'].max()
                df_cv_last_fold = df_cv[df_cv['cutoff'] == last_cutoff]

                import plotly.graph_objects as go
                fig = go.Figure([
                    go.Scatter(x=df_cv_last_fold['ds'], y=df_cv_last_fold['y'], name='Actual', mode='lines', line=dict(color='#1f77b4')),
                    go.Scatter(x=df_cv_last_fold['ds'], y=df_cv_last_fold['yhat'], name='Forecast', mode='lines', line=dict(color='#ff7f0e', dash='dash')),
                    go.Scatter(x=df_cv_last_fold['ds'], y=df_cv_last_fold['yhat_lower'], fill='tonexty', mode='none', fillcolor='rgba(255,127,14,0.2)', showlegend=False),
                    go.Scatter(x=df_cv_last_fold['ds'], y=df_cv_last_fold['yhat_upper'], fill='tonexty', mode='none', fillcolor='rgba(255,127,14,0.2)', showlegend=False)
                ])
                fig.update_layout(title=f"CV Forecast vs Actuals (Cutoff: {last_cutoff.date()})")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Errore durante la cross-validation: {e}")

    else:
        if not make_forecast:
            if not return_metrics:
                st.markdown("### Backtesting personalizzato")

            if test_start_date and test_end_date:
                test_mask = (prophet_df['ds'] >= pd.to_datetime(test_start_date)) & (prophet_df['ds'] <= pd.to_datetime(test_end_date))
                test_df = prophet_df[test_mask]
                train_df = prophet_df[prophet_df['ds'] < pd.to_datetime(test_start_date)]
            else:
                split_point = int(len(prophet_df) * 0.8)
                train_df = prophet_df.iloc[:split_point]
                test_df = prophet_df.iloc[split_point:]

            if not return_metrics:
                st.info(f"Training: {train_df['ds'].min().date()} → {train_df['ds'].max().date()}\nTest: {test_df['ds'].min().date()} → {test_df['ds'].max().date()}")

            model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                holidays=holidays
            )
            model.fit(train_df)

            future = model.make_future_dataframe(periods=len(test_df), freq=freq)
            forecast = model.predict(future)
            forecast = forecast[forecast['ds'].isin(test_df['ds'])]

            results = evaluate_forecast(test_df, forecast)
            if not return_metrics:
                st.plotly_chart(plot_plotly(model, forecast), use_container_width=True)
                st.plotly_chart(plot_components_plotly(model, forecast), use_container_width=True)
            
                st.write("### Evaluation Metrics")
                cols = st.columns(len(selected_metrics))
                for i, metric in enumerate(selected_metrics):
                    if metric in results:
                        cols[i].metric(metric, f"{results[metric]:.3f}")

                if st.button("📥 Scarica Forecast in Excel", key="prophet_download_btn_1"):
                    import io
                    forecast_out = forecast.copy()
                    forecast_out = forecast_out[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    buffer = io.BytesIO()
                    forecast_out.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    st.download_button(
                    label="Download .xlsx",
                    data=buffer,
                    file_name="forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        else:
            model = Prophet(
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                holidays=holidays
            )
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=horizon, freq=freq)
            forecast = model.predict(future)

            if not return_metrics:
                st.markdown("### Forecast futuro")

            import plotly.graph_objects as go

            fig = go.Figure([
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    fill=None,
                    showlegend=False
                ),
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,127,14,0.2)',
                    showlegend=False
                ),
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    name='Forecast',
                    mode='lines',
                    line=dict(shape='spline', color='#ff7f0e')
                ),
                go.Scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    name='Storico',
                    mode='markers',
                    marker=dict(color='lightgrey', size=6)
                )
            ])

            if not return_metrics:
                st.plotly_chart(fig, use_container_width=True)

                st.plotly_chart(plot_components(model, forecast), use_container_width=True)
            metrics = evaluate_forecast(prophet_df, forecast)
            
            if not return_metrics:
                st.write("### Evaluation Metrics (su dati storici)")
                cols = st.columns(len(selected_metrics))
                for i, metric in enumerate(selected_metrics):
                    if metric in metrics:
                        cols[i].metric(metric, f"{metrics[metric]:.3f}")

                if st.button("📥 Scarica Forecast in Excel", key="prophet_download_btn_2"):
                    import io
                    forecast_out = forecast.copy()
                    forecast_out = forecast_out[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                    buffer = io.BytesIO()
                    forecast_out.to_excel(buffer, index=False, engine='openpyxl')
                    buffer.seek(0)
                    st.download_button(
                    label="Download .xlsx",
                    data=buffer,
                    file_name="forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # Restituisce le metriche se richiesto (per il modulo Exploratory)
    if return_metrics:
        if use_cv:
            # Restituisce le metriche medie della cross-validation
            return avg_metrics if 'avg_metrics' in locals() else {}
        elif not make_forecast:
            # Restituisce le metriche del test set (backtesting)
            return results if 'results' in locals() else {}
        else:
            # Restituisce le metriche del forecast standard o futuro
            return metrics if 'metrics' in locals() else {}
    
    return None  # Default per mantenere compatibilità
