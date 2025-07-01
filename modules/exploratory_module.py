import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_all_metrics(y_true, y_pred):
    valid_indices = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid_indices], y_pred[valid_indices]
    if len(y_true) == 0: return {k: np.nan for k in ["MAE", "MSE", "RMSE", "MAPE", "SMAPE"]}

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
    }
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        metrics["MAPE"] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else: metrics["MAPE"] = np.nan

    denominator = np.abs(y_true) + np.abs(y_pred)
    non_zero_denom_mask = denominator != 0
    if np.any(non_zero_denom_mask):
        metrics["SMAPE"] = np.mean(2 * np.abs(y_pred[non_zero_denom_mask] - y_true[non_zero_denom_mask]) / denominator[non_zero_denom_mask]) * 100
    else: metrics["SMAPE"] = np.nan
    return metrics

def plot_decomposition(decomposition, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Osservato'))
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Stagionalità'))
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='markers', name='Residuo'))
    fig.update_layout(title=title, height=600)
    return fig

def run_exploratory_analysis(df: pd.DataFrame, date_col: str, target_col: str):
    st.subheader("📊 Analisi Esplorativa e Suggerimento Modello")
    st.markdown("Questo modulo analizza la serie storica, la decompone e testa diversi modelli per suggerire il più performante come punto di partenza.")

    series = df.set_index(date_col)[target_col]
    series.index = pd.to_datetime(series.index)
    series = series.asfreq(pd.infer_freq(series.index), fill_value='ffill')

    # --- 1. Analisi Iniziale ---
    st.markdown("### 1. Analisi Descrittiva")
    st.dataframe(series.describe().to_frame().T)

    # --- 2. Decomposizione ---
    st.markdown("### 2. Decomposizione della Serie Storica")
    seasonal_period = st.slider("Seleziona il periodo stagionale per la decomposizione", min_value=2, max_value=365, value=12, key="decomp_period")
    
    try:
        decomp_add = seasonal_decompose(series, model='additive', period=seasonal_period)
        decomp_mul = seasonal_decompose(series, model='multiplicative', period=seasonal_period)
        
        tab1, tab2 = st.tabs(["Decomposizione Additiva", "Decomposizione Moltiplicativa"])
        with tab1:
            st.plotly_chart(plot_decomposition(decomp_add, "Decomposizione Additiva"), use_container_width=True)
        with tab2:
            st.plotly_chart(plot_decomposition(decomp_mul, "Decomposizione Moltiplicativa"), use_container_width=True)
    except Exception as e:
        st.warning(f"Impossibile eseguire la decomposizione con periodo {seasonal_period}: {e}")

    # --- 3. Model Bake-Off ---
    st.markdown("### 3. Gara dei Modelli (Backtest Automatico)")
    st.info("I modelli vengono addestrati sull'85% dei dati e testati sul restante 15% per trovare il più accurato.")

    split_point = int(len(series) * 0.85)
    train, test = series.iloc[:split_point], series.iloc[split_point:]
    horizon = len(test)
    results = {}

    # Modello 1: SARIMA
    try:
        model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, seasonal_period)).fit(disp=False)
        forecast_sarima = model_sarima.get_forecast(steps=horizon).predicted_mean
        results['SARIMA'] = compute_all_metrics(test, forecast_sarima)
    except Exception as e: results['SARIMA'] = {"Errore": str(e)}

    # Modello 2: Holt-Winters
    try:
        model_hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit()
        forecast_hw = model_hw.forecast(steps=horizon)
        results['Holt-Winters'] = compute_all_metrics(test, forecast_hw)
    except Exception as e: results['Holt-Winters'] = {"Errore": str(e)}

    # Modello 3: Prophet
    try:
        model_prophet = Prophet().fit(train.reset_index().rename(columns={date_col: 'ds', target_col: 'y'}))
        future_df = model_prophet.make_future_dataframe(periods=horizon, freq=series.index.freq)
        forecast_prophet = model_prophet.predict(future_df)['yhat'].iloc[-horizon:]
        results['Prophet'] = compute_all_metrics(test, forecast_prophet.values)
    except Exception as e: results['Prophet'] = {"Errore": str(e)}

    # --- 4. Risultati e Raccomandazione ---
    st.markdown("### 4. Risultati e Raccomandazione")
    
    # Controlla se ci sono risultati prima di creare il DataFrame
    if not any(results.values()):
        st.error("Nessun modello è riuscito a produrre metriche di valutazione. Impossibile confrontare i modelli.")
        return

    results_df = pd.DataFrame(results).T.dropna(axis=1, how='all')

    # Se il dataframe è vuoto dopo il dropna, significa che nessun modello ha prodotto metriche
    if results_df.empty:
        st.error("Nessun modello è riuscito a produrre metriche di valutazione valide. Impossibile confrontare i modelli.")
        return

    # Trova la prima metrica disponibile per ordinare
    sort_metric = None
    available_metrics = ["MAPE", "SMAPE", "MAE", "RMSE"]
    for metric in available_metrics:
        if metric in results_df.columns:
            sort_metric = metric
            break
    
    if sort_metric:
        results_df = results_df.sort_values(by=sort_metric)
        st.metric(label="🏆 Best Model", value=results_df.index[0])
    else:
        st.warning("Nessuna metrica standard (MAPE, SMAPE, MAE, RMSE) è stata calcolata. Impossibile suggerire un modello.")


    # Formattazione delle metriche percentuali
    for col in ['MAPE', 'SMAPE']:
        if col in results_df.columns:
            results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)

    st.dataframe(results_df)

    st.header("Confronto Modelli")
    st.write("Vengono confrontati i modelli Prophet, SARIMA e Holt-Winters su un backtest per suggerire il modello migliore.")

    results = {}
    
    # Modello 1: SARIMA
    try:
        model_sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, seasonal_period)).fit(disp=False)
        forecast_sarima = model_sarima.get_forecast(steps=horizon).predicted_mean
        results['SARIMA'] = compute_all_metrics(test, forecast_sarima)
    except Exception as e: results['SARIMA'] = {"Errore": str(e)}

    # Modello 2: Holt-Winters
    try:
        model_hw = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=seasonal_period).fit()
        forecast_hw = model_hw.forecast(steps=horizon)
        results['Holt-Winters'] = compute_all_metrics(test, forecast_hw)
    except Exception as e: results['Holt-Winters'] = {"Errore": str(e)}

    # Modello 3: Prophet
    try:
        model_prophet = Prophet().fit(train.reset_index().rename(columns={date_col: 'ds', target_col: 'y'}))
        future_df = model_prophet.make_future_dataframe(periods=horizon, freq=series.index.freq)
        forecast_prophet = model_prophet.predict(future_df)['yhat'].iloc[-horizon:]
        results['Prophet'] = compute_all_metrics(test, forecast_prophet.values)
    except Exception as e: results['Prophet'] = {"Errore": str(e)}

    if not results:
        st.warning("Nessun modello è riuscito a produrre un risultato valido.")
        return

    results_df = pd.DataFrame(results).T
    
    # Rimuovi colonne che sono interamente NaN
    results_df = results_df.dropna(axis=1, how='all')

    if not results_df.empty:
        # Prova a ordinare per MAPE, altrimenti per MAE, altrimenti non ordinare
        if 'MAPE' in results_df.columns:
            results_df = results_df.sort_values("MAPE")
            best_model = results_df.index[0]
            best_mape = results_df.iloc[0]['MAPE']
            st.success(f"🎉 Il modello suggerito è **{best_model}** con un MAPE del **{best_mape:.2f}%**.")
        elif 'MAE' in results_df.columns:
            results_df = results_df.sort_values("MAE")
            best_model = results_df.index[0]
            st.info(f"Il modello con il MAE più basso è **{best_model}**. MAPE non disponibile per il confronto.")
        else:
            st.warning("Non è stato possibile calcolare metriche valide (es. MAPE, MAE) per ordinare i modelli.")

        # Formattazione delle metriche in percentuale
        for col in ['MAPE', 'SMAPE']:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else x)
        
        st.dataframe(results_df)
    else:
        st.warning("Nessun modello ha prodotto metriche di valutazione valide.")
