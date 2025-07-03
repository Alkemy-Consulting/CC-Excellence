import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import copy # Importato per deepcopy
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    fig.update_layout(title=title, height=600, showlegend=True)
    return fig

def plot_acf_pacf(series):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(series, ax=axes[0], lags=40)
    plot_pacf(series, ax=axes[1], lags=40)
    fig.tight_layout()
    return fig

def run_exploratory_analysis(df, date_col, target_col, freq, selected_metrics,
                             prophet_params, holtwinters_params, arima_params, sarima_params,
                             **kwargs):
    """
    Esegue un'analisi esplorativa confrontando diversi modelli di forecasting.
    """
    st.header("📊 Analisi Esplorativa dei Modelli")

    # Estrai gli argomenti comuni da kwargs
    horizon = kwargs.get('horizon', 30)
    make_forecast = kwargs.get('make_forecast', True)
    use_cv = kwargs.get('use_cv', False)
    test_start_date = kwargs.get('test_start_date')
    test_end_date = kwargs.get('test_end_date')

    # --- 1. Analisi Statistica Preliminare ---
    st.subheader("1. Analisi Statistica della Serie Storica")
    
    series = df.set_index(date_col)[target_col]
    series.index = pd.to_datetime(series.index)

    # Split train/test per il grafico di confronto finale
    if test_start_date:
        train_series = series[series.index < pd.to_datetime(test_start_date)]
        test_series = series[series.index >= pd.to_datetime(test_start_date)]
        if test_end_date:
            test_series = test_series[test_series.index <= pd.to_datetime(test_end_date)]
    else:
        # Fallback se le date non sono definite (anche se non dovrebbe succedere)
        train_series = series
        test_series = pd.Series()


    with st.expander("Visualizza Analisi Dettagliata", expanded=False):
        
        st.markdown("#### Statistiche Descrittive")
        st.dataframe(series.describe().to_frame().T)

        # Grafico della serie temporale con trend
        st.markdown("#### Visualizzazione Serie Temporale")
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=series.index, 
            y=series, 
            mode='lines', 
            name='Serie Storica',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Aggiungi media mobile per visualizzare il trend
        window = min(max(len(series) // 12, 7), 30)  # Finestra adattiva
        if len(series) > window:
            rolling_mean = series.rolling(window=window, center=True).mean()
            fig_ts.add_trace(go.Scatter(
                x=rolling_mean.index,
                y=rolling_mean,
                mode='lines',
                name=f'Media Mobile ({window} periodi)',
                line=dict(color='#ff7f0e', width=3, dash='dash')
            ))
        
        fig_ts.update_layout(
            title="Serie Storica con Trend",
            xaxis_title="Data",
            yaxis_title="Valore",
            height=400
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Analisi della distribuzione
        st.markdown("#### Analisi della Distribuzione")
        col1, col2 = st.columns(2)
        
        with col1:
            # Istogramma
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=series.dropna(),
                nbinsx=30,
                name='Distribuzione',
                marker_color='lightblue',
                opacity=0.7
            ))
            fig_hist.update_layout(
                title="Distribuzione dei Valori",
                xaxis_title="Valore",
                yaxis_title="Frequenza",
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot per identificare outlier
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=series.dropna(),
                name='Box Plot',
                marker_color='lightgreen'
            ))
            fig_box.update_layout(
                title="Box Plot - Identificazione Outlier",
                yaxis_title="Valore",
                height=300
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # Analisi stagionalità
        st.markdown("#### Analisi della Stagionalità")
        try:
            # Grafico stagionalità per mese (se abbiamo dati sufficienti)
            if len(series) > 24:  # Almeno 2 anni di dati
                series_df = series.to_frame(name='value').reset_index()
                series_df['month'] = series_df[series_df.columns[0]].dt.month
                series_df['year'] = series_df[series_df.columns[0]].dt.year
                
                monthly_stats = series_df.groupby('month')['value'].agg(['mean', 'std']).reset_index()
                
                fig_seasonal = go.Figure()
                fig_seasonal.add_trace(go.Scatter(
                    x=monthly_stats['month'],
                    y=monthly_stats['mean'],
                    mode='lines+markers',
                    name='Media Mensile',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                ))
                
                # Aggiungi bande di confidenza
                fig_seasonal.add_trace(go.Scatter(
                    x=monthly_stats['month'],
                    y=monthly_stats['mean'] + monthly_stats['std'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig_seasonal.add_trace(go.Scatter(
                    x=monthly_stats['month'],
                    y=monthly_stats['mean'] - monthly_stats['std'],
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(0,100,80,0.2)',
                    fill='tonexty',
                    name='±1 Deviazione Standard',
                    hoverinfo='skip'
                ))
                
                fig_seasonal.update_layout(
                    title="Profilo Stagionale (per Mese)",
                    xaxis_title="Mese",
                    yaxis_title="Valore Medio",
                    xaxis=dict(tickmode='linear', tick0=1, dtick=1),
                    height=400
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)
            else:
                st.info("Non ci sono dati sufficienti per l'analisi stagionale dettagliata.")
        except Exception as e:
            st.warning(f"Impossibile generare l'analisi stagionale: {e}")

        st.markdown("#### Decomposizione della Serie Storica (STL)")
        st.write("La decomposizione STL (Seasonal-Trend decomposition using Loess) scompone la serie in tre componenti: trend, stagionalità e residuo. Questo aiuta a capire i pattern sottostanti.")
        
        seasonal_period = holtwinters_params.get('seasonal_periods', 12)
        if series.shape[0] <= 2 * seasonal_period:
            st.warning(f"La serie è troppo corta per la decomposizione STL con periodo {seasonal_period}. Salto questa analisi.")
        else:
            try:
                decomp = seasonal_decompose(series, model='additive', period=seasonal_period)
                st.plotly_chart(plot_decomposition(decomp, "Decomposizione Additiva"), use_container_width=True)
            except Exception as e:
                st.error(f"Errore durante la decomposizione STL: {e}")

        st.markdown("#### Analisi di Stazionarietà (ACF, PACF, ADF Test)")
        st.write("Questi test aiutano a capire se la serie è stazionaria, una proprietà importante per molti modelli (come ARIMA). Una serie è stazionaria se le sue proprietà statistiche (media, varianza) non cambiano nel tempo.")

        # ACF/PACF
        try:
            st.pyplot(plot_acf_pacf(series))
            st.info("""
            - **ACF (Autocorrelation Function):** Mostra la correlazione della serie con le sue versioni passate (lag). Un decadimento lento suggerisce non-stazionarietà.
            - **PACF (Partial Autocorrelation Function):** Mostra la correlazione 'pura' con un lag, rimuovendo l'effetto dei lag intermedi.
            """)
        except Exception as e:
            st.error(f"Errore nella generazione dei grafici ACF/PACF: {e}")

        # ADF Test
        try:
            adf_result = adfuller(series.dropna())
            st.write("**Augmented Dickey-Fuller Test:**")
            st.write(f'ADF Statistic: {adf_result[0]:.4f}')
            st.write(f'p-value: {adf_result[1]:.4f}')
            if adf_result[1] <= 0.05:
                st.success("Il p-value è <= 0.05. Si rigetta l'ipotesi nulla (H0), quindi la serie è probabilmente **stazionaria**.")
            else:
                st.warning("Il p-value è > 0.05. Non si può rigettare l'ipotesi nulla (H0), quindi la serie è probabilmente **non-stazionaria** e potrebbe richiedere una differenziazione (parametro 'd' in ARIMA/SARIMA).")
        except Exception as e:
            st.error(f"Errore durante l'esecuzione del test ADF: {e}")


    # --- 2. Confronto Performance Modelli ---
    st.subheader("2. Confronto Performance Modelli")

    # Importa le funzioni dei modelli qui per evitare import circolari
    from .prophet_module import run_prophet_model
    from .arima_module import run_arima_model
    from .sarima_module import run_sarima_model
    from .holtwinters_module import run_holt_winters_model

    # Parametri base comuni a tutti i modelli
    base_model_args = {
        "df": df,
        "date_col": date_col,
        "target_col": target_col,
        "horizon": horizon,
        "selected_metrics": selected_metrics
    }

    # Parametri estesi per Prophet (supporta CV e backtesting avanzato)
    prophet_model_args = {
        **base_model_args,
        "freq": freq,
        "make_forecast": make_forecast,
        "use_cv": use_cv,
        "cv_start_date": kwargs.get('cv_start_date'),
        "cv_end_date": kwargs.get('cv_end_date'),
        "n_folds": kwargs.get('n_folds', 5),
        "fold_horizon": kwargs.get('fold_horizon', 30),
        "test_start_date": test_start_date,
        "test_end_date": test_end_date,
    }

    # Definizione dei modelli da eseguire con i loro parametri specifici
    models_to_run = {
        "Prophet": (run_prophet_model, prophet_model_args, prophet_params),
        "ARIMA": (run_arima_model, base_model_args, arima_params),
        "SARIMA": (run_sarima_model, base_model_args, sarima_params),
        "Holt-Winters": (run_holt_winters_model, base_model_args, holtwinters_params)
    }

    results = {}
    model_metrics = {}  # Dizionario per conservare le metriche di ogni modello
    progress_bar = st.progress(0)
    total_models = len(models_to_run)

    for i, (model_name, (model_func, model_args, model_params)) in enumerate(models_to_run.items()):
        st.write(f"---")
        st.write(f"### Esecuzione Modello: **{model_name}**")
        try:
            # FIX: Usa deepcopy per i parametri di Prophet per evitare side-effects
            current_params = copy.deepcopy(model_params) if model_name == "Prophet" else model_params
            
            # Prima esecuzione: raccoglie le metriche (senza output visivi)
            model_result = model_func(**model_args, params=current_params, return_metrics=True)
            
            # Seconda esecuzione: mostra i grafici e l'output completo (senza raccogliere metriche)
            model_func(**model_args, params=current_params, return_metrics=False)
            
            # Salva le metriche del modello
            if model_result and isinstance(model_result, dict):
                model_metrics[model_name] = model_result
                results[model_name] = {"metrics": "Eseguito", "forecast": None, "fig": None}
                st.success(f"✅ Modello {model_name} eseguito con successo.")
            else:
                results[model_name] = {"metrics": None, "forecast": None, "fig": None}
                st.error(f"❌ Modello {model_name} non ha restituito risultati validi.")
        except Exception as e:
            st.error(f"❌ Errore durante l'esecuzione di {model_name}: {e}")
            results[model_name] = {"metrics": None, "forecast": None, "fig": None}
        progress_bar.progress((i + 1) / total_models)

    # --- 3. Risultati e Riepilogo ---
    st.subheader("3. Risultati dell'Esecuzione")
    
    executed_models = []
    failed_models = []
    
    for model_name, result in results.items():
        if result["metrics"] is not None:
            executed_models.append(model_name)
        else:
            failed_models.append(model_name)

    if executed_models:
        st.success(f"✅ **Modelli eseguiti con successo:** {', '.join(executed_models)}")
        st.info("💡 Tutti i modelli sono stati eseguiti con i parametri specificati dall'utente. I risultati completi sono visibili sopra per ogni modello.")
    
    if failed_models:
        st.error(f"❌ **Modelli falliti:** {', '.join(failed_models)}")

    # --- 3. Tabella di Comparazione delle Performance ---
    if model_metrics and len(model_metrics) > 1:
        st.subheader("3. Comparazione Performance dei Modelli")
        
        # Crea una tabella con le metriche di tutti i modelli
        metrics_df = pd.DataFrame(model_metrics).T
        
        # Rimuovi colonne che non sono metriche numeriche
        numeric_cols = []
        for col in metrics_df.columns:
            try:
                pd.to_numeric(metrics_df[col])
                numeric_cols.append(col)
            except:
                continue
        
        if numeric_cols:
            metrics_df_clean = metrics_df[numeric_cols].round(3)
            
            st.write("**Tabella delle Metriche per Modello:**")
            st.dataframe(metrics_df_clean, use_container_width=True)
            
            # Trova il modello migliore per ogni metrica
            st.write("**Modello Migliore per Metrica:**")
            best_models = {}
            
            for metric in numeric_cols:
                if metric.upper() in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE']:
                    # Per queste metriche, il valore più basso è migliore
                    best_model = metrics_df_clean[metric].idxmin()
                    best_value = metrics_df_clean[metric].min()
                else:
                    # Per altre metriche, il valore più alto è migliore
                    best_model = metrics_df_clean[metric].idxmax()
                    best_value = metrics_df_clean[metric].max()
                
                best_models[metric] = (best_model, best_value)
            
            cols = st.columns(len(best_models))
            for i, (metric, (model, value)) in enumerate(best_models.items()):
                if metric.upper() in ['MAPE', 'SMAPE']:
                    cols[i].metric(f"Migliore {metric}", f"{model}", f"{value:.1f}%")
                else:
                    cols[i].metric(f"Migliore {metric}", f"{model}", f"{value:.3f}")
            
            # Suggerimento del modello complessivamente migliore
            st.write("**🏆 Suggerimento Modello Migliore:**")
            
            # Calcola uno score composito basato su ranking delle metriche
            model_scores = {}
            for model in metrics_df_clean.index:
                score = 0
                for metric in numeric_cols:
                    if metric.upper() in ['MAE', 'MSE', 'RMSE', 'MAPE', 'SMAPE']:
                        # Ranking inverso per metriche dove il valore basso è migliore
                        rank = metrics_df_clean[metric].rank(ascending=True)
                    else:
                        # Ranking normale per metriche dove il valore alto è migliore
                        rank = metrics_df_clean[metric].rank(ascending=False)
                    score += rank[model]
                model_scores[model] = score
            
            best_overall_model = min(model_scores, key=model_scores.get)
            
            st.success(f"🎯 **{best_overall_model}** appare essere il modello con le performance migliori nel complesso.")
            
            with st.expander("💡 Come interpretare i risultati", expanded=False):
                st.info("""
                **Criteri di Valutazione:**
                - **MAE (Mean Absolute Error)**: Errore medio assoluto - più basso è meglio
                - **MSE (Mean Squared Error)**: Errore quadratico medio - più basso è meglio  
                - **RMSE (Root Mean Squared Error)**: Radice dell'errore quadratico medio - più basso è meglio
                - **MAPE (Mean Absolute Percentage Error)**: Errore percentuale medio - più basso è meglio
                - **SMAPE (Symmetric Mean Absolute Percentage Error)**: Errore percentuale simmetrico - più basso è meglio
                
                **Raccomandazione del Modello:**
                Il modello suggerito è scelto in base a un ranking composito che considera tutte le metriche.
                Tuttavia, la scelta finale dovrebbe considerare anche:
                - Il tipo di dati e la loro stagionalità
                - La facilità di interpretazione del modello
                - I requisiti di velocità di esecuzione
                - La stabilità delle previsioni nel tempo
                """)
        else:
            st.warning("⚠️ Non sono state trovate metriche numeriche valide per la comparazione.")
    elif len(model_metrics) == 1:
        st.info("ℹ️ Solo un modello è stato eseguito con successo. Per una comparazione completa, prova ad eseguire più modelli.")
    else:
        st.warning("⚠️ Nessun modello ha restituito metriche valide per la comparazione.")

    # --- 5. Note Metodologiche ---
    st.subheader("4. Note Metodologiche")
    
    st.info("""
    **Confronto dei Modelli:**
    
    - **Prophet**: Modello robusto per serie temporali con stagionalità e trend non lineari. Supporta festività e gestisce automaticamente outlier.
    - **ARIMA**: Modello classico per serie temporali stazionarie. Richiede spesso pre-processing (differenziazione).
    - **SARIMA**: Estensione di ARIMA che include componenti stagionali. Adatto per dati con stagionalità regolare.
    - **Holt-Winters**: Metodo di smoothing esponenziale. Buono per dati con trend e stagionalità evidenti.
    
    **Parametri Utilizzati:**
    - Ogni modello è stato eseguito con i parametri specifici configurati dall'utente
    - Prophet utilizza parametri avanzati come cross-validation se abilitata
    - Altri modelli utilizzano parametri base per garantire stabilità
    """)

    # Rimuovi le sezioni sui grafici di confronto che non funzionano senza i dati di forecast
    # st.subheader("5. Confronto Visivo delle Previsioni")
    # fig_comp = go.Figure()
    # ... resto del codice del grafico rimosso per ora
