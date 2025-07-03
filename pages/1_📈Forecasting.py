import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from modules.prophet_module import run_prophet_model
from modules.arima_module import run_arima_model
from modules.holtwinters_module import run_holt_winters_model
from modules.sarima_module import run_sarima_model
from modules.exploratory_module import run_exploratory_analysis

st.title("📈 Contact Center Forecasting Tool")

# --- Funzioni di Supporto ---

def generate_sample_data():
    """Genera un DataFrame di esempio con trend, stagionalità e rumore."""
    today = datetime.today()
    dates = pd.date_range(start=today - timedelta(days=730), end=today, freq='D')
    trend = np.linspace(50, 150, len(dates))
    seasonality = 25 * (1 + np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25)) + \
                  15 * (1 + np.sin(np.arange(len(dates)) * 2 * np.pi / 7))
    noise = np.random.normal(0, 15, len(dates))
    volume = trend + seasonality + noise
    volume = np.maximum(0, volume)  # Assicura che non ci siano valori negativi
    return pd.DataFrame({'date': dates, 'volume': volume})

def clean_data(df, cleaning_preferences, target_col):
    if cleaning_preferences['remove_zeros']:
        df = df[df[target_col] != 0]
    if cleaning_preferences['replace_outliers']:
        q1 = df[target_col].quantile(0.25)
        q3 = df[target_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df.loc[(df[target_col] < lower_bound) | (df[target_col] > upper_bound), target_col] = df[target_col].median()
    if cleaning_preferences['remove_negatives']:
        df[target_col] = df[target_col].clip(lower=0)
    
    # Gestione dei valori mancanti
    if 'nan_handling' in cleaning_preferences:
        if cleaning_preferences['nan_handling'] == "Riempi con zero":
            df[target_col] = df[target_col].fillna(0)
        elif cleaning_preferences['nan_handling'] == "Forward fill":
            df[target_col] = df[target_col].ffill()
        elif cleaning_preferences['nan_handling'] == "Backward fill":
            df[target_col] = df[target_col].bfill()

    return df

def check_data_quality(df, date_col, target_col):
    """Controlla la qualità del DataFrame e mostra warning."""
    # Warning per valori negativi
    if (df[target_col] < 0).any():
        st.warning(f"⚠️ La colonna target '{target_col}' contiene valori negativi. Verranno trasformati in zero se l'opzione 'Trasforma valori negativi in zero' è attiva.")

    # Warning per NaN dopo aggregazione
    nan_percentage = df[target_col].isna().sum() / len(df)
    if nan_percentage > 0.2:
        st.warning(f"⚠️ Attenzione: oltre il {nan_percentage:.0%} dei valori nella colonna '{target_col}' sono mancanti (NaN) dopo l'aggregazione. Considera di usare un metodo di riempimento o di rivedere la granularità.")

def check_data_size(df):
    if len(df) < 20: # Aumentato il limite per modelli più robusti
        st.error("Il dataset ha meno di 20 punti dati dopo la pulizia. Aggiungi più dati o modifica le regole di pulizia/filtro.")
        st.stop()

def aggregate_data(df, date_col, target_col, freq, aggregation_method):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Feedback su NaT
    nat_rows = df[df[date_col].isna()]
    if not nat_rows.empty:
        st.warning(f"⚠️ Trovate {len(nat_rows)} righe con date non valide (NaT) che verranno escluse.")
        df = df.dropna(subset=[date_col])

    df = df.set_index(date_col).resample(freq).agg({target_col: aggregation_method}).reset_index()
    return df

# --- Sidebar ---
with st.sidebar:
    st.header("1. Dataset")

    use_sample_data = st.checkbox("Usa dataset di esempio", value=True, help="Deseleziona per caricare il tuo file CSV.")

    file = None
    if not use_sample_data:
        with st.expander("📂 File import"):
            delimiter = st.selectbox("Delimitatore CSV", [",", ";", "|", "\t"], index=0, help="Il carattere che separa le colonne nel tuo file.")
            user_friendly_format = st.selectbox("Formato data", [
                "aaaa-mm-gg", "gg/mm/aaaa", "gg/mm/aa",
                "mm/gg/aaaa", "gg.mm.aaaa", "aaaa/mm/gg"
            ], index=0, help="Seleziona il formato delle date nel tuo file per una corretta interpretazione.")
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
    else:
        st.info("ℹ️ Verrà utilizzato un dataset di esempio generato automaticamente.")

    df, date_col, target_col, freq, aggregation_method = None, None, None, "D", "sum"
    clip_negatives = replace_outliers = clean_zeros = False
    test_start, test_end = None, None

    # Caricamento e configurazione iniziale del DataFrame
    if use_sample_data:
        df = generate_sample_data()
        date_col = 'date'
        target_col = 'volume'
    elif file:
        df = pd.read_csv(file, delimiter=delimiter)
        columns = df.columns.tolist()

        with st.expander("📊 Colonne"):
            date_col = st.selectbox("Colonna data", options=columns, help="La colonna che contiene le date.")
            target_col = st.selectbox("Colonna target", options=columns, index=1 if len(columns) > 1 else 0, help="La colonna con i valori numerici da prevedere.")

    if df is not None and not df.empty:
        # --- Filtro Intervallo Temporale ---
        with st.expander("🗓️ Filtro Intervallo Temporale", expanded=True):
            df[date_col] = pd.to_datetime(df[date_col], format=None if use_sample_data else date_format, errors='coerce')
            df = df.dropna(subset=[date_col]) # Rimuovi NaT prima di calcolare min/max
            
            min_date = df[date_col].min().date()
            max_date = df[date_col].max().date()

            start_date = st.date_input("Data di inizio", value=min_date, min_value=min_date, max_value=max_date, help="Filtra i dati per iniziare da questa data.")
            end_date = st.date_input("Data di fine", value=max_date, min_value=start_date, max_value=max_date, help="Filtra i dati per finire a questa data.")
            
            df = df[(df[date_col] >= pd.to_datetime(start_date)) & (df[date_col] <= pd.to_datetime(end_date))]

        with st.expander("⏱️ Granularità"):
            try:
                df_sorted = df.sort_values(by=date_col)
                inferred = pd.infer_freq(df_sorted[date_col])
                detected_freq = inferred if inferred else "D"
            except Exception:
                detected_freq = "D"
            st.text(f"Granularità rilevata: {detected_freq}")
            freq_map = {
                "Daily": "D", "Weekly": "W", "Monthly": "M",
                "Quarterly": "Q", "Yearly": "Y"
            }
            user_friendly_freq = {v: k for k, v in freq_map.items()}.get(detected_freq, "Daily")
            selected_granularity = st.selectbox("Seleziona una nuova granularità", list(freq_map.keys()), index=list(freq_map.keys()).index(user_friendly_freq) if user_friendly_freq in freq_map.keys() else 0, help="Aggrega i dati a un livello temporale diverso (es. da giornaliero a settimanale).")
            freq = freq_map[selected_granularity]
            aggregation_method = st.selectbox("Metodo di aggregazione", ["sum", "mean", "max", "min"], help="La funzione da usare per aggregare i dati (es. somma dei volumi giornalieri per ottenere il totale settimanale).")

        with st.expander("🧹 Data Cleaning"):
            clean_zeros = st.checkbox("Rimuovi righe con zero nel target", value=True, help="Esclude i giorni/periodi con zero attività. Utile se gli zeri indicano assenza di servizio.")
            replace_outliers = st.checkbox("Sostituisci outlier (z-score > 3) con mediana", value=True, help="Smussa i picchi anomali che potrebbero distorcere il modello, sostituendoli con un valore più tipico.")
            clip_negatives = st.checkbox("Trasforma valori negativi in zero", value=True, help="Forza i valori negativi (impossibili per i volumi) a zero.")
            nan_handling_method = st.selectbox(
                "Gestisci valori mancanti (post-aggregazione)",
                ["Riempi con zero", "Forward fill", "Backward fill"],
                index=0,
                help="Scegli come trattare le date senza dati. 'Riempi con zero' è conservativo, 'Forward fill' propaga l'ultimo valore noto."
            )

        st.header("2. Modello")
        model_tab = st.selectbox("Seleziona il modello", ["Prophet", "ARIMA", "SARIMA", "Holt-Winters", "Exploratory"], help="Scegli l'algoritmo di forecasting da utilizzare. 'Exploratory' li confronta tutti.")

        # --- Parametri Comuni per Exploratory ---
        if model_tab == "Exploratory":
            st.info("Exploratory confronterà tutti i modelli. I parametri qui sotto servono a configurare ciascun modello per il confronto.")

        # --- Parametri Prophet (usati anche da Exploratory) ---
        prophet_params = {}
        if model_tab in ["Prophet", "Exploratory"]:
            with st.expander("⚙️ Parametri Prophet"):
                prophet_params['seasonality_mode'] = st.selectbox(
                    "Modalità stagionalità", ['additive', 'multiplicative'], index=0,
                    help="Additiva: la stagionalità ha un'ampiezza costante. Moltiplicativa: l'ampiezza cresce con il trend."
                )
                prophet_params['changepoint_prior_scale'] = st.slider(
                    "Flessibilità del trend", 0.001, 0.5, 0.05, 0.001, format="%.3f",
                    help="Aumenta per rendere il trend più flessibile e adattarsi ai cambiamenti, diminuisci per renderlo più stabile."
                )
                prophet_params['holidays_country'] = st.selectbox(
                    "Includi festività nazionali", [None, 'IT', 'US', 'UK', 'DE', 'FR', 'ES'],
                    help="Aggiunge al modello il calendario delle festività del paese selezionato per catturarne l'effetto."
                )

        # --- Parametri Holt-Winters (usati anche da Exploratory) ---
        holt_winters_params = {}
        if model_tab in ["Holt-Winters", "Exploratory"]:
            with st.expander("⚙️ Parametri Holt-Winters"):
                holt_winters_params['trend_type'] = st.selectbox("Tipo di Trend", ['add', 'mul'], index=0, key="hw_trend", help="'add' per trend lineare, 'mul' per trend esponenziale.")
                holt_winters_params['seasonal_type'] = st.selectbox("Tipo di Stagionalità", ['add', 'mul'], index=0, key="hw_seasonal", help="'add' per stagionalità costante, 'mul' per stagionalità che cresce con il trend.")
                holt_winters_params['damped_trend'] = st.checkbox("Smorza Trend (Damped)", value=True, key="hw_damped", help="Se attivo, smorza il trend nel lungo periodo, rendendo il forecast più conservativo.")
                holt_winters_params['seasonal_periods'] = st.number_input("Periodi stagionali", min_value=2, max_value=365, value=12, key="hw_seasonal_periods", help="Il numero di periodi in un ciclo stagionale (es. 7 per dati giornalieri con stagionalità settimanale, 12 per mensili).")
                
                use_custom = st.checkbox("Utilizza parametri di smoothing custom", value=False, key="hw_custom", help="Attiva per impostare manualmente i pesi per livello, trend e stagionalità.")
                holt_winters_params['use_custom'] = use_custom
                
                if use_custom:
                    holt_winters_params['smoothing_level'] = st.slider("Alpha (livello)", 0.0, 1.0, 0.2, 0.01, key="hw_alpha", help="Peso dato alle osservazioni recenti per il calcolo del livello.")
                    holt_winters_params['smoothing_trend'] = st.slider("Beta (trend)", 0.0, 1.0, 0.1, 0.01, key="hw_beta", help="Peso dato alle osservazioni recenti per il calcolo del trend.")
                    holt_winters_params['smoothing_seasonal'] = st.slider("Gamma (stagionalità)", 0.0, 1.0, 0.1, 0.01, key="hw_gamma", help="Peso dato alle osservazioni recenti per il calcolo della stagionalità.")

        # --- Parametri ARIMA (usati anche da Exploratory) ---
        arima_params = {}
        if model_tab in ["ARIMA", "Exploratory"]:
            with st.expander("⚙️ Parametri ARIMA"):
                arima_params['p'] = st.number_input("Ordine AR (p)", 0, 10, 1, key="arima_p", help="Componente Autoregressiva: numero di osservazioni passate da includere nel modello.")
                arima_params['d'] = st.number_input("Ordine Differenziazione (d)", 0, 5, 1, key="arima_d", help="Numero di volte che i dati vengono differenziati per renderli stazionari.")
                arima_params['q'] = st.number_input("Ordine MA (q)", 0, 10, 0, key="arima_q", help="Media Mobile: numero di errori di previsione passati da includere nel modello.")

        # --- Parametri SARIMA (usati anche da Exploratory) ---
        sarima_params = {}
        if model_tab in ["SARIMA", "Exploratory"]:
            with st.expander("⚙️ Parametri SARIMA"):
                st.write("Componente non stagionale (ARIMA)")
                sarima_params['p'] = st.number_input("Ordine AR (p)", 0, 10, 1, key="sarima_p", help="Parte AR non stagionale.")
                sarima_params['d'] = st.number_input("Ordine Diff. (d)", 0, 5, 1, key="sarima_d", help="Parte di differenziazione non stagionale.")
                sarima_params['q'] = st.number_input("Ordine MA (q)", 0, 10, 0, key="sarima_q", help="Parte MA non stagionale.")
                st.write("Componente stagionale")
                sarima_params['P'] = st.number_input("Ordine AR Stag. (P)", 0, 10, 1, key="sarima_P", help="Parte AR stagionale.")
                sarima_params['D'] = st.number_input("Ordine Diff. Stag. (D)", 0, 5, 1, key="sarima_D", help="Parte di differenziazione stagionale.")
                sarima_params['Q'] = st.number_input("Ordine MA Stag. (Q)", 0, 10, 0, key="sarima_Q", help="Parte MA stagionale.")
                sarima_params['s'] = st.number_input("Periodo Stag. (s)", 1, 365, 12, key="sarima_s", help="Durata del ciclo stagionale (es. 12 per dati mensili).")

        st.header("3. Backtesting")
        with st.expander("📊 Split", expanded=True):
            use_cv = st.checkbox("Usa Cross-Validation", help="Valida il modello su più 'fold' (segmenti) di dati per una stima più robusta delle performance.")
            if use_cv:
                end_date_default = df[date_col].max()
                start_date_default = max(df[date_col].min(), end_date_default - pd.DateOffset(years=1))

                col1, col2 = st.columns(2)
                with col1:
                    cv_start_date = st.date_input("Data inizio CV", value=start_date_default, help="La data da cui iniziare a creare i fold di validazione.")
                with col2:
                    cv_end_date = st.date_input("Data fine CV", value=end_date_default, help="La data finale per i fold di validazione.")
                n_folds = st.number_input("Numero di folds", 2, 20, 5, help="In quanti segmenti dividere i dati per la validazione incrociata.")
                fold_horizon = st.number_input("Orizzonte per fold (in periodi)", min_value=1, value=30, help="Quanti periodi futuri prevedere per ogni fold.")
            else:
                cv_start_date = cv_end_date = None
                n_folds = 5
                fold_horizon = 30
                
                split_point = st.slider(
                    "Punto di divisione Train/Test", 
                    min_value=0.1, max_value=0.9, value=0.8, step=0.05, format="%.2f",
                    help="Scegli la percentuale di dati da usare per l'addestramento (train). Il resto sarà usato per la validazione (test)."
                )
                
                if df is not None and not df.empty:
                    train_size = int(len(df) * split_point)
                    test_size = len(df) - train_size
                    test_start = df[date_col].iloc[train_size]
                    test_end = df[date_col].iloc[-1]
                    
                    st.info(
                        f"**Train set**: {train_size} righe ({split_point:.0%}) - da {df[date_col].min().date()} a {df[date_col].iloc[train_size-1].date()}\n\n"
                        f"**Test set**: {test_size} righe ({1-split_point:.0%}) - da {test_start.date()} a {test_end.date()}"
                    )

        with st.expander("📏 Metriche"):
            selected_metrics = st.multiselect(
                "Seleziona le metriche di valutazione",
                options=["MAPE", "MAE", "MSE", "RMSE", "SMAPE"],
                default=["MAPE", "MAE", "RMSE"],
                help="Scegli gli indicatori per misurare l'accuratezza del modello. MAE e RMSE sono in valore assoluto, MAPE in percentuale."
            )

        st.header("4. Forecast")
        with st.expander("📅 Parametri Forecast"):
            make_forecast = st.checkbox("Esegui forecast su date future", value=True, help="Se attivo, il modello genererà previsioni per il futuro, oltre a validare sul passato.")
            horizon = 30
            if make_forecast:
                horizon = st.number_input("Orizzonte (numero di periodi)", min_value=1, value=30, help="Per quanti periodi futuri (giorni, settimane, mesi) vuoi la previsione.")
                if df is not None and not df.empty:
                    start_date_fc = df[date_col].max() + pd.tseries.frequencies.to_offset(freq)
                    end_date_fc = start_date_fc + pd.tseries.frequencies.to_offset(freq) * (horizon - 1)
                    st.success(f"Il forecast coprirà da **{start_date_fc.date()}** a **{end_date_fc.date()}**")
        
        forecast_button = st.button("🚀 Avvia il forecast")

# --- Azione Principale ---
if (file or use_sample_data) and forecast_button:
    # Aggregazione e pulizia
    df_agg = aggregate_data(df, date_col, target_col, freq, aggregation_method)
    cleaning_preferences = {
        'remove_zeros': clean_zeros, 'remove_negatives': clip_negatives,
        'replace_outliers': replace_outliers, 'nan_handling': nan_handling_method
    }
    df_clean = clean_data(df_agg, cleaning_preferences, target_col)

    # Centralizzazione della gestione della frequenza per coerenza tra i moduli
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    df_clean = df_clean.set_index(date_col)
    df_clean = df_clean.asfreq(freq, method='ffill')
    df_clean = df_clean.reset_index()
    
    # Feedback post-pulizia
    st.markdown("### 📦 Anteprima Dati Post-Cleaning")
    st.dataframe(df_clean.head())
    check_data_quality(df_clean, date_col, target_col)
    check_data_size(df_clean)

    # Plot della serie storica con trendline e filtri
    st.markdown("### 📈 Serie Storica e Trendline")
    
    df_plot = df_clean.sort_values(date_col).reset_index(drop=True)

    x = np.arange(len(df_plot))
    m, b = np.polyfit(x, df_plot[target_col], 1)
    df_plot['trend'] = m * x + b

    fig = go.Figure([
        go.Scatter(x=df_plot[date_col], y=df_plot[target_col], name='Storico', mode='lines', line=dict(shape='spline', color='#1f77b4')),
        go.Scatter(x=df_plot[date_col], y=df_plot['trend'], name='Trendline', mode='lines', line=dict(shape='spline', dash='dash', color='#ff7f0e'))
    ])
    fig.update_layout(title="Andamento Storico e Trend Lineare", xaxis_title="Data", yaxis_title=target_col)
    st.plotly_chart(fig, use_container_width=True)

    # Definizione dei parametri base comuni a tutti i modelli
    base_args = {
        "df": df_clean,
        "date_col": date_col,
        "target_col": target_col,
        "horizon": horizon,
        "selected_metrics": selected_metrics
    }

    # Parametri aggiuntivi per Prophet (unico modello che supporta CV e backtesting avanzato)
    prophet_args = {
        **base_args,
        "freq": freq,
        "make_forecast": make_forecast,
        "use_cv": use_cv,
        "cv_start_date": cv_start_date,
        "cv_end_date": cv_end_date,
        "n_folds": n_folds,
        "fold_horizon": fold_horizon,
        "test_start_date": test_start,
        "test_end_date": test_end
    }

    # Esecuzione del modello selezionato
    if model_tab == "Prophet":
        run_prophet_model(**prophet_args, params=prophet_params)

    elif model_tab == "ARIMA":
        run_arima_model(**base_args, params=arima_params)

    elif model_tab == "SARIMA":
        run_sarima_model(**base_args, params=sarima_params)

    elif model_tab == "Holt-Winters":
        run_holt_winters_model(**base_args, params=holt_winters_params)

    elif model_tab == "Exploratory":
        # Passa tutti i parametri necessari per l'analisi esplorativa
        run_exploratory_analysis(
            **prophet_args,  # Usa prophet_args che contiene tutti i parametri
            prophet_params=prophet_params,
            holtwinters_params=holt_winters_params,
            arima_params=arima_params,
            sarima_params=sarima_params,
        )
elif not use_sample_data and not file:
    st.info("☝️ Per iniziare, carica un file CSV o seleziona il dataset di esempio dalla sidebar.")