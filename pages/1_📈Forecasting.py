import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from modules.prophet_module import run_prophet_model
from modules.arima_module import run_arima_model
from modules.holtwinters_module import run_holt_winters_model
from modules.sarima_module import run_sarima_model
from modules.exploratory_module import run_exploratory_analysis

st.title("📈 Contact Center Forecasting Tool")

# Funzioni di supporto
def clean_data(df, cleaning_preferences, target_col):
    if cleaning_preferences['remove_zeros']:
        df = df[df[target_col] != 0]
    if cleaning_preferences['replace_outliers']:
        z = (df[target_col] - df[target_col].mean())/df[target_col].std()
        df.loc[np.abs(z)>3, target_col] = df[target_col].median()
    if cleaning_preferences['remove_negatives']:
        df[target_col] = df[target_col].clip(lower=0)
    
    # Gestione dei valori mancanti introdotti dal resampling
    if 'nan_handling' in cleaning_preferences:
        if cleaning_preferences['nan_handling'] == "Riempi con zero":
            df[target_col] = df[target_col].fillna(0)
        elif cleaning_preferences['nan_handling'] == "Forward fill":
            df[target_col] = df[target_col].ffill()
        elif cleaning_preferences['nan_handling'] == "Backward fill":
            df[target_col] = df[target_col].bfill()

    return df

def check_data_size(df):
    if len(df) < 10:
        st.error("Il dataset è troppo piccolo dopo la pulizia. Aggiungi più dati o modifica le regole di pulizia.")
        st.stop()

def aggregate_data(df, date_col, target_col, freq, aggregation_method):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).resample(freq).agg({target_col: aggregation_method}).reset_index()
    return df

# Sidebar
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
    test_start, test_end = None, None

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
            freq_map = {
                "Daily": "D",
                "Weekly": "W",
                "Monthly": "M",
                "Quarterly": "Q",
                "Yearly": "Y"
            }
            user_friendly_freq = {v: k for k, v in freq_map.items()}.get(detected_freq, "Daily")
            selected_granularity = st.selectbox("Seleziona una nuova granularità", list(freq_map.keys()), index=list(freq_map.keys()).index(user_friendly_freq) if user_friendly_freq in freq_map.keys() else 0)
            freq = freq_map[selected_granularity]
            aggregation_method = st.selectbox("Metodo di aggregazione", ["sum", "mean", "max", "min"])

        with st.expander("🧹 Data Cleaning"):
            clean_zeros = st.checkbox("Rimuovi righe con zero nel target", value=True)
            replace_outliers = st.checkbox("Sostituisci outlier (z-score > 3) con mediana", value=True)
            clip_negatives = st.checkbox("Trasforma valori negativi in zero", value=True)
            nan_handling_method = st.selectbox(
                "Gestisci valori mancanti (post-aggregazione)",
                ["Riempi con zero", "Forward fill", "Backward fill"],
                index=0,
                help="Scegli come trattare le date senza dati che possono essere state create durante l'aggregazione."
            )

        st.header("2. Modello")
        model_tab = st.selectbox("Seleziona il modello", ["Prophet", "ARIMA", "SARIMA", "Holt-Winters", "Exploratory"])

        # Parametri specifici per Prophet
        prophet_params = {}
        if model_tab == "Prophet":
            with st.expander("⚙️ Parametri Prophet"):
                prophet_params['seasonality_mode'] = st.selectbox(
                    "Modalità stagionalità",
                    ['additive', 'multiplicative'],
                    index=0
                )
                prophet_params['changepoint_prior_scale'] = st.slider(
                    "Flessibilità del trend",
                    min_value=0.001,
                    max_value=0.5,
                    value=0.05,
                    step=0.001,
                    format="%.3f"
                )
                prophet_params['holidays_country'] = st.selectbox(
                    "Includi festività nazionali",
                    [None, 'IT', 'US', 'UK', 'DE', 'FR', 'ES']
                )

        # Parametri specifici per Holt-Winters
        holt_winters_params = {}
        if model_tab == "Holt-Winters":
            with st.expander("⚙️ Parametri Holt-Winters"):
                holt_winters_params['trend_type'] = st.selectbox("Tipo di Trend", ['add', 'mul'], index=0, key="hw_trend")
                holt_winters_params['seasonal_type'] = st.selectbox("Tipo di Stagionalità", ['add', 'mul'], index=0, key="hw_seasonal")
                holt_winters_params['damped_trend'] = st.checkbox("Smorza Trend (Damped)", value=True, key="hw_damped")
                holt_winters_params['seasonal_periods'] = st.number_input("Periodi stagionali", min_value=2, max_value=365, value=12, key="hw_seasonal_periods")
                
                use_custom = st.checkbox("Utilizza parametri di smoothing custom", value=False, key="hw_custom")
                holt_winters_params['use_custom'] = use_custom
                
                if use_custom:
                    holt_winters_params['smoothing_level'] = st.slider("Alpha (livello)", 0.0, 1.0, 0.2, 0.01, key="hw_alpha")
                    holt_winters_params['smoothing_trend'] = st.slider("Beta (trend)", 0.0, 1.0, 0.1, 0.01, key="hw_beta")
                    holt_winters_params['smoothing_seasonal'] = st.slider("Gamma (stagionalità)", 0.0, 1.0, 0.1, 0.01, key="hw_gamma")

        # Parametri specifici per ARIMA
        arima_params = {}
        if model_tab == "ARIMA":
            with st.expander("⚙️ Parametri ARIMA"):
                arima_params['p'] = st.number_input("Ordine AR (p)", min_value=0, max_value=10, value=1, key="arima_p")
                arima_params['d'] = st.number_input("Ordine Differenziazione (d)", min_value=0, max_value=5, value=1, key="arima_d")
                arima_params['q'] = st.number_input("Ordine MA (q)", min_value=0, max_value=10, value=0, key="arima_q")

        # Parametri specifici per SARIMA
        sarima_params = {}
        if model_tab == "SARIMA":
            with st.expander("⚙️ Parametri SARIMA"):
                st.write("Componente non stagionale (ARIMA)")
                sarima_params['p'] = st.number_input("Ordine AR (p)", min_value=0, max_value=10, value=1, key="sarima_p")
                sarima_params['d'] = st.number_input("Ordine Diff. (d)", min_value=0, max_value=5, value=1, key="sarima_d")
                sarima_params['q'] = st.number_input("Ordine MA (q)", min_value=0, max_value=10, value=0, key="sarima_q")
                st.write("Componente stagionale")
                sarima_params['P'] = st.number_input("Ordine AR Stag. (P)", min_value=0, max_value=10, value=1, key="sarima_P")
                sarima_params['D'] = st.number_input("Ordine Diff. Stag. (D)", min_value=0, max_value=5, value=1, key="sarima_D")
                sarima_params['Q'] = st.number_input("Ordine MA Stag. (Q)", min_value=0, max_value=10, value=0, key="sarima_Q")
                sarima_params['s'] = st.number_input("Periodo Stag. (s)", min_value=1, max_value=365, value=12, key="sarima_s")


        st.header("3. Backtesting")
        with st.expander("📊 Split"):
            use_cv = st.checkbox("Usa Cross-Validation")
            if use_cv:
                # Calcola le date di default per la CV
                end_date_default = df[date_col].max()
                start_date_default = max(df[date_col].min(), end_date_default - pd.DateOffset(years=1))

                col1, col2 = st.columns(2)
                with col1:
                    cv_start_date = st.date_input("Data inizio CV", value=start_date_default)
                with col2:
                    cv_end_date = st.date_input("Data fine CV", value=end_date_default)
                n_folds = st.number_input("Numero di folds", min_value=2, max_value=20, value=5)
                fold_horizon = st.number_input("Orizzonte per fold (in periodi)", min_value=1, value=30)
            else:
                cv_start_date = cv_end_date = None
                n_folds = 5
                fold_horizon = 30

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
            horizon = 30
            if make_forecast:
                horizon = st.number_input("Orizzonte (numero di periodi)", min_value=1, value=30)
                if df is not None and not df.empty:
                    start_date = df[date_col].max() + pd.tseries.frequencies.to_offset(freq)
                    end_date = start_date + pd.tseries.frequencies.to_offset(freq) * (horizon - 1)
                    st.success(f"Il forecast coprirà il periodo da **{start_date.date()}** a **{end_date.date()}**")
        forecast_button = st.button("🚀 Avvia il forecast")

# Main action
if file and forecast_button:
    df = aggregate_data(df, date_col, target_col, freq, aggregation_method)
    cleaning_preferences = {
        'remove_zeros': clean_zeros,
        'remove_negatives': clip_negatives,
        'replace_outliers': replace_outliers,
        'nan_handling': nan_handling_method
    }
    df = clean_data(df, cleaning_preferences, target_col)
    check_data_size(df)

    # Plot della serie storica con trendline
    st.markdown("### Serie storica e trendline")
    df_plot = df.sort_values(date_col).reset_index(drop=True)
    # costruiamo l'indice X per il fit
    x = np.arange(len(df_plot))
    # calcolo pendenza e intercetta
    m, b = np.polyfit(x, df_plot[target_col], 1)
    df_plot['trend'] = m * x + b

    fig = go.Figure([
        go.Scatter(
            x=df_plot[date_col],
            y=df_plot[target_col],
            name='Storico',
            mode='lines',
            line=dict(shape='spline', color='#1f77b4')
        ),
        go.Scatter(
            x=df_plot[date_col],
            y=df_plot['trend'],
            name='Trendline',
            mode='lines',
            line=dict(shape='spline', dash='dash', color='#ff7f0e')
        )
    ])
    st.plotly_chart(fig, use_container_width=True)

    if model_tab == "Prophet":
        run_prophet_model(
            df,
            date_col,
            target_col,
            freq,
            horizon,
            make_forecast=make_forecast,
            use_cv=use_cv,
            cv_start_date=cv_start_date,
            cv_end_date=cv_end_date,
            n_folds=n_folds,
            fold_horizon=fold_horizon,
            test_start_date=test_start if not use_cv else None,
            test_end_date=test_end if not use_cv else None,
            params=prophet_params,
            selected_metrics=selected_metrics
        )

    elif model_tab == "ARIMA":
        run_arima_model(
            df,
            date_col=date_col,
            target_col=target_col,
            horizon=horizon,
            selected_metrics=selected_metrics,
            params=arima_params
        )

    elif model_tab == "SARIMA":
        run_sarima_model(
            df,
            date_col=date_col,
            target_col=target_col,
            horizon=horizon,
            selected_metrics=selected_metrics,
            params=sarima_params
        )

    elif model_tab == "Holt-Winters":
        run_holt_winters_model(
            df,
            date_col=date_col,
            target_col=target_col,
            horizon=horizon,
            selected_metrics=selected_metrics,
            params=holt_winters_params
        )

    elif model_tab == "Exploratory":
        run_exploratory_analysis(df, date_col, target_col)
