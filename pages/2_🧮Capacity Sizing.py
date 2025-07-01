import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson

st.set_page_config(layout="wide")

st.title("🧮 Capacity Sizing Tool")
st.markdown("Questo strumento calcola il numero di operatori necessari per gestire i volumi di chiamate in base ai KPI aziendali.")

# --- Funzioni di Calcolo ---

def erlang_c(arrival_rate, aht, service_level_target, answer_time_target):
    """
    Calcola il numero di agenti necessari utilizzando la formula di Erlang C.
    """
    if arrival_rate == 0:
        return 0, 1.0, 0.0

    # Intensità di traffico in Erlang
    traffic_intensity = arrival_rate * aht / 3600
    
    # Inizia con un numero di agenti leggermente superiore all'intensità di traffico
    num_agents = int(np.ceil(traffic_intensity))
    
    while True:
        # Probabilità di blocco (Erlang B)
        erlang_b_prob = traffic_intensity**num_agents / np.math.factorial(num_agents)
        erlang_b_sum = sum(traffic_intensity**i / np.math.factorial(i) for i in range(num_agents + 1))
        
        if erlang_b_sum == 0: # Evita divisione per zero
            return num_agents + 1, 0.0, 1.0

        prob_wait = (erlang_b_prob / erlang_b_sum)
        
        # Probabilità che una chiamata debba attendere (Formula di Erlang C)
        if num_agents > traffic_intensity:
            c_prob = prob_wait / (1 - traffic_intensity / num_agents + prob_wait * traffic_intensity / num_agents)
        else:
            c_prob = 1.0 # Se gli agenti sono meno dell'intensità, tutte le chiamate attendono
        
        # Calcolo del Service Level
        if num_agents > traffic_intensity:
            service_level = 1 - c_prob * np.exp(-(num_agents - traffic_intensity) * answer_time_target / aht)
        else:
            service_level = 0
        
        # Calcolo dell'occupazione
        occupancy = traffic_intensity / num_agents if num_agents > 0 else 0

        # Se il SL non è raggiunto, aggiungi un agente e ricalcola
        if service_level < service_level_target and num_agents < 150: # Limite per evitare loop infiniti
            num_agents += 1
        else:
            return num_agents, service_level, occupancy

def calculate_inbound_staffing(df, aht, sl_target, answer_time, shrinkage, max_occupancy):
    """
    Calcola il fabbisogno di personale per le chiamate inbound.
    """
    results = []
    for _, row in df.iterrows():
        calls = row['Numero di chiamate']
        # Calcola il tasso di arrivo orario basato sulla durata dello slot
        arrival_rate_per_hour = calls * (3600 / row['slot_duration_seconds'])
        
        raw_agents, sl_achieved, occupancy = erlang_c(arrival_rate_per_hour, aht, sl_target, answer_time)
        
        # Applica lo shrinkage
        agents_with_shrinkage = raw_agents / (1 - shrinkage)
        
        # Adeguamento per la massima occupazione
        if occupancy > max_occupancy and max_occupancy > 0:
            required_for_occupancy = (arrival_rate_per_hour * aht / 3600) / max_occupancy
            agents_with_shrinkage = max(agents_with_shrinkage, required_for_occupancy / (1-shrinkage))

        results.append({
            'Data': row['Data'],
            'Giorno': row['Giorno'], # <-- Aggiunta la colonna mancante
            'Time slot': row['Time slot'],
            'Numero di chiamate': calls,
            'Operatori necessari (raw)': raw_agents,
            'Operatori necessari (con shrinkage)': np.ceil(agents_with_shrinkage),
            'Service Level Stimato': sl_achieved,
            'Occupazione Stimata': occupancy
        })
    return pd.DataFrame(results)


def calculate_outbound_staffing(df, cph, shrinkage):
    """
    Calcola il fabbisogno di personale per le chiamate outbound.
    """
    df['Operatori necessari'] = df['Numero di chiamate'] / cph
    df['Operatori necessari (con shrinkage)'] = np.ceil(df['Operatori necessari'] / (1 - shrinkage))
    return df


# --- Funzioni di Supporto e UI ---

def get_default_data():
    """
    Genera un DataFrame di esempio.
    """
    dates = pd.to_datetime(pd.date_range(start="2023-01-02", periods=7, freq='D'))
    time_slots = pd.to_datetime(pd.date_range(start="08:00", end="19:00", freq="30min")).strftime('%H:%M')
    
    data = []
    for date in dates:
        for slot in time_slots:
            # Modello di chiamate: picco a metà mattina e metà pomeriggio, meno la sera
            hour = int(slot.split(':')[0])
            base_calls = 50
            # Curva sinusoidale per simulare un andamento realistico
            multiplier = (np.sin((hour - 8) * np.pi / 11) + 0.5) * 1.5 
            if date.weekday() >= 5: # Weekend
                multiplier *= 0.4
            
            calls = int(base_calls * multiplier + np.random.randint(-10, 10))
            data.append([date.strftime('%Y-%m-%d'), slot, max(0, calls)])
            
    df = pd.DataFrame(data, columns=['Data', 'Time slot', 'Numero di chiamate'])
    return df

# --- Sidebar ---
with st.sidebar:
    st.header("1. Dati di Input")
    
    use_default_data = st.checkbox("Usa dati di esempio", value=True)
    
    if use_default_data:
        df = get_default_data()
        st.success("Dati di esempio caricati.")
    else:
        uploaded_file = st.file_uploader("Carica un file CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            df = None

    if df is not None:
        st.header("2. Parametri del Modello")
        
        call_type = st.selectbox("Tipo di Chiamate", ["Inbound", "Outbound"])
        
        slot_duration = st.number_input("Durata slot (minuti)", min_value=1, value=30)
        
        if call_type == "Inbound":
            st.subheader("Parametri Inbound (Erlang C)")
            aht = st.slider("Average Handle Time (AHT) in secondi", 100, 600, 300)
            sl_target = st.slider("Service Level Target (%)", 50, 100, 80) / 100.0
            answer_time = st.slider("Tempo di risposta target (secondi)", 5, 60, 20)
            max_occupancy = st.slider("Massima Occupazione Agente (%)", 50, 100, 85) / 100.0
            shrinkage = st.slider("Shrinkage (%)", 0, 50, 30, help="Percentuale di tempo in cui un agente non è disponibile (pause, formazione, etc.)") / 100.0
        
        elif call_type == "Outbound":
            st.subheader("Parametri Outbound")
            cph = st.slider("Contatti per Ora (CPH) per agente", 1, 100, 20)
            shrinkage = st.slider("Shrinkage (%)", 0, 50, 30) / 100.0

        run_button = st.button("🚀 Calcola Fabbisogno")

# --- Main App ---
if 'run_button' in locals() and run_button and df is not None:
    
    # --- Pre-processing dei dati ---
    try:
        df['Data'] = pd.to_datetime(df['Data'])
        df['Giorno'] = df['Data'].dt.day_name()
        df['slot_duration_seconds'] = slot_duration * 60
    except Exception as e:
        st.error(f"Errore nella conversione delle colonne. Assicurati che il CSV abbia le colonne 'Data' e 'Time slot'. Dettaglio: {e}")
        st.stop()

    st.markdown("### Dati di Input Aggregati per Giorno e Ora")
    
    # Aggrega i dati per calcolare la media delle chiamate per ogni slot
    agg_df = df.groupby(['Giorno', 'Time slot'])['Numero di chiamate'].mean().reset_index()
    agg_df['Numero di chiamate'] = agg_df['Numero di chiamate'].round().astype(int)
    
    # Ordina i giorni della settimana
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    agg_df['Giorno'] = pd.Categorical(agg_df['Giorno'], categories=days_order, ordered=True)
    agg_df = agg_df.sort_values(['Giorno', 'Time slot'])
    agg_df['Data'] = pd.to_datetime(agg_df['Giorno'], format='%A') # Aggiungi colonna data per compatibilità
    agg_df['slot_duration_seconds'] = slot_duration * 60


    st.dataframe(agg_df.head())

    # --- Calcolo del Fabbisogno ---
    if call_type == "Inbound":
        results_df = calculate_inbound_staffing(agg_df, aht, sl_target, answer_time, shrinkage, max_occupancy)
        required_col = 'Operatori necessari (con shrinkage)'
    else: # Outbound
        # Per l'outbound, il calcolo è più semplice e non richiede aggregazione complessa
        outbound_df = df.copy()
        outbound_df['slot_duration_seconds'] = slot_duration * 60
        results_df = calculate_outbound_staffing(outbound_df, cph, shrinkage)
        required_col = 'Operatori necessari (con shrinkage)'
        # Aggrega i risultati per la visualizzazione
        agg_df = results_df.groupby(['Giorno', 'Time slot'])[required_col].mean().reset_index()
        agg_df[required_col] = agg_df[required_col].round().astype(int)


    st.markdown("---")
    st.markdown("### Risultati del Capacity Sizing")

    # --- Visualizzazioni ---
    
    # 1. Heatmap
    st.markdown("#### Heatmap Fabbisogno Operatori")
    
    try:
        # Usa il dataframe aggregato corretto
        df_for_heatmap = agg_df if call_type == 'Outbound' else results_df
        heatmap_data = df_for_heatmap.pivot_table(values=required_col, index='Time slot', columns='Giorno')
        heatmap_data = heatmap_data.reindex(columns=days_order).dropna(how='all', axis=1)

        fig_heatmap = px.imshow(
            heatmap_data,
            labels=dict(x="Giorno della Settimana", y="Fascia Oraria", color="Operatori"),
            x=heatmap_data.columns,
            y=heatmap_data.index,
            text_auto=True,
            aspect="auto",
            color_continuous_scale=px.colors.sequential.Viridis
        )
        fig_heatmap.update_layout(title="Fabbisogno di Operatori per Giorno e Ora")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    except Exception as e:
        st.warning(f"Non è stato possibile generare la heatmap. Dettaglio: {e}")


    # 2. Grafico a barre per giorno
    st.markdown("#### Fabbisogno Totale per Giorno")
    daily_needs = results_df.groupby('Giorno')[required_col].sum().reindex(days_order).dropna()
    fig_daily = px.bar(
        daily_needs,
        x=daily_needs.index,
        y=daily_needs.values,
        labels={'x': 'Giorno della Settimana', 'y': 'Totale Operatori-Ora'},
        title="Fabbisogno Totale di Operatori-Ora per Giorno"
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # 3. Grafico a linee per fascia oraria
    st.markdown("#### Fabbisogno Medio per Fascia Oraria")
    hourly_needs = results_df.groupby('Time slot')[required_col].mean().reset_index()
    fig_hourly = px.line(
        hourly_needs,
        x='Time slot',
        y=required_col,
        labels={'Time slot': 'Fascia Oraria', required_col: 'Fabbisogno Medio Operatori'},
        title="Fabbisogno Medio di Operatori per Fascia Oraria"
    )
    fig_hourly.update_traces(mode='lines+markers')
    st.plotly_chart(fig_hourly, use_container_width=True)

    # 4. Tabella dei risultati
    with st.expander("Mostra tabella dettagliata dei risultati"):
        st.dataframe(results_df.style.format({
            "Service Level Stimato": "{:.2%}",
            "Occupazione Stimata": "{:.2%}"
        }))

else:
    st.info("Configura i parametri nella sidebar e clicca su 'Calcola Fabbisogno' per visualizzare i risultati.")

