import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson

st.set_page_config(layout="wide")

st.title("🧮 Capacity Sizing Tool")
st.markdown("Questo strumento calcola il numero di operatori necessari (fabbisogno) per gestire i volumi di chiamate, sia in entrata (inbound) che in uscita (outbound), in base ai KPI e ai costi operativi.")

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

def calculate_inbound_staffing(df, aht, sl_target, answer_time, shrinkage, max_occupancy, cost_per_hour):
    """
    Calcola il fabbisogno di personale per le chiamate inbound.
    """
    results = []
    for _, row in df.iterrows():
        calls = row['Numero di chiamate']
        slot_duration_hours = row['slot_duration_seconds'] / 3600
        arrival_rate_per_hour = calls / slot_duration_hours if slot_duration_hours > 0 else 0
        
        raw_agents, sl_achieved, occupancy = erlang_c(arrival_rate_per_hour, aht, sl_target, answer_time)
        
        agents_with_shrinkage = raw_agents / (1 - shrinkage) if (1 - shrinkage) > 0 else raw_agents
        
        if occupancy > max_occupancy and max_occupancy > 0:
            required_for_occupancy = (arrival_rate_per_hour * aht / 3600) / max_occupancy
            agents_with_shrinkage = max(agents_with_shrinkage, required_for_occupancy / (1 - shrinkage))

        final_agents = np.ceil(agents_with_shrinkage)
        total_cost = final_agents * cost_per_hour * slot_duration_hours

        results.append({
            'Data': row['Data'],
            'Giorno': row['Giorno'],
            'Time slot': row['Time slot'],
            'Numero di chiamate': calls,
            'Operatori necessari (raw)': raw_agents,
            'Operatori necessari (con shrinkage)': final_agents,
            'Service Level Stimato': sl_achieved,
            'Occupazione Stimata': occupancy,
            'Costo Stimato (€)': total_cost
        })
    return pd.DataFrame(results)

def calculate_outbound_staffing(df, cph, shrinkage, cost_per_hour):
    """
    Calcola il fabbisogno di personale per le chiamate outbound.
    """
    slot_duration_hours = df['slot_duration_seconds'] / 3600
    df['Operatori necessari (raw)'] = (df['Numero di chiamate'] / cph) 
    df['Operatori necessari (con shrinkage)'] = np.ceil(df['Operatori necessari (raw)'] / (1 - shrinkage))
    df['Costo Stimato (€)'] = df['Operatori necessari (con shrinkage)'] * cost_per_hour * slot_duration_hours
    return df

# --- Funzioni di Supporto e UI ---

def get_default_data():
    """
    Genera un DataFrame di esempio più realistico.
    """
    dates = pd.to_datetime(pd.date_range(start="2023-01-02", periods=14, freq='D'))
    time_slots = pd.to_datetime(pd.date_range(start="08:00", end="20:00", freq="30min")).strftime('%H:%M')
    
    data = []
    for date in dates:
        for slot in time_slots:
            hour = int(slot.split(':')[0])
            base_calls = 60
            # Curva con due picchi (11:00 e 16:00)
            peak1 = np.exp(-((hour - 11)**2) / 4) 
            peak2 = np.exp(-((hour - 16)**2) / 5)
            multiplier = (peak1 + peak2 * 0.8) * 1.8
            
            if date.weekday() == 5: # Sabato
                multiplier *= 0.5
            elif date.weekday() == 6: # Domenica
                multiplier *= 0.2
            
            calls = int(base_calls * multiplier + np.random.randint(-15, 15))
            data.append([date.strftime('%Y-%m-%d'), slot, max(0, calls)])
            
    df = pd.DataFrame(data, columns=['Data', 'Time slot', 'Numero di chiamate'])
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- Sidebar ---
with st.sidebar:
    st.header("1. Dati di Input")
    
    use_default_data = st.checkbox("Usa dati di esempio", value=True, help="Seleziona per usare un dataset pre-caricato con un andamento di chiamate realistico. Deseleziona per caricare il tuo file.")
    
    df = None
    if use_default_data:
        df = get_default_data()
        st.info("ℹ️ Dati di esempio caricati.")
    else:
        uploaded_file = st.file_uploader("Carica un file CSV", type=["csv"], help="Il file deve contenere le colonne 'Data', 'Time slot' (formato HH:MM), e 'Numero di chiamate'.")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                # Validazione colonne
                required_cols = ['Data', 'Time slot', 'Numero di chiamate']
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Il file CSV deve contenere le colonne: {required_cols}")
                    df = None
            except Exception as e:
                st.error(f"Errore nel caricamento del file: {e}")
                df = None

    if df is not None and not df.empty:
        st.header("2. Parametri del Modello")
        
        call_type = st.selectbox("Tipo di Chiamate", ["Inbound", "Outbound"], help="**Inbound**: per chiamate ricevute (es. customer service). **Outbound**: per chiamate effettuate (es. telemarketing).")
        
        slot_duration = st.number_input("Durata slot (minuti)", min_value=1, value=30, help="La durata in minuti di ogni intervallo di tempo nel dataset (es. 30 minuti).")
        cost_per_hour = st.number_input("Costo orario per operatore (€)", min_value=0.0, value=20.0, step=0.5, help="Il costo lordo orario di un singolo operatore. Usato per stimare i costi totali.")

        if call_type == "Inbound":
            st.subheader("Parametri Inbound (Erlang C)")
            aht = st.slider("Average Handle Time (AHT) in secondi", 100, 800, 320, help="Il tempo medio totale per gestire una chiamata, inclusi conversazione e post-chiamata.")
            sl_target = st.slider("Service Level Target (%)", 50, 99, 80, help="L'obiettivo percentuale di chiamate a cui rispondere entro il tempo target (es. 80% delle chiamate in 20 secondi).") / 100.0
            answer_time = st.slider("Tempo di risposta target (secondi)", 5, 60, 20, help="Il tempo massimo in secondi entro cui si dovrebbe rispondere per rispettare il Service Level.")
            max_occupancy = st.slider("Massima Occupazione Agente (%)", 50, 100, 85, help="La percentuale massima di tempo che un operatore dovrebbe dedicare alla gestione delle chiamate per evitare burnout.") / 100.0
            shrinkage = st.slider("Shrinkage (%)", 0, 50, 30, help="La percentuale di tempo retribuito in cui un operatore non è disponibile a ricevere chiamate (pause, formazione, riunioni, etc.).") / 100.0
        
        elif call_type == "Outbound":
            st.subheader("Parametri Outbound")
            cph = st.slider("Contatti Utili per Ora (CPH)", 1, 100, 15, help="Il numero di chiamate con esito positivo (es. vendita, appuntamento) che un operatore riesce a completare in un'ora.")
            shrinkage = st.slider("Shrinkage (%)", 0, 50, 30, help="La percentuale di tempo retribuito in cui un operatore non è disponibile a effettuare chiamate (pause, formazione, etc.).") / 100.0

        run_button = st.button("🚀 Calcola Fabbisogno")

# --- Main App ---
if 'run_button' in locals() and run_button and df is not None and not df.empty:
    
    with st.spinner("Elaborazione in corso... Attendere prego."):
        # --- Pre-processing dei dati ---
        try:
            df['Data'] = pd.to_datetime(df['Data'])
            df['Giorno'] = df['Data'].dt.day_name()
            df['slot_duration_seconds'] = slot_duration * 60
        except Exception as e:
            st.error(f"Errore nella conversione delle colonne. Assicurati che la colonna 'Data' sia in un formato riconoscibile. Dettaglio: {e}")
            st.stop()

        st.markdown("### 📊 Dati di Input Aggregati")
        st.info("I dati mostrati di seguito rappresentano la media delle chiamate per ogni giorno della settimana e fascia oraria, calcolata a partire dal dataset fornito.")
        
        agg_df = df.groupby(['Giorno', 'Time slot'])['Numero di chiamate'].mean().reset_index()
        agg_df['Numero di chiamate'] = agg_df['Numero di chiamate'].round().astype(int)
        
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        agg_df['Giorno'] = pd.Categorical(agg_df['Giorno'], categories=days_order, ordered=True)
        agg_df = agg_df.sort_values(['Giorno', 'Time slot'])
        agg_df['Data'] = pd.to_datetime(agg_df['Giorno'], format='%A') 
        agg_df['slot_duration_seconds'] = slot_duration * 60

        st.dataframe(agg_df[[ 'Giorno', 'Time slot', 'Numero di chiamate']].head())

        # --- Calcolo del Fabbisogno ---
        if call_type == "Inbound":
            results_df = calculate_inbound_staffing(agg_df, aht, sl_target, answer_time, shrinkage, max_occupancy, cost_per_hour)
            required_col = 'Operatori necessari (con shrinkage)'
            cost_col = 'Costo Stimato (€)'
        else: # Outbound
            results_df = calculate_outbound_staffing(agg_df, cph, shrinkage, cost_per_hour)
            required_col = 'Operatori necessari (con shrinkage)'
            cost_col = 'Costo Stimato (€)'

    st.markdown("---")
    st.markdown("### 📈 Risultati del Capacity Sizing")

    # --- Metriche Chiave ---
    total_agents_needed = results_df[required_col].sum()
    total_cost_estimated = results_df[cost_col].sum()
    avg_occupancy = results_df['Occupazione Stimata'].mean() if 'Occupazione Stimata' in results_df else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Totale Ore-Operatore Settimanali", f"{total_agents_needed:,.0f}")
    col2.metric("Costo Totale Stimato Settimanale", f"€ {total_cost_estimated:,.2f}")
    if call_type == "Inbound":
        col3.metric("Occupazione Media Stimata", f"{avg_occupancy:.1%}")

    # --- Visualizzazioni ---
    tab1, tab2, tab3 = st.tabs(["Heatmap Fabbisogno", "Grafici di Dettaglio", "Tabella Risultati"])

    with tab1:
        st.markdown("#### Heatmap Fabbisogno Operatori")
        st.info("Questa mappa di calore mostra il numero di operatori necessari (inclusa la shrinkage) per ogni fascia oraria e giorno della settimana.")
        try:
            heatmap_data = results_df.pivot_table(values=required_col, index='Time slot', columns='Giorno')
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

    with tab2:
        st.markdown("#### Fabbisogno Totale per Giorno")
        daily_needs = results_df.groupby('Giorno')[required_col].sum().reindex(days_order).dropna()
        fig_daily = px.bar(
            daily_needs,
            x=daily_needs.index,
            y=daily_needs.values,
            labels={'x': 'Giorno della Settimana', 'y': 'Totale Ore-Operatore'},
            title="Fabbisogno Totale di Ore-Operatore per Giorno"
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        st.markdown("#### Fabbisogno Medio per Fascia Oraria")
        hourly_needs = results_df.groupby('Time slot')[required_col].mean().reset_index()
        fig_hourly = px.line(
            hourly_needs,
            x='Time slot',
            y=required_col,
            labels={'Time slot': 'Fascia Oraria', required_col: 'Fabbisogno Medio Operatori'},
            title="Fabbisogno Medio di Operatori per Fascia Oraria (tutti i giorni)"
        )
        fig_hourly.update_traces(mode='lines+markers')
        st.plotly_chart(fig_hourly, use_container_width=True)

    with tab3:
        st.markdown("#### Tabella Dettagliata dei Risultati")
        st.info("Questa tabella mostra i calcoli dettagliati per ogni singolo intervallo di tempo.")
        
        # Formattazione condizionale
        def format_table(df_to_format):
            format_dict = {
                "Costo Stimato (€)": "{:.2f}"
            }
            if 'Service Level Stimato' in df_to_format.columns:
                format_dict["Service Level Stimato"] = "{:.2%}"
            if 'Occupazione Stimata' in df_to_format.columns:
                format_dict["Occupazione Stimata"] = "{:.2%}"
            return df_to_format.style.format(format_dict)

        st.dataframe(format_table(results_df), use_container_width=True)
        
        csv = convert_df_to_csv(results_df)
        st.download_button(
            label="📥 Scarica Risultati in CSV",
            data=csv,
            file_name=f'capacity_sizing_{call_type.lower()}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
        )

elif df is not None and df.empty:
    st.warning("Il DataFrame è vuoto. Controlla il file caricato o i filtri applicati.")
else:
    st.info("☝️ Per iniziare, configura i parametri nella sidebar e clicca su 'Calcola Fabbisogno'.")

