import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(layout="wide")

st.title("✅ Adherence Monitoring")
st.markdown("Questo strumento analizza l'aderenza degli agenti confrontando la pianificazione dei turni con i log di stato effettivi.")

# --- Funzioni di Supporto e UI ---

@st.cache_data
def get_default_schedule():
    """Genera un DataFrame di esempio per la pianificazione."""
    agents = [f"Agente {i+1}" for i in range(5)]
    days = ["2023-01-23"] * 5 # Un solo giorno per semplicità
    shifts = [
        ("09:00", "17:00"), ("09:00", "13:00"), ("13:00", "17:00"),
        ("09:00", "17:00"), ("10:00", "14:00")
    ]
    data = []
    for i, agent in enumerate(agents):
        start = f"{days[i]} {shifts[i][0]}"
        end = f"{days[i]} {shifts[i][1]}"
        data.append({'Agent': agent, 'ShiftStart': pd.to_datetime(start), 'ShiftEnd': pd.to_datetime(end)})
    return pd.DataFrame(data)

@st.cache_data
def get_default_activity_logs():
    """Genera un DataFrame di esempio per i log di attività."""
    schedule = get_default_schedule()
    logs = []
    statuses = ['On Call', 'Available', 'Break', 'On Call', 'Available', 'Away']
    for _, row in schedule.iterrows():
        current_time = row['ShiftStart'] - timedelta(minutes=np.random.randint(5, 15))
        end_time = row['ShiftEnd'] + timedelta(minutes=np.random.randint(5, 30))
        while current_time < end_time:
            status = np.random.choice(statuses, p=[0.4, 0.3, 0.15, 0.1, 0.04, 0.01])
            # Simula una non aderenza
            if row['Agent'] == 'Agente 2' and pd.to_datetime("11:00").time() <= current_time.time() < pd.to_datetime("11:30").time():
                status = 'Away'
            logs.append({'Agent': row['Agent'], 'Timestamp': current_time, 'Status': status})
            current_time += timedelta(minutes=np.random.randint(1, 10))
    return pd.DataFrame(logs)

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def calculate_adherence(schedule_df, activity_df):
    """Calcola l'aderenza per ogni agente."""
    activity_df['Timestamp'] = pd.to_datetime(activity_df['Timestamp'])
    schedule_df['ShiftStart'] = pd.to_datetime(schedule_df['ShiftStart'])
    schedule_df['ShiftEnd'] = pd.to_datetime(schedule_df['ShiftEnd'])

    # Calcola la durata di ogni stato
    activity_df = activity_df.sort_values(['Agent', 'Timestamp'])
    activity_df['Duration'] = activity_df.groupby('Agent')['Timestamp'].diff().dt.total_seconds().fillna(0)

    results = []
    for agent, group in activity_df.groupby('Agent'):
        agent_schedule = schedule_df[schedule_df['Agent'] == agent]
        if agent_schedule.empty:
            continue

        total_scheduled_time = (agent_schedule['ShiftEnd'].iloc[0] - agent_schedule['ShiftStart'].iloc[0]).total_seconds()
        adherent_time = 0

        for _, log in group.iterrows():
            is_scheduled = (log['Timestamp'] >= agent_schedule['ShiftStart'].iloc[0]) and \
                           (log['Timestamp'] < agent_schedule['ShiftEnd'].iloc[0])
            is_productive = log['Status'] in ['On Call', 'Available']

            if is_scheduled and is_productive:
                adherent_time += log['Duration']
        
        adherence_percentage = (adherent_time / total_scheduled_time) * 100 if total_scheduled_time > 0 else 100
        results.append({
            'Agent': agent,
            'Adherence (%)': adherence_percentage,
            'Adherent Time (min)': adherent_time / 60,
            'Scheduled Time (min)': total_scheduled_time / 60
        })

    return pd.DataFrame(results)

# --- Sidebar ---
with st.sidebar:
    st.header("1. Dati di Input")
    use_default_data = st.checkbox("Usa dati di esempio", value=True, help="Usa dati pre-caricati per vedere subito come funziona lo strumento.")

    schedule_df, activity_df = None, None

    if use_default_data:
        schedule_df = get_default_schedule()
        activity_df = get_default_activity_logs()
        st.info("ℹ️ Dati di esempio caricati.")
    else:
        st.subheader("Pianificazione Turni")
        schedule_file = st.file_uploader("Carica CSV Pianificazione", type=["csv"], help="Deve contenere: Agent, ShiftStart (YYYY-MM-DD HH:MM), ShiftEnd (YYYY-MM-DD HH:MM)")
        if schedule_file:
            schedule_df = pd.read_csv(schedule_file)

        st.subheader("Log di Attività Agenti")
        activity_file = st.file_uploader("Carica CSV Log Attività", type=["csv"], help="Deve contenere: Agent, Timestamp (YYYY-MM-DD HH:MM), Status")
        if activity_file:
            activity_df = pd.read_csv(activity_file)

    if schedule_df is not None and activity_df is not None:
        run_button = st.button("🚀 Calcola Aderenza")

# --- Main App ---
if 'run_button' in locals() and run_button and schedule_df is not None and activity_df is not None:
    with st.spinner("Calcolo dell'aderenza in corso..."):
        adherence_results = calculate_adherence(schedule_df, activity_df)

    st.markdown("---")
    st.markdown("### 📊 Risultati di Aderenza")

    # KPI Principale
    overall_adherence = adherence_results['Adherence (%)'].mean()
    st.metric("Aderenza Media Complessiva", f"{overall_adherence:.2f}%")

    tab1, tab2, tab3 = st.tabs(["Grafici di Aderenza", "Timeline Agente", "Tabella Dettagliata"])

    with tab1:
        st.markdown("#### Aderenza per Agente")
        fig_bar = px.bar(
            adherence_results.sort_values('Adherence (%)'),
            x='Adherence (%)', y='Agent', orientation='h',
            title="Percentuale di Aderenza per Agente",
            text=adherence_results['Adherence (%)'].apply(lambda x: f'{x:.1f}%')
        )
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.markdown("#### Analisi Timeline Individuale")
        st.info("Seleziona un agente per visualizzare il confronto tra il suo turno pianificato e le attività svolte.")
        selected_agent = st.selectbox("Seleziona un Agente", options=schedule_df['Agent'].unique())

        agent_schedule = schedule_df[schedule_df['Agent'] == selected_agent]
        agent_activity = activity_df[activity_df['Agent'] == selected_agent].sort_values('Timestamp')

        fig_gantt = go.Figure()

        # Aggiungi turno pianificato
        fig_gantt.add_trace(go.Bar(
            y=[selected_agent],
            base=[agent_schedule['ShiftStart'].iloc[0]],
            x=[(agent_schedule['ShiftEnd'].iloc[0] - agent_schedule['ShiftStart'].iloc[0]).total_seconds()],
            name="Turno Pianificato",
            orientation='h',
            marker_color='#1f77b4', opacity=0.3
        ))

        # Aggiungi stati di attività
        color_map = {'On Call': '#2ca02c', 'Available': '#98df8a', 'Break': '#ff7f0e', 'Away': '#d62728'}
        for i, log in agent_activity.iterrows():
            next_ts = agent_activity['Timestamp'].shift(-1).iloc[i]
            if pd.isna(next_ts):
                next_ts = log['Timestamp'] + timedelta(minutes=5) # stima durata ultimo stato
            
            fig_gantt.add_trace(go.Bar(
                y=[selected_agent],
                base=[log['Timestamp']],
                x=[(next_ts - log['Timestamp']).total_seconds()],
                name=log['Status'],
                orientation='h',
                marker_color=color_map.get(log['Status'], 'grey')
            ))

        fig_gantt.update_layout(title=f"Timeline Attività vs. Pianificazione per {selected_agent}", barmode='stack', yaxis_title="", xaxis_title="Ora del giorno")
        st.plotly_chart(fig_gantt, use_container_width=True)

    with tab3:
        st.markdown("#### Dati di Aderenza Dettagliati")
        st.dataframe(adherence_results)
        csv = convert_df_to_csv(adherence_results)
        st.download_button(
            label="📥 Scarica Risultati in CSV",
            data=csv,
            file_name=f'adherence_results_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.csv',
            mime='text/csv',
        )

elif not ('run_button' in locals()):
    st.info("☝️ Per iniziare, carica i file di pianificazione e attività o usa i dati di esempio, poi clicca 'Calcola Aderenza' nella sidebar.")
