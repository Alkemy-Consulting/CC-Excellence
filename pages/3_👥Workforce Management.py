import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ortools.sat.python import cp_model

st.set_page_config(layout="wide")

st.title("👥 Workforce Management & Shift Planning")
st.markdown("Questo strumento utilizza la **Programmazione a Vincoli (Constraint Programming)** per generare una pianificazione dei turni ottimale che soddisfi il fabbisogno di personale, minimizzando i costi e rispettando i vincoli operativi.")

# --- Funzioni di Supporto e UI ---

@st.cache_data
def get_default_requirements():
    """Genera un DataFrame di esempio con il fabbisogno di operatori."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    time_slots = pd.to_datetime(pd.date_range(start="08:00", end="20:00", freq="60min")).strftime('%H:%M')
    
    data = []
    for day in days:
        for slot in time_slots:
            hour = int(slot.split(':')[0])
            base_req = 18
            # Curva con due picchi (11:00 e 16:00)
            peak1 = np.exp(-((hour - 11)**2) / 4)
            peak2 = np.exp(-((hour - 16)**2) / 5)
            multiplier = (peak1 + peak2 * 0.8) * 1.5

            if day == "Saturday":
                multiplier *= 0.5
            elif day == "Sunday":
                multiplier *= 0.2
            
            requirement = int(base_req * multiplier + np.random.randint(-2, 2))
            data.append([day, slot, max(2, requirement)])
            
    df = pd.DataFrame(data, columns=['Giorno', 'Time slot', 'Operatori necessari'])
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def solve_shift_scheduling(requirements_df, shift_config, constraints, optimization_mode):
    """Risolve il problema di scheduling con CP-SAT."""
    model = cp_model.CpModel()

    # Dati del problema
    num_agents = constraints['total_agents']
    days_map = {day: i for i, day in enumerate(requirements_df['Giorno'].unique())}
    slots = requirements_df['Time slot'].unique()
    num_days = len(days_map)
    
    shifts = {}
    for a in range(num_agents):
        for d_idx in range(num_days):
            for s_idx, s_info in enumerate(shift_config):
                shifts[(a, d_idx, s_idx)] = model.NewBoolVar(f'shift_a{a}_d{d_idx}_s{s_idx}')

    # Vincolo 1: Copertura del fabbisogno
    shortfall_vars = []
    for d_idx, d in enumerate(days_map.keys()):
        for t_idx, t in enumerate(slots):
            required = int(requirements_df[(requirements_df['Giorno'] == d) & (requirements_df['Time slot'] == t)]['Operatori necessari'].iloc[0])
            
            agents_on_duty = []
            for a in range(num_agents):
                for s_idx, s_info in enumerate(shift_config):
                    start_hour = int(s_info['start'].split(':')[0])
                    end_hour = start_hour + s_info['length']
                    slot_hour = int(t.split(':')[0])
                    
                    if start_hour <= slot_hour < end_hour:
                        agents_on_duty.append(shifts[(a, d_idx, s_idx)])
            
            if optimization_mode == "Stretta":
                model.Add(sum(agents_on_duty) >= required)
            else: # Modalità Flessibile
                shortfall = model.NewIntVar(0, required, f'shortfall_d{d_idx}_t{t_idx}')
                model.Add(sum(agents_on_duty) + shortfall >= required)
                shortfall_vars.append(shortfall)

    # Vincolo 2: Ogni agente lavora al massimo un turno al giorno
    for a in range(num_agents):
        for d_idx in range(num_days):
            model.Add(sum(shifts[(a, d_idx, s_idx)] for s_idx in range(len(shift_config))) <= 1)

    # Vincolo 3: Giorni di lavoro settimanali per agente
    for a in range(num_agents):
        total_shifts_per_agent = sum(shifts[(a, d_idx, s_idx)] for d_idx in range(num_days) for s_idx in range(len(shift_config)))
        model.Add(total_shifts_per_agent >= constraints['min_days_per_week'])
        model.Add(total_shifts_per_agent <= constraints['max_days_per_week'])

    # Funzione Obiettivo
    total_assigned_shifts = sum(shifts.values())
    total_hours = sum(s_info['length'] * shifts[(a, d_idx, s_idx)] 
                      for a, d_idx, s_idx in shifts 
                      for s_info in [shift_config[s_idx]])

    if optimization_mode == "Stretta":
        model.Minimize(total_hours) # Minimizza le ore totali (e quindi i costi)
    else: # Modalità Flessibile
        total_shortfall = sum(shortfall_vars)
        model.Minimize(total_shortfall * 1000 + total_hours)

    # Risoluzione
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    # Estrazione dei risultati
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        schedule = []
        day_indices = {i: d for d, i in days_map.items()}
        for a in range(num_agents):
            for d_idx in range(num_days):
                for s_idx, s_info in enumerate(shift_config):
                    if solver.Value(shifts[(a, d_idx, s_idx)]) == 1:
                        day_name = day_indices[d_idx]
                        # Usiamo una data base per il Gantt, il giorno corretto viene dal nome
                        base_date = pd.to_datetime(f"2023-01-{d_idx+2}")
                        start_time = base_date.replace(hour=int(s_info['start'].split(':')[0]), minute=0)
                        end_time = start_time + pd.Timedelta(hours=s_info['length'])
                        schedule.append(dict(
                            Agent=f"Agente {a+1}", 
                            Day=day_name, 
                            Shift=s_info['name'], 
                            Start=start_time.strftime("%Y-%m-%d %H:%M"), 
                            Finish=end_time.strftime("%Y-%m-%d %H:%M"),
                            Hours=s_info['length']
                        ))
        return pd.DataFrame(schedule), solver.StatusName(status)
    else:
        return None, solver.StatusName(status)

# --- Sidebar ---
with st.sidebar:
    st.header("1. Fabbisogno di Personale")
    use_default_reqs = st.checkbox("Usa fabbisogno di esempio", value=True, help="Seleziona per usare un dataset pre-caricato con una curva di fabbisogno realistica. Deseleziona per caricare il tuo file.")

    req_df = None
    if use_default_reqs:
        req_df = get_default_requirements()
        st.info("ℹ️ Fabbisogno di esempio caricato.")
    else:
        uploaded_file = st.file_uploader("Carica CSV con fabbisogno", type=["csv"], help="Il file deve contenere le colonne: 'Giorno', 'Time slot' (HH:MM), 'Operatori necessari'.")
        if uploaded_file:
            try:
                req_df = pd.read_csv(uploaded_file)
                required_cols = ['Giorno', 'Time slot', 'Operatori necessari']
                if not all(col in req_df.columns for col in required_cols):
                    st.error(f"Il file CSV deve contenere le colonne: {required_cols}")
                    req_df = None
            except Exception as e:
                st.error(f"Errore nel caricamento del file: {e}")
                req_df = None

    if req_df is not None and not req_df.empty:
        st.header("2. Configurazione Turni")
        st.info("Definisci i tipi di turno che il modello può assegnare.")
        
        c1, c2 = st.columns(2)
        with c1:
            shift_8h_morning = st.checkbox("Turno 8h (08-16)", value=True)
            shift_8h_mid = st.checkbox("Turno 8h (10-18)", value=True)
            shift_8h_late = st.checkbox("Turno 8h (12-20)", value=True)
        with c2:
            shift_6h_flex = st.checkbox("Turno 6h (flessibile)", value=False)
            shift_4h_part = st.checkbox("Turno 4h (part-time)", value=False)

        shift_config = []
        if shift_8h_morning: shift_config.append({"name": "T_08-16", "start": "08:00", "length": 8})
        if shift_8h_mid: shift_config.append({"name": "T_10-18", "start": "10:00", "length": 8})
        if shift_8h_late: shift_config.append({"name": "T_12-20", "start": "12:00", "length": 8})
        if shift_6h_flex:
            shift_config.append({"name": "T_08-14", "start": "08:00", "length": 6})
            shift_config.append({"name": "T_14-20", "start": "14:00", "length": 6})
        if shift_4h_part:
            shift_config.append({"name": "P_09-13", "start": "09:00", "length": 4})
            shift_config.append({"name": "P_14-18", "start": "14:00", "length": 4})

        st.header("3. Vincoli e Ottimizzazione")
        cost_per_hour = st.number_input("Costo orario per operatore (€)", 10.0, 50.0, 22.0, 0.5, help="Il costo lordo orario di un singolo operatore, usato per calcolare il costo totale della pianificazione.")
        
        optimization_mode = st.radio(
            "Modalità di Ottimizzazione", ["Stretta", "Flessibile"],
            index=1, help="**Stretta**: il fabbisogno deve essere sempre soddisfatto (può fallire se non ci sono abbastanza risorse). **Flessibile**: cerca la migliore copertura possibile, anche se incompleta, minimizzando il deficit."
        )
        total_agents = st.slider("Numero totale di agenti disponibili", 5, 150, 45, help="Il numero massimo di persone che possono essere pianificate.")
        min_days = st.slider("Min giorni lavorativi/settimana", 3, 6, 4, help="Il numero minimo di giorni che un agente deve lavorare in una settimana.")
        max_days = st.slider("Max giorni lavorativi/settimana", 4, 6, 5, help="Il numero massimo di giorni che un agente può lavorare in una settimana.")

        constraints = {
            "total_agents": total_agents,
            "min_days_per_week": min_days,
            "max_days_per_week": max_days
        }

        run_button = st.button("🚀 Genera Pianificazione")

# --- Main App ---
if 'run_button' in locals() and run_button and req_df is not None:
    if not shift_config:
        st.error("Seleziona almeno un tipo di turno dalla sidebar.")
        st.stop()

    with st.spinner("Il modello di ottimizzazione è al lavoro... Potrebbe richiedere fino a 30 secondi."):
        schedule_df, status = solve_shift_scheduling(req_df, shift_config, constraints, optimization_mode)

    st.markdown("---")
    if schedule_df is not None:
        st.success(f"Pianificazione trovata con successo! (Stato: {status})")
        
        # Calcolo Copertura e KPI
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        req_df['Giorno'] = pd.Categorical(req_df['Giorno'], categories=days_order, ordered=True)
        req_df = req_df.sort_values(['Giorno', 'Time slot'])
        
        coverage_data = []
        for _, row_req in req_df.iterrows():
            d, t, required = row_req['Giorno'], row_req['Time slot'], row_req['Operatori necessari']
            slot_hour = int(t.split(':')[0])
            count = 0
            if not schedule_df.empty:
                agents_on_duty = schedule_df[schedule_df['Day'] == d]
                for _, agent_shift in agents_on_duty.iterrows():
                    start_h, end_h = pd.to_datetime(agent_shift['Start']).hour, pd.to_datetime(agent_shift['Finish']).hour
                    if start_h <= slot_hour < (end_h if end_h > start_h else 24):
                        count += 1
            coverage_data.append({'Giorno': d, 'Slot': t, 'Fabbisogno': required, 'Pianificato': count})
        
        coverage_df = pd.DataFrame(coverage_data)
        coverage_df['Deficit/Surplus'] = coverage_df['Pianificato'] - coverage_df['Fabbisogno']
        coverage_df['OraCompleta'] = coverage_df['Giorno'] + " " + coverage_df['Slot']

        # KPI
        total_hours_planned = schedule_df['Hours'].sum() if not schedule_df.empty else 0
        total_cost = total_hours_planned * cost_per_hour
        total_required = coverage_df['Fabbisogno'].sum()
        total_planned = coverage_df['Pianificato'].sum()
        avg_coverage = total_planned / total_required if total_required > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Totale Ore Pianificate", f"{total_hours_planned:,.0f} ore")
        col2.metric("Costo Totale Stimato", f"€ {total_cost:,.2f}")
        col3.metric("Copertura Media Fabbisogno", f"{avg_coverage:.1%}")

        # Visualizzazioni
        tab1, tab2, tab3 = st.tabs(["📊 Grafici di Copertura", "📅 Gantt di Pianificazione", "📋 Tabella Dettagliata"])

        with tab1:
            st.markdown("#### Copertura vs. Fabbisogno")
            st.info("Questo grafico confronta il numero di operatori richiesti (linea) con quelli pianificati (barre) per ogni ora.")
            fig_coverage = go.Figure()
            fig_coverage.add_trace(go.Scatter(x=coverage_df['OraCompleta'], y=coverage_df['Fabbisogno'], name='Fabbisogno', mode='lines+markers', line=dict(shape='spline', color='#ff7f0e')))
            fig_coverage.add_trace(go.Bar(x=coverage_df['OraCompleta'], y=coverage_df['Pianificato'], name='Pianificato', marker_color='#1f77b4'))
            fig_coverage.update_layout(title='Confronto Fabbisogno vs. Copertura Pianificata', xaxis_title='Giorno e Ora', yaxis_title='Numero di Operatori', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_coverage, use_container_width=True)

            st.markdown("#### Deficit e Surplus")
            st.info("Questo grafico mostra dove la pianificazione ha un surplus di personale (barre verdi) o un deficit (barre rosse).")
            colors = ['#2ca02c' if x >= 0 else '#d62728' for x in coverage_df['Deficit/Surplus']]
            fig_surplus = go.Figure(go.Bar(x=coverage_df['OraCompleta'], y=coverage_df['Deficit/Surplus'], marker_color=colors))
            fig_surplus.update_layout(title='Deficit/Surplus di Copertura per Ora', xaxis_title='Giorno e Ora', yaxis_title='Operatori in Eccesso/Mancanti')
            st.plotly_chart(fig_surplus, use_container_width=True)

        with tab2:
            st.markdown("#### Gantt Chart della Pianificazione")
            if not schedule_df.empty:
                schedule_df['Agent'] = pd.Categorical(schedule_df['Agent'], categories=sorted(schedule_df['Agent'].unique(), key=lambda x: int(x.split(' ')[1])), ordered=True)
                fig_gantt = px.timeline(
                    schedule_df.sort_values('Agent'),
                    x_start="Start", x_end="Finish", y="Agent", color="Shift",
                    labels={"Agent": "Agente"}, title="Pianificazione Turni Settimanale per Agente"
                )
                fig_gantt.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_gantt, use_container_width=True)
            else:
                st.warning("Nessun turno assegnato. Il Gantt non può essere visualizzato.")

        with tab3:
            st.markdown("#### Tabella di Pianificazione Dettagliata")
            st.dataframe(schedule_df)
            csv = convert_df_to_csv(schedule_df)
            st.download_button(
                label="📥 Scarica Pianificazione in CSV",
                data=csv,
                file_name=f'shift_plan_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.csv',
                mime='text/csv',
            )

    else:
        st.error(f"**Impossibile trovare una soluzione (Stato: {status}).**\n\n" + 
            "Questo solitamente accade se i vincoli sono troppo stringenti per le risorse disponibili. Prova a:\n" + 
            "- **Aumentare il 'Numero totale di agenti disponibili'.**\n" + 
            "- **Selezionare la modalità di ottimizzazione 'Flessibile'.**\n" + 
            "- **Aggiungere più tipi di turno** (es. part-time o turni più corti).\n" + 
            "- **Rivedere i vincoli** sui giorni lavorativi minimi/massimi.")

elif req_df is not None and req_df.empty:
    st.warning("Il DataFrame del fabbisogno è vuoto. Controlla il file caricato.")
else:
    st.info("☝️ Per iniziare, configura i parametri nella sidebar e clicca su 'Genera Pianificazione'.")

