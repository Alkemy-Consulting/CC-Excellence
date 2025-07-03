import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ortools.sat.python import cp_model

st.set_page_config(layout="wide")

st.title("👥 Workforce Management & Shift Planning")
st.markdown("Questo strumento utilizza la programmazione a vincoli (Constraint Programming) per generare una pianificazione dei turni ottimale che soddisfi il fabbisogno di personale.")

# --- Funzioni di Supporto e UI ---

def get_default_requirements():
    """Genera un DataFrame di esempio con il fabbisogno di operatori."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    time_slots = pd.to_datetime(pd.date_range(start="08:00", end="20:00", freq="60min")).strftime('%H:%M')
    
    data = []
    for day in days:
        for slot in time_slots:
            hour = int(slot.split(':')[0])
            base_req = 15  # Ridotto per renderlo più risolvibile di default
            multiplier = (np.sin((hour - 8) * np.pi / 12) + 0.8) * 1.2
            if day in ["Saturday", "Sunday"]:
                multiplier *= 0.5
            
            requirement = int(base_req * multiplier + np.random.randint(-2, 2))
            data.append([day, slot, max(3, requirement)])
            
    df = pd.DataFrame(data, columns=['Giorno', 'Time slot', 'Operatori necessari'])
    return df

def solve_shift_scheduling(requirements_df, shift_config, constraints, optimization_mode):
    """Risolve il problema di scheduling con CP-SAT."""
    model = cp_model.CpModel()

    # Dati del problema
    num_agents = constraints['total_agents']
    days = requirements_df['Giorno'].unique()
    slots = requirements_df['Time slot'].unique()
    
    shifts = {}
    for a in range(num_agents):
        for d_idx, d in enumerate(days):
            for s_idx, s_info in enumerate(shift_config):
                shifts[(a, d_idx, s_idx)] = model.NewBoolVar(f'shift_a{a}_d{d_idx}_s{s_idx}')

    # Vincolo 1: Copertura del fabbisogno per ogni slot
    shortfall_vars = [] # Variabili per la modalità flessibile
    for d_idx, d in enumerate(days):
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
        for d_idx, d in enumerate(days):
            model.Add(sum(shifts[(a, d_idx, s_idx)] for s_idx in range(len(shift_config))) <= 1)

    # Vincolo 3: Giorni di lavoro settimanali per agente
    for a in range(num_agents):
        total_shifts_per_agent = sum(shifts[(a, d_idx, s_idx)] for d_idx in range(len(days)) for s_idx in range(len(shift_config)))
        model.Add(total_shifts_per_agent >= constraints['min_days_per_week'])
        model.Add(total_shifts_per_agent <= constraints['max_days_per_week'])

    # Funzione Obiettivo
    if optimization_mode == "Stretta":
        model.Minimize(sum(shifts.values()))
    else: # Modalità Flessibile: minimizza prima il deficit, poi i turni
        total_shortfall = sum(shortfall_vars)
        total_assigned_shifts = sum(shifts.values())
        # Diamo una penalità molto alta al deficit per minimizzarlo come priorità
        model.Minimize(total_shortfall * 1000 + total_assigned_shifts)

    # Risoluzione
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    # Estrazione dei risultati
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = []
        for a in range(num_agents):
            for d_idx, d in enumerate(days):
                for s_idx, s_info in enumerate(shift_config):
                    if solver.Value(shifts[(a, d_idx, s_idx)]) == 1:
                        start_time = f"2023-01-{d_idx+2} {s_info['start']}"
                        finish_time_str = (pd.to_datetime(s_info['start']) + pd.Timedelta(hours=s_info['length'])).strftime('%H:%M')
                        end_time = f"2023-01-{d_idx+2} {finish_time_str}"
                        schedule.append(dict(Agent=f"Agente {a+1}", Day=d, Shift=s_info['name'], Start=start_time, Finish=end_time))
        return pd.DataFrame(schedule), solver.ObjectiveValue()
    else:
        return None, None

# --- Sidebar ---
with st.sidebar:
    st.header("1. Fabbisogno di Personale")
    use_default_reqs = st.checkbox("Usa fabbisogno di esempio", value=True)

    if use_default_reqs:
        req_df = get_default_requirements()
        st.success("Fabbisogno di esempio caricato.")
    else:
        uploaded_file = st.file_uploader("Carica CSV con fabbisogno", type=["csv"])
        if uploaded_file:
            req_df = pd.read_csv(uploaded_file)
        else:
            req_df = None

    if req_df is not None:
        st.header("2. Configurazione Turni")
        
        st.write("Definisci i tipi di turno disponibili.")
        shift_8h = st.checkbox("Turno 8 ore (08:00-16:00)", value=True)
        shift_8h_mid = st.checkbox("Turno 8 ore (10:00-18:00)", value=True)
        shift_8h_late = st.checkbox("Turno 8 ore (12:00-20:00)", value=True) # Attivato di default
        shift_4h = st.checkbox("Turno 4 ore (08:00-12:00)", value=False)

        shift_config = []
        if shift_8h: shift_config.append({"name": "08-16", "start": "08:00", "length": 8})
        if shift_8h_mid: shift_config.append({"name": "10-18", "start": "10:00", "length": 8})
        if shift_8h_late: shift_config.append({"name": "12-20", "start": "12:00", "length": 8})
        if shift_4h: shift_config.append({"name": "08-12", "start": "08:00", "length": 4})

        st.header("3. Vincoli e Ottimizzazione")
        optimization_mode = st.radio(
            "Modalità di Ottimizzazione",
            ["Stretta", "Flessibile"],
            index=1, # Flessibile di default
            help="Stretta: il fabbisogno deve essere sempre soddisfatto. Flessibile: cerca la migliore copertura possibile anche se il fabbisogno non può essere raggiunto."
        )
        total_agents = st.slider("Numero totale di agenti disponibili", 10, 100, 40) # Aumentato di default
        min_days = st.slider("Min giorni lavorativi/settimana", 3, 5, 4)
        max_days = st.slider("Max giorni lavorativi/settimana", 4, 6, 5)

        constraints = {
            "total_agents": total_agents,
            "min_days_per_week": min_days,
            "max_days_per_week": max_days
        }

        run_button = st.button("🚀 Genera Pianificazione")

# --- Main App ---
if 'run_button' in locals() and run_button and req_df is not None:
    if not shift_config:
        st.error("Seleziona almeno un tipo di turno.")
        st.stop()

    with st.spinner("Il modello di ottimizzazione è al lavoro... Potrebbe richiedere fino a 30 secondi."):
        schedule_df, objective_value = solve_shift_scheduling(req_df, shift_config, constraints, optimization_mode)

    st.markdown("---")
    if schedule_df is not None:
        st.success(f"Pianificazione trovata!")
        
        st.markdown("### Gantt Chart della Pianificazione")
        if not schedule_df.empty:
            schedule_df['Agent'] = pd.Categorical(schedule_df['Agent'], categories=sorted(schedule_df['Agent'].unique(), key=lambda x: int(x.split(' ')[1])), ordered=True)
            
            fig_gantt = px.timeline(
                schedule_df,
                x_start="Start", 
                x_end="Finish", 
                y="Agent", 
                color="Shift",
                labels={"Agent": "Agente", "Day": "Giorno"},
                title="Pianificazione Turni Settimanale"
            )
            fig_gantt.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_gantt, use_container_width=True)
        else:
            st.warning("Nessun turno assegnato con i vincoli correnti.")

        # Grafico di copertura
        st.markdown("### Copertura vs. Fabbisogno")
        coverage_data = []
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        # Assicura che i giorni siano ordinati
        req_df['Giorno'] = pd.Categorical(req_df['Giorno'], categories=days_order, ordered=True)
        req_df = req_df.sort_values(['Giorno', 'Time slot'])

        for index, row_req in req_df.iterrows():
            d = row_req['Giorno']
            t = row_req['Time slot']
            required = row_req['Operatori necessari']
            
            slot_hour = int(t.split(':')[0])
            count = 0
            if not schedule_df.empty:
                scheduled_agents_day = schedule_df[schedule_df['Day'] == d]
                for _, row_sched in scheduled_agents_day.iterrows():
                    start_h = pd.to_datetime(row_sched['Start']).hour
                    finish_h = pd.to_datetime(row_sched['Finish']).hour
                    if start_h <= slot_hour < finish_h:
                        count += 1
            coverage_data.append({'Giorno': d, 'Slot': t, 'Fabbisogno': required, 'Pianificato': count})
        
        coverage_df = pd.DataFrame(coverage_data)
        coverage_df['OraCompleta'] = coverage_df['Giorno'] + " " + coverage_df['Slot']

        fig_coverage = go.Figure()
        fig_coverage.add_trace(go.Scatter(x=coverage_df['OraCompleta'], y=coverage_df['Fabbisogno'], mode='lines+markers', name='Fabbisogno', line=dict(shape='spline')))
        fig_coverage.add_trace(go.Bar(x=coverage_df['OraCompleta'], y=coverage_df['Pianificato'], name='Pianificato', opacity=0.7))
        fig_coverage.update_layout(title='Confronto tra Fabbisogno e Copertura Pianificata', xaxis_title='Giorno e Ora', yaxis_title='Numero di Operatori')
        st.plotly_chart(fig_coverage, use_container_width=True)

        with st.expander("Mostra tabella di pianificazione dettagliata"):
            st.dataframe(schedule_df)

    else:
        st.error("Non è stata trovata una soluzione. Prova a rilassare i vincoli (es. aumentando il numero di agenti disponibili o aggiungendo più tipi di turno).")

else:
    st.info("Configura i parametri nella sidebar e clicca su 'Genera Pianificazione' per avviare l'ottimizzazione.")

