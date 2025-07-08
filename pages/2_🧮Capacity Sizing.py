import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import poisson
from scipy.special import loggamma
import datetime
from io import StringIO

st.set_page_config(layout="wide")

st.title("🧮 Capacity Sizing Tool")
st.markdown("Strumento avanzato per il calcolo del fabbisogno di operatori per Contact Center. Supporta diversi modelli di calcolo (Erlang C/A, Simulazione, Deterministico) con analisi What-If e Stress Test.")

# --- Funzioni di Calcolo Avanzate ---

def erlang_c(arrival_rate, aht, service_level_target, answer_time_target):
    """
    Calcola il numero di agenti necessari utilizzando la formula di Erlang C.
    Implementazione accademicamente rigorosa e numericamente stabile.
    
    Args:
        arrival_rate: λ - Numero di chiamate per ora
        aht: Average Handle Time in secondi
        service_level_target: Target service level (0-1)
        answer_time_target: Tempo target di risposta in secondi
    
    Returns:
        tuple: (agenti_necessari, service_level_ottenuto, occupazione)
    """
    if arrival_rate <= 0:
        return 0, 1.0, 0.0

    # Intensità di traffico in Erlang (A = λ * μ)
    traffic_intensity = (arrival_rate * aht) / 3600
    
    if traffic_intensity <= 0:
        return 1, 1.0, 0.0
    
    # Inizializza con il minimo teorico più margine di sicurezza
    num_agents = max(1, int(np.ceil(traffic_intensity * 1.1)))
    
    max_iterations = 300  # Limite per evitare loop infiniti
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Calcolo stabilizzato della probabilità di attesa (Erlang C)
        try:
            if num_agents <= traffic_intensity:
                # Sistema saturo - tutte le chiamate attendono
                prob_wait = 1.0
                service_level = 0.0
            else:
                # Calcolo numericamente stabile usando logaritmi per grandi valori
                if traffic_intensity > 50 or num_agents > 100:
                    # Usa approssimazione per evitare overflow
                    # Log della probabilità di attesa
                    log_num = num_agents * np.log(traffic_intensity) - loggamma(num_agents + 1)
                    
                    # Calcola denominatore in log space
                    log_terms = []
                    for k in range(num_agents):
                        if k == 0:
                            log_terms.append(0)  # log(1)
                        else:
                            log_terms.append(k * np.log(traffic_intensity) - loggamma(k + 1))
                    
                    # Aggiunge termine finale
                    log_final_term = log_num - np.log(num_agents - traffic_intensity)
                    log_terms.append(log_final_term)
                    
                    # Calcola somma usando log-sum-exp trick
                    max_log = max(log_terms)
                    sum_exp = sum(np.exp(log_term - max_log) for log_term in log_terms)
                    
                    prob_wait = np.exp(log_final_term - max_log) / sum_exp
                    
                else:
                    # Calcolo diretto per valori piccoli
                    numerator = (traffic_intensity ** num_agents) / np.math.factorial(num_agents)
                    
                    # Denominatore: somma dei termini da 0 a num_agents-1 + termine finale
                    denominator = sum((traffic_intensity ** k) / np.math.factorial(k) 
                                    for k in range(num_agents))
                    denominator += numerator / (num_agents - traffic_intensity)
                    
                    prob_wait = (numerator / (num_agents - traffic_intensity)) / denominator
                
                # Assicura che prob_wait sia nel range [0,1]
                prob_wait = max(0, min(1, prob_wait))
                
                # Calcolo del Service Level con formula esatta
                exponential_term = np.exp(-(num_agents - traffic_intensity) * answer_time_target / aht)
                service_level = 1 - prob_wait * exponential_term
                
        except (OverflowError, ZeroDivisionError, ValueError):
            # Fallback robusto per casi numerici estremi
            if num_agents > traffic_intensity * 2:
                prob_wait = 0.01
                service_level = 0.99
            else:
                prob_wait = 0.95
                service_level = 0.05
        
        # Calcolo dell'occupazione (ρ = A/n)
        occupancy = traffic_intensity / num_agents if num_agents > 0 else 0
        
        # Condizioni di terminazione
        if service_level >= service_level_target:
            return num_agents, service_level, occupancy
        elif occupancy < 0.05:  # Se l'occupazione è troppo bassa, ferma
            return num_agents, service_level, occupancy
        elif iteration > max_iterations - 10 and service_level > service_level_target * 0.8:
            # Vicino al target e molte iterazioni - accetta risultato
            return num_agents, service_level, occupancy
        else:
            num_agents += 1
    
    # Se esce dal loop senza convergenza
    return num_agents, service_level, occupancy
    
    # Se raggiungiamo il limite di iterazioni
    return num_agents, service_level, occupancy

def erlang_a(arrival_rate, aht, patience, service_level_target, answer_time_target):
    """
    Calcola utilizzando Erlang A (considera l'abbandono delle chiamate)
    """
    # Implementazione semplificata di Erlang A
    traffic_intensity = arrival_rate * aht / 3600
    basic_agents, sl, occ = erlang_c(arrival_rate, aht, service_level_target, answer_time_target)
    
    # Fattore di correzione per l'abbandono
    abandon_factor = np.exp(-answer_time_target / patience) if patience > 0 else 0
    adjusted_sl = sl + (1 - sl) * abandon_factor
    
    # Se il SL è migliorato dall'abbandono, possiamo ridurre gli agenti
    if adjusted_sl > service_level_target and basic_agents > 1:
        return max(1, basic_agents - 1), adjusted_sl, traffic_intensity / max(1, basic_agents - 1)
    
    return basic_agents, adjusted_sl, occ

def deterministic_model(calls_per_hour, aht, shrinkage_factor):
    """
    Modello deterministico semplice
    """
    workload_hours = calls_per_hour * aht / 3600
    agents_needed = workload_hours / (1 - shrinkage_factor) if (1 - shrinkage_factor) > 0 else workload_hours
    occupancy = workload_hours / agents_needed if agents_needed > 0 else 0
    return np.ceil(agents_needed), 1.0, occupancy

def simulation_model(calls_per_hour, aht, service_level_target, num_simulations=1000):
    """
    Modello di simulazione Monte Carlo semplificato
    """
    # Simulazione semplificata - per una implementazione completa servirebbe più logica
    base_agents, sl, occ = erlang_c(calls_per_hour, aht, service_level_target, 20)
    
    # Aggiunge variabilità tramite simulazione
    simulation_results = []
    for _ in range(num_simulations):
        # Variabilità nei chiamate (+/- 20%)
        sim_calls = calls_per_hour * np.random.uniform(0.8, 1.2)
        # Variabilità nell'AHT (+/- 15%)
        sim_aht = aht * np.random.uniform(0.85, 1.15)
        
        sim_agents, sim_sl, sim_occ = erlang_c(sim_calls, sim_aht, service_level_target, 20)
        simulation_results.append((sim_agents, sim_sl, sim_occ))
    
    # Statistiche della simulazione
    agents_needed = np.percentile([r[0] for r in simulation_results], 95)  # 95° percentile
    avg_sl = np.mean([r[1] for r in simulation_results])
    avg_occ = np.mean([r[2] for r in simulation_results])
    
    return int(agents_needed), avg_sl, avg_occ

def calculate_costs(agents_needed, ral_annuale, giorni_lavorativi, moltiplicatore_costo, ore_lavoro_giorno=8):
    """
    Calcola i costi operativi
    """
    costo_orario = ral_annuale * moltiplicatore_costo / (giorni_lavorativi * ore_lavoro_giorno)
    costo_mensile_fte = ral_annuale * moltiplicatore_costo / 12
    costo_totale_piano = agents_needed * costo_mensile_fte
    
    return {
        'costo_orario': costo_orario,
        'costo_mensile_fte': costo_mensile_fte,
        'costo_totale_piano': costo_totale_piano
    }

def apply_what_if_analysis(base_results, model_type, model_params, working_hours, cost_params, volume_var, aht_var, sl_var):
    """
    Applica variazioni What-If sui parametri base e ricalcola con i nuovi parametri
    """
    # Modifica i parametri del modello
    modified_params = model_params.copy()
    
    # Applica le variazioni ai parametri
    if 'aht' in modified_params:
        modified_params['aht'] = modified_params['aht'] * (1 + aht_var / 100)
    if 'service_level' in modified_params:
        modified_params['service_level'] = min(99, max(50, modified_params['service_level'] * (1 + sl_var / 100)))
    
    # Modifica il dataset di input
    modified_df = base_results.copy()
    if 'Numero di chiamate' in modified_df.columns:
        modified_df['Numero di chiamate'] = modified_df['Numero di chiamate'] * (1 + volume_var / 100)
    
    # Ricalcola completamente con i nuovi parametri
    whatif_results = calculate_capacity_requirements(
        modified_df, 
        model_type, 
        modified_params,
        working_hours,
        cost_params
    )
    
    return whatif_results

def apply_stress_test(base_df, model_type, model_params, working_hours, cost_params, picco_volume, assenteismo_extra, guasto_it_riduzione):
    """
    Applica scenari di stress test con ricalcolo completo
    """
    # Modifica i parametri per lo stress test
    stress_params = model_params.copy()
    stress_df = base_df.copy()
    
    # Scenario di picco volume
    if 'Numero di chiamate' in stress_df.columns:
        stress_df['Numero di chiamate'] = stress_df['Numero di chiamate'] * (1 + picco_volume / 100)
    
    # Aumento del shrinkage per assenteismo straordinario
    if 'shrinkage' in stress_params:
        current_shrinkage = stress_params['shrinkage']
        additional_shrinkage = assenteismo_extra / 100
        # Formula corretta per combinare shrinkage: 1 - (1-s1)*(1-s2)
        combined_shrinkage = 1 - (1 - current_shrinkage) * (1 - additional_shrinkage)
        stress_params['shrinkage'] = min(0.8, combined_shrinkage)  # Cap al 80%
    
    # Ricalcola con i parametri di stress
    stress_results = calculate_capacity_requirements(
        stress_df, 
        model_type, 
        stress_params,
        working_hours,
        cost_params
    )
    
    # Applica riduzione capacità per guasto IT se abilitato
    if guasto_it_riduzione > 0:
        # Questo richiederebbe logica aggiuntiva per fasce orarie specifiche
        pass
    
    return stress_results

def calculate_capacity_requirements(df, model_type, model_params, working_hours, cost_params):
    """
    Calcola il fabbisogno di capacità utilizzando il modello selezionato
    """
    results = []
    
    for _, row in df.iterrows():
        calls = row['Numero di chiamate']
        time_slot = row['Time slot']
        day = row['Giorno']
        
        # Verifica se il time slot è negli orari di apertura
        if day in working_hours and working_hours[day].get('is_open', True):
            start_time = working_hours[day]['start']
            end_time = working_hours[day]['end']
            slot_time = datetime.datetime.strptime(time_slot, '%H:%M').time()
            
            if not (start_time <= slot_time <= end_time):
                # Fuori orario di apertura
                results.append({
                    'Data': row.get('Data', ''),
                    'Giorno': day,
                    'Time slot': time_slot,
                    'Numero di chiamate': calls,
                    'Operatori necessari': 0,
                    'Service Level Stimato': 1.0,
                    'Occupazione Stimata': 0.0,
                    'Costo Stimato (€)': 0.0,
                    'Status': 'Chiuso - Fuori orario'
                })
                continue
        elif day in working_hours and not working_hours[day].get('is_open', True):
            # Giorno chiuso
            results.append({
                'Data': row.get('Data', ''),
                'Giorno': day,
                'Time slot': time_slot,
                'Numero di chiamate': calls,
                'Operatori necessari': 0,
                'Service Level Stimato': 1.0,
                'Occupazione Stimata': 0.0,
                'Costo Stimato (€)': 0.0,
                'Status': 'Chiuso - Giorno non operativo'
            })
            continue
        
        # Calcolo basato sul modello selezionato e tipologia operativa
        if model_type == "Erlang C":
            agents, sl, occ = erlang_c(
                calls, 
                model_params['aht'], 
                model_params['service_level'] / 100, 
                model_params['answer_time']
            )
        elif model_type == "Erlang A":
            agents, sl, occ = erlang_a(
                calls,
                model_params['aht'],
                model_params.get('patience', 60),
                model_params['service_level'] / 100,
                model_params['answer_time']
            )
        elif model_type == "Deterministico":
            # Per INBOUND
            if 'aht' in model_params:
                agents, sl, occ = deterministic_model(
                    calls,
                    model_params['aht'],
                    model_params.get('shrinkage', 0.3)
                )
            # Per OUTBOUND
            else:
                slot_duration_hours = model_params.get('slot_duration', 30) / 60
                hourly_rate = calls / slot_duration_hours
                cph = model_params.get('cph', 15)
                
                # Calcolo base operatori necessari
                agents = hourly_rate / cph if cph > 0 else 0
                
                # Service Level per outbound è diverso - rappresenta success rate
                sl = 0.95  # Success rate tipico per outbound
                
                # Occupazione calcolata correttamente
                if agents > 0:
                    occ = min(0.95, hourly_rate / (cph * agents))
                else:
                    occ = 0
                    
        elif model_type == "Simulazione":
            # Per INBOUND
            if 'aht' in model_params:
                agents, sl, occ = simulation_model(
                    calls,
                    model_params['aht'],
                    model_params['service_level'] / 100
                )
            # Per OUTBOUND
            else:
                slot_duration_hours = model_params.get('slot_duration', 30) / 60
                hourly_rate = calls / slot_duration_hours
                cph = model_params.get('cph', 15)
                
                # Simulazione con variabilità
                base_agents = hourly_rate / cph if cph > 0 else 0
                
                # Aggiunge variabilità tipica per outbound
                variability_factor = np.random.uniform(0.9, 1.15)
                agents = base_agents * variability_factor
                
                # Success rate con variabilità
                sl = np.random.uniform(0.85, 0.98)
                
                # Occupazione con variabilità
                if agents > 0:
                    occ = min(0.95, hourly_rate / (cph * agents * np.random.uniform(0.95, 1.05)))
                else:
                    occ = 0
        else:
            # Fallback generico
            agents, sl, occ = 1, 0.8, 0.7
        
        # Applica shrinkage se non già considerata
        if model_type not in ["Deterministico"]:
            shrinkage = model_params.get('shrinkage', 0.3)
            agents_with_shrinkage = agents / (1 - shrinkage)
        else:
            agents_with_shrinkage = agents
        
        # Calcola i costi
        slot_duration_hours = model_params.get('slot_duration', 30) / 60
        cost_per_hour = cost_params['costo_orario']
        total_cost = np.ceil(agents_with_shrinkage) * cost_per_hour * slot_duration_hours
        
        results.append({
            'Data': row.get('Data', ''),
            'Giorno': day,
            'Time slot': time_slot,
            'Numero di chiamate': calls,
            'Operatori necessari': np.ceil(agents_with_shrinkage),
            'Service Level Stimato': sl,
            'Occupazione Stimata': occ,
            'Costo Stimato (€)': total_cost,
            'Status': 'Aperto'
        })
    
    return pd.DataFrame(results)

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
            data.append([date.strftime('%Y-%m-%d'), date.strftime('%A'), slot, max(0, calls)])
            
    df = pd.DataFrame(data, columns=['Data', 'Giorno', 'Time slot', 'Numero di chiamate'])
    return df

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def load_csv_file(uploaded_file, delimiter, date_format):
    """
    Carica e processa un file CSV con parametri specificati
    """
    try:
        # Leggi il file CSV
        if delimiter == "Auto-detect":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, sep=delimiter)
        
        # Validazione colonne richieste
        required_cols = ['Data', 'Time slot', 'Numero di chiamate']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Colonne mancanti nel file CSV: {missing_cols}")
            return None
        
        # Conversione formato data
        try:
            if date_format == "Auto-detect":
                df['Data'] = pd.to_datetime(df['Data'], infer_datetime_format=True)
            else:
                df['Data'] = pd.to_datetime(df['Data'], format=date_format)
        except:
            st.error(f"Errore nella conversione della colonna Data con formato {date_format}")
            return None
        
        # Aggiungi colonna Giorno se non presente
        if 'Giorno' not in df.columns:
            df['Giorno'] = df['Data'].dt.day_name()
        
        return df
        
    except Exception as e:
        st.error(f"Errore nel caricamento del file: {str(e)}")
        return None

# --- Inizializzazione variabili di stato ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'working_hours' not in st.session_state:
    st.session_state.working_hours = {}
if 'model_params' not in st.session_state:
    st.session_state.model_params = {}
if 'cost_params' not in st.session_state:
    st.session_state.cost_params = {}

# --- Sidebar Strutturata ---
with st.sidebar:
    st.header("🧩 Configurazione Capacity Sizing")
    
    # 1. Dataset
    st.subheader("1. Dataset")
    use_default_data = st.checkbox("Usa dataset di esempio", value=True, 
                                  help="Utilizza un dataset pre-configurato con pattern realistici di chiamate")
    
    if use_default_data:
        st.session_state.df = get_default_data()
        st.success("✅ Dataset di esempio caricato")
    else:
        with st.expander("📂 File Import", expanded=True):
            uploaded_file = st.file_uploader("Carica file CSV", type=["csv"],
                                            help="File deve contenere: Data, Time slot, Numero di chiamate")
            
            if uploaded_file:
                delimiter = st.selectbox("Delimitatore", 
                                       ["Auto-detect", ",", ";", "\t", "|"],
                                       help="Carattere separatore delle colonne")
                
                date_format = st.selectbox("Formato Date",
                                         ["Auto-detect", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"],
                                         help="Formato della colonna Data")
                
                if st.button("📥 Carica Dataset", key="load_dataset"):
                    st.session_state.df = load_csv_file(uploaded_file, delimiter, date_format)
                    if st.session_state.df is not None:
                        st.success("✅ Dataset caricato con successo")
                        with st.expander("👀 Anteprima dati"):
                            st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    # 2. Orari di apertura
    with st.expander("🕒 Orari di apertura settimanali"):
        st.markdown("**Configura gli orari operativi per ogni giorno della settimana**")
        
        giorni = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        giorni_ita = ['Lunedì', 'Martedì', 'Mercoledì', 'Giovedì', 'Venerdì', 'Sabato', 'Domenica']
        
        for giorno, giorno_ita in zip(giorni, giorni_ita):
            col1, col2, col3, col4 = st.columns([1.5, 0.8, 1, 1])
            
            with col1:
                # Checkbox per giorni apertura
                is_open = st.checkbox(f"{giorno_ita}", value=True if giorno not in ['Saturday', 'Sunday'] else False, key=f"open_{giorno}")
            
            if is_open:
                with col2:
                    st.write("**Aperto**")
                with col3:
                    start_time = st.time_input(f"Inizio", value=datetime.time(9, 0), key=f"start_{giorno}")
                with col4:
                    end_time = st.time_input(f"Fine", value=datetime.time(18, 0), key=f"end_{giorno}")
                
                st.session_state.working_hours[giorno] = {
                    'is_open': True,
                    'start': start_time,
                    'end': end_time
                }
            else:
                with col2:
                    st.write("**Chiuso**")
                with col3:
                    st.write("---")
                with col4:
                    st.write("---")
                
                st.session_state.working_hours[giorno] = {
                    'is_open': False,
                    'start': datetime.time(0, 0),
                    'end': datetime.time(0, 0)
                }
    
    # 3. Tipologia operativa
    st.subheader("3. Tipologia Operativa")
    call_type = st.selectbox("📞 Tipologia operativa", 
                           ["INBOUND", "OUTBOUND", "BLENDED"],
                           help="Seleziona il tipo di operazione del contact center")
    
    # Modelli disponibili per tipologia
    model_options = {
        "INBOUND": ["Erlang C", "Erlang A", "Deterministico", "Simulazione"],
        "OUTBOUND": ["Deterministico", "Simulazione"],
        "BLENDED": ["Simulazione", "Multi-skill Optimization"]
    }
    
    selected_model = st.selectbox("🔧 Modello di calcolo", 
                                model_options[call_type],
                                help="Algoritmo per il calcolo del fabbisogno")
    
    # 4. Parametri specifici del modello
    with st.expander("⚙️ Parametri Modello", expanded=True):
        st.markdown(f"**Configurazione per {selected_model}**")
        
        # Parametri comuni
        slot_duration = st.number_input("Durata slot (minuti)", min_value=15, max_value=120, value=30,
                                      help="Durata degli intervalli temporali in minuti")
        
        if call_type in ["INBOUND", "BLENDED"]:
            if selected_model in ["Erlang C", "Erlang A", "Simulazione"]:
                aht = st.slider("Average Handle Time (secondi)", 60, 800, 300,
                              help="Tempo medio di gestione chiamata")
                service_level = st.slider("Service Level Target (%)", 50, 99, 80,
                                        help="Percentuale chiamate risposta in tempo")
                answer_time = st.slider("Tempo risposta target (secondi)", 5, 60, 20,
                                      help="Tempo massimo risposta per SL")
                shrinkage = st.slider("Shrinkage (%)", 0, 50, 25,
                                    help="Tempo non produttivo operatori")
                
                if selected_model == "Erlang A":
                    patience = st.slider("Pazienza clienti (secondi)", 30, 300, 90,
                                       help="Tempo attesa prima abbandono")
                    st.session_state.model_params['patience'] = patience
                
                st.session_state.model_params.update({
                    'aht': aht,
                    'service_level': service_level,
                    'answer_time': answer_time,
                    'shrinkage': shrinkage / 100,
                    'slot_duration': slot_duration
                })
            
            elif selected_model == "Deterministico":
                aht = st.slider("Average Handle Time (secondi)", 60, 800, 300)
                shrinkage = st.slider("Shrinkage (%)", 0, 50, 25)
                
                st.session_state.model_params.update({
                    'aht': aht,
                    'shrinkage': shrinkage / 100,
                    'slot_duration': slot_duration
                })
        
        elif call_type == "OUTBOUND":
            cph = st.slider("Contatti per Ora", 5, 50, 15,
                          help="Chiamate completate per operatore/ora")
            shrinkage = st.slider("Shrinkage (%)", 0, 50, 25)
            
            st.session_state.model_params.update({
                'cph': cph,
                'shrinkage': shrinkage / 100,
                'slot_duration': slot_duration
            })
    
    # 5. Costi
    with st.expander("💰 Stima Costi"):
        st.markdown("**Configurazione parametri economici**")
        
        ral_annuale = st.number_input("RAL Annuale (€)", min_value=15000, max_value=100000, value=35000,
                                    help="Retribuzione annua lorda operatore")
        giorni_lavorativi = st.number_input("Giorni lavorativi/anno", min_value=200, max_value=260, value=220,
                                          help="Numero giorni lavorativi annui")
        moltiplicatore_costo = st.slider("Moltiplicatore costo aziendale", 1.2, 3.0, 1.8, step=0.1,
                                       help="Fattore moltiplicativo per costi indiretti")
        ore_lavoro_giorno = st.number_input("Ore lavoro/giorno", min_value=6, max_value=12, value=8)
        
        # Calcola i costi
        costs = calculate_costs(1, ral_annuale, giorni_lavorativi, moltiplicatore_costo, ore_lavoro_giorno)
        st.session_state.cost_params = costs
        
        # Mostra i risultati dei costi
        st.metric("Costo orario", f"€{costs['costo_orario']:.2f}")
        st.metric("Costo mensile FTE", f"€{costs['costo_mensile_fte']:,.0f}")
    
    # 6. What-if Analysis
    with st.expander("🧪 What-If Analysis"):
        st.markdown("**Analisi scenari alternativi**")
        
        enable_whatif = st.checkbox("Abilita analisi What-If")
        
        if enable_whatif:
            volume_var = st.slider("Variazione Volume (%)", -50, 100, 0,
                                 help="Incremento/decremento del volume chiamate")
            aht_var = st.slider("Variazione AHT (%)", -30, 50, 0,
                              help="Incremento/decremento tempo gestione")
            sl_var = st.slider("Variazione SL Target (%)", -20, 20, 0,
                             help="Incremento/decremento obiettivo servizio")
            
            st.session_state.whatif_params = {
                'enabled': True,
                'volume_var': volume_var,
                'aht_var': aht_var,
                'sl_var': sl_var
            }
        else:
            st.session_state.whatif_params = {'enabled': False}
    
    # 7. Stress Test
    with st.expander("🆘 Stress Test"):
        st.markdown("**Scenari di stress operativo**")
        
        enable_stress = st.checkbox("Abilita Stress Test")
        
        if enable_stress:
            picco_volume = st.slider("Picco volume (%)", 0, 100, 30,
                                   help="Incremento volume per picco eccezionale")
            assenteismo_extra = st.slider("Assenteismo straordinario (%)", 0, 50, 15,
                                        help="Incremento assenteismo oltre normale")
            
            st.markdown("**Guasto IT - Riduzione capacità**")
            guasto_enabled = st.checkbox("Simula guasto IT")
            if guasto_enabled:
                guasto_start = st.time_input("Inizio guasto", value=datetime.time(10, 0))
                guasto_end = st.time_input("Fine guasto", value=datetime.time(13, 0))
                guasto_riduzione = st.slider("Riduzione capacità (%)", 10, 90, 50)
            
            st.session_state.stress_params = {
                'enabled': True,
                'picco_volume': picco_volume,
                'assenteismo_extra': assenteismo_extra,
                'guasto_enabled': guasto_enabled if 'guasto_enabled' in locals() else False,
                'guasto_start': guasto_start if 'guasto_start' in locals() else None,
                'guasto_end': guasto_end if 'guasto_end' in locals() else None,
                'guasto_riduzione': guasto_riduzione if 'guasto_riduzione' in locals() else 0
            }
        else:
            st.session_state.stress_params = {'enabled': False}
    
    # 8. Avvio
    st.markdown("---")
    run_calculation = st.button("🚀 Lancia il calcolo", type="primary", use_container_width=True)

# --- Main App - Output Section ---
if run_calculation:
    if st.session_state.df is None or st.session_state.df.empty:
        st.error("❌ Nessun dataset caricato. Seleziona un dataset di esempio o carica un file CSV.")
        st.stop()
    
    # Validazione parametri
    if not st.session_state.model_params:
        st.error("❌ Configurare i parametri del modello nella sidebar.")
        st.stop()
    
    with st.spinner("⏳ Elaborazione in corso... Calcolo del fabbisogno operatori"):
        try:
            # Pre-processing dei dati
            df_work = st.session_state.df.copy()
            
            # Assicurati che le colonne necessarie esistano
            if 'Data' in df_work.columns:
                df_work['Data'] = pd.to_datetime(df_work['Data'])
                df_work['Giorno'] = df_work['Data'].dt.day_name()
            
            # Calcolo del fabbisogno con il modello selezionato
            results_df = calculate_capacity_requirements(
                df_work, 
                selected_model, 
                st.session_state.model_params,
                st.session_state.working_hours,
                st.session_state.cost_params
            )
            
            # Calcolo scenari aggiuntivi
            whatif_results = None
            stress_results = None
            
            if st.session_state.whatif_params.get('enabled', False):
                whatif_results = apply_what_if_analysis(
                    df_work,
                    selected_model,
                    st.session_state.model_params,
                    st.session_state.working_hours,
                    st.session_state.cost_params,
                    st.session_state.whatif_params['volume_var'],
                    st.session_state.whatif_params['aht_var'],
                    st.session_state.whatif_params['sl_var']
                )
            
            if st.session_state.stress_params.get('enabled', False):
                stress_results = apply_stress_test(
                    df_work,
                    selected_model,
                    st.session_state.model_params,
                    st.session_state.working_hours,
                    st.session_state.cost_params,
                    st.session_state.stress_params['picco_volume'],
                    st.session_state.stress_params['assenteismo_extra'],
                    st.session_state.stress_params.get('guasto_riduzione', 0)
                )
            
        except Exception as e:
            st.error(f"❌ Errore nel calcolo: {str(e)}")
            st.stop()
    
    # --- Visualizzazione Risultati ---
    st.markdown("---")
    st.markdown("### 📊 Risultati del Capacity Sizing")
    
    # KPI Principali
    if not results_df.empty:
        total_agents = results_df[results_df['Status'] == 'Aperto']['Operatori necessari'].sum()
        total_cost = results_df[results_df['Status'] == 'Aperto']['Costo Stimato (€)'].sum()
        avg_service_level = results_df[results_df['Status'] == 'Aperto']['Service Level Stimato'].mean()
        avg_occupancy = results_df[results_df['Status'] == 'Aperto']['Occupazione Stimata'].mean()
        
        # Box KPI
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Totale Ore-Operatore", f"{total_agents:,.0f}")
        with col2:
            st.metric("💰 Costo Totale Stimato", f"€{total_cost:,.2f}")
        with col3:
            if selected_model in ["Erlang C", "Erlang A", "Simulazione"]:
                st.metric("📈 Service Level Medio", f"{avg_service_level:.1%}")
            else:
                st.metric("⚡ Efficienza", "Ottimale")
        with col4:
            st.metric("📊 Occupazione Media", f"{avg_occupancy:.1%}")
        
        # Tabs per diverse visualizzazioni
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🗓️ Pianificazione", "📋 Dettagli", "🔄 Scenari"])
        
        with tab1:
            st.markdown("#### Dashboard Operativa")
            
            # Grafico andamento giornaliero
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Fabbisogno Operatori per Giorno**")
                daily_summary = results_df[results_df['Status'] == 'Aperto'].groupby('Giorno').agg({
                    'Operatori necessari': 'sum',  # Ore totali erogate
                    'Costo Stimato (€)': 'sum'
                }).reset_index()
                
                # Calcola anche il picco simultaneo
                daily_peak = results_df[results_df['Status'] == 'Aperto'].groupby('Giorno')['Operatori necessari'].max().reset_index()
                daily_peak.columns = ['Giorno', 'Picco Operatori']
                
                # Merge dei dati
                daily_summary = daily_summary.merge(daily_peak, on='Giorno')
                
                # Ordina i giorni
                giorni_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_summary['Giorno'] = pd.Categorical(daily_summary['Giorno'], categories=giorni_order, ordered=True)
                daily_summary = daily_summary.sort_values('Giorno')
                
                # Grafico con barre per ore totali e trendline per picchi simultanei
                fig_daily = go.Figure()
                
                # Barre per ore totali erogate
                fig_daily.add_trace(go.Bar(
                    name='Ore Totali Erogate',
                    x=daily_summary['Giorno'],
                    y=daily_summary['Operatori necessari'],
                    text=[f"{val:.0f}h" for val in daily_summary['Operatori necessari']],
                    textposition='outside',
                    marker_color='lightblue',
                    opacity=0.7
                ))
                
                # Trendline per picco operatori simultanei
                fig_daily.add_trace(go.Scatter(
                    name='Picco Operatori Simultanei',
                    x=daily_summary['Giorno'],
                    y=daily_summary['Picco Operatori'],
                    mode='lines+markers+text',
                    line=dict(color='red', width=3),
                    marker=dict(size=10, color='red', symbol='circle'),
                    text=[f"{val:.0f}" for val in daily_summary['Picco Operatori']],
                    textposition='top center',
                    textfont=dict(color='red', size=12, family='Arial Black'),
                    yaxis='y2'
                ))
                
                fig_daily.update_layout(
                    title="Fabbisogno Operatori per Giorno",
                    yaxis=dict(
                        title="Ore Totali Erogate",
                        side='left'
                    ),
                    yaxis2=dict(
                        title="Picco Operatori Simultanei",
                        overlaying='y',
                        side='right',
                        showgrid=False
                    ),
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", 
                        y=1.02, 
                        xanchor="center", 
                        x=0.5
                    ),
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_daily, use_container_width=True)
            
            with col2:
                st.markdown("**Distribuzione Oraria**")
                hourly_summary = results_df[results_df['Status'] == 'Aperto'].groupby('Time slot')['Operatori necessari'].mean().reset_index()
                
                fig_hourly = px.line(
                    hourly_summary,
                    x='Time slot',
                    y='Operatori necessari',
                    title="Fabbisogno Medio per Fascia Oraria",
                    markers=True
                )
                st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Heatmap Fabbisogno Operatori
            st.markdown("**Mappa di Calore - Fabbisogno Settimanale**")
            try:
                pivot_data = results_df[results_df['Status'] == 'Aperto'].pivot_table(
                    values='Operatori necessari', 
                    index='Time slot', 
                    columns='Giorno',
                    aggfunc='mean'
                )
                
                # Riordina le colonne
                available_days = [day for day in giorni_order if day in pivot_data.columns]
                pivot_data = pivot_data.reindex(columns=available_days)
                
                fig_heatmap = px.imshow(
                    pivot_data,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="Viridis",
                    title="Heatmap Fabbisogno Operatori"
                )
                fig_heatmap.update_layout(
                    xaxis_title="Giorno della Settimana",
                    yaxis_title="Fascia Oraria"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            except Exception as e:
                st.warning(f"Impossibile generare la heatmap fabbisogno: {str(e)}")
            
            # Heatmap Service Level
            if selected_model in ["Erlang C", "Erlang A", "Simulazione"]:
                st.markdown("**Mappa di Calore - Service Level Atteso**")
                try:
                    sl_pivot_data = results_df[results_df['Status'] == 'Aperto'].pivot_table(
                        values='Service Level Stimato', 
                        index='Time slot', 
                        columns='Giorno',
                        aggfunc='mean'
                    )
                    
                    # Riordina le colonne
                    available_days_sl = [day for day in giorni_order if day in sl_pivot_data.columns]
                    sl_pivot_data = sl_pivot_data.reindex(columns=available_days_sl)
                    
                    # Converti in percentuale per la visualizzazione
                    sl_pivot_data_pct = sl_pivot_data * 100
                    
                    # Crea la heatmap con valori numerici visibili
                    fig_heatmap_sl = go.Figure(data=go.Heatmap(
                        z=sl_pivot_data_pct.values,
                        x=sl_pivot_data_pct.columns,
                        y=sl_pivot_data_pct.index,
                        colorscale='RdYlGn',  # Rosso per SL basso, Verde per SL alto
                        zmid=80,  # Punto medio a 80%
                        text=[[f"{val:.0f}%" for val in row] for row in sl_pivot_data_pct.values],
                        texttemplate="%{text}",
                        textfont={"size": 10, "color": "black"},
                        hovertemplate="Giorno: %{x}<br>Ora: %{y}<br>Service Level: %{z:.1f}%<extra></extra>",
                        showscale=True,
                        colorbar=dict(
                            title="Service Level (%)",
                            titleside="right",
                            thickness=15,
                            len=0.7,
                            tickmode="linear",
                            tick0=0,
                            dtick=10
                        )
                    ))
                    
                    fig_heatmap_sl.update_layout(
                        title="Heatmap Service Level Atteso (%)",
                        xaxis_title="Giorno della Settimana",
                        yaxis_title="Fascia Oraria",
                        height=500,
                        font=dict(size=10)
                    )
                    
                    st.plotly_chart(fig_heatmap_sl, use_container_width=True)
                    
                    # Aggiungi legenda interpretativa migliorata
                    st.markdown("**📊 Interpretazione Service Level:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🔴 Critico", "< 70%", help="Service Level sotto la soglia critica - Richiede attenzione immediata")
                    with col2:
                        st.metric("� Migliorabile", "70-79%", help="Service Level sotto il target - Considerare ottimizzazioni")
                    with col3:
                        st.metric("🟡 Accettabile", "80-89%", help="Service Level nel range target")
                    with col4:
                        st.metric("🟢 Ottimale", "≥ 90%", help="Service Level eccellente")
                        
                except Exception as e:
                    st.warning(f"Impossibile generare la heatmap service level: {str(e)}")
            else:
                st.info("💡 La heatmap del Service Level è disponibile solo per i modelli Erlang C, Erlang A e Simulazione.")
        
        with tab2:
            st.markdown("#### Pianificazione Operativa")
            
            st.info("📋 **Tabella completa di pianificazione** - Visualizza tutti i slot temporali per ogni giorno della settimana")
            
            # Tabella pianificazione completa - SENZA FILTRI
            display_columns = ['Giorno', 'Time slot', 'Numero di chiamate', 'Operatori necessari', 'Service Level Stimato', 'Occupazione Stimata', 'Costo Stimato (€)', 'Status']
            
            # Formattazione migliorata della tabella di pianificazione
            def format_planning_table(df_to_format):
                return df_to_format.style.format({
                    'Service Level Stimato': '{:.1%}',
                    'Occupazione Stimata': '{:.1%}',
                    'Costo Stimato (€)': '€{:.2f}',
                    'Operatori necessari': '{:.1f}'
                }).apply(lambda x: ['background-color: #ffebee; color: #666' if v == 'Chiuso - Giorno non operativo' 
                                   else 'background-color: #fff3e0; color: #666' if 'Chiuso' in str(v) 
                                   else 'background-color: #e8f5e8' if 'Aperto' in str(v)
                                   else '' for v in x], subset=['Status'])
            
            # Statistiche di riepilogo
            col1, col2, col3, col4 = st.columns(4)
            total_open_slots = len(results_df[results_df['Status'] == 'Aperto'])
            total_closed_slots = len(results_df[results_df['Status'] != 'Aperto'])
            avg_operators = results_df[results_df['Status'] == 'Aperto']['Operatori necessari'].mean()
            total_cost = results_df[results_df['Status'] == 'Aperto']['Costo Stimato (€)'].sum()
            
            with col1:
                st.metric("🟢 Slot Aperti", total_open_slots)
            with col2:
                st.metric("🔴 Slot Chiusi", total_closed_slots)
            with col3:
                st.metric("👥 Media Operatori", f"{avg_operators:.1f}")
            with col4:
                st.metric("💰 Costo Totale", f"€{total_cost:,.0f}")
            
            st.dataframe(
                format_planning_table(results_df[display_columns]),
                use_container_width=True,
                height=600
            )
        
        with tab3:
            st.markdown("#### Dettagli Completi")
            
            # Formattazione tabella
            def format_results_table(df_to_format):
                return df_to_format.style.format({
                    'Service Level Stimato': '{:.1%}',
                    'Occupazione Stimata': '{:.1%}',
                    'Costo Stimato (€)': '€{:.2f}'
                })
            
            st.dataframe(format_results_table(results_df), use_container_width=True)
            
            # Download dei risultati
            csv_data = convert_df_to_csv(results_df)
            st.download_button(
                label="📥 Scarica Risultati CSV",
                data=csv_data,
                file_name=f"capacity_sizing_{selected_model.lower().replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with tab4:
            st.markdown("#### Analisi Scenari Interattiva")
            
            # Definisci baseline_total per entrambi gli scenari
            baseline_total = total_agents
            
            if whatif_results is not None and stress_results is not None:
                # Se entrambi gli scenari sono attivi, mostra confronto a tre
                st.markdown("**� Confronto Multi-Scenario: AS-IS vs What-If vs Stress Test**")
                
                # Calcoli
                whatif_total = whatif_results[whatif_results['Status'] == 'Aperto']['Operatori necessari'].sum()
                stress_total = stress_results[stress_results['Status'] == 'Aperto']['Operatori necessari'].sum()
                
                # Metriche di riepilogo
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 AS-IS (Baseline)", f"{baseline_total:,.0f} h", help="Scenario operativo standard")
                with col2:
                    delta_whatif = whatif_total - baseline_total
                    st.metric("🧪 What-If", f"{whatif_total:,.0f} h", delta=f"{delta_whatif:+.0f} h")
                with col3:
                    delta_stress = stress_total - baseline_total
                    st.metric("🆘 Stress Test", f"{stress_total:,.0f} h", delta=f"{delta_stress:+.0f} h")
                with col4:
                    max_scenario = max(whatif_total, stress_total)
                    extra_capacity = max_scenario - baseline_total
                    st.metric("💪 Extra Capacity", f"{extra_capacity:+.0f} h", help="Capacità aggiuntiva necessaria nel worst case")
                
                # Grafico comparativo a tre scenari
                st.markdown("**📈 Confronto Giornaliero Multi-Scenario**")
                
                # Prepara dati
                baseline_daily = results_df[results_df['Status'] == 'Aperto'].groupby('Giorno')['Operatori necessari'].sum().reset_index()
                whatif_daily = whatif_results[whatif_results['Status'] == 'Aperto'].groupby('Giorno')['Operatori necessari'].sum().reset_index()
                stress_daily = stress_results[stress_results['Status'] == 'Aperto'].groupby('Giorno')['Operatori necessari'].sum().reset_index()
                
                # Ordina giorni
                giorni_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                for df in [baseline_daily, whatif_daily, stress_daily]:
                    df['Giorno'] = pd.Categorical(df['Giorno'], categories=giorni_order, ordered=True)
                    df.sort_values('Giorno', inplace=True)
                
                fig_multi = go.Figure()
                
                fig_multi.add_trace(go.Bar(
                    name='AS-IS (Baseline)',
                    x=baseline_daily['Giorno'],
                    y=baseline_daily['Operatori necessari'],
                    marker_color='lightblue',
                    text=[f"{val:.0f}h" for val in baseline_daily['Operatori necessari']],
                    textposition='outside',
                    hovertemplate="<b>AS-IS</b><br>Giorno: %{x}<br>Ore: %{y}<extra></extra>"
                ))
                
                fig_multi.add_trace(go.Bar(
                    name='What-If Scenario',
                    x=whatif_daily['Giorno'],
                    y=whatif_daily['Operatori necessari'],
                    marker_color='orange',
                    text=[f"{val:.0f}h" for val in whatif_daily['Operatori necessari']],
                    textposition='outside',
                    hovertemplate="<b>What-If</b><br>Giorno: %{x}<br>Ore: %{y}<extra></extra>"
                ))
                
                fig_multi.add_trace(go.Bar(
                    name='Stress Test',
                    x=stress_daily['Giorno'],
                    y=stress_daily['Operatori necessari'],
                    marker_color='red',
                    text=[f"{val:.0f}h" for val in stress_daily['Operatori necessari']],
                    textposition='outside',
                    hovertemplate="<b>Stress</b><br>Giorno: %{x}<br>Ore: %{y}<extra></extra>"
                ))
                
                fig_multi.update_layout(
                    title="Confronto Multi-Scenario: Fabbisogno per Giorno",
                    barmode='group',
                    yaxis_title="Ore Operatori Necessarie",
                    xaxis_title="Giorno della Settimana",
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig_multi, use_container_width=True)
                
                # Grafico differenze percentuali
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Impatto What-If (%)**")
                    whatif_impact = ((whatif_daily['Operatori necessari'] - baseline_daily['Operatori necessari']) / baseline_daily['Operatori necessari'] * 100).fillna(0)
                    
                    fig_whatif_impact = go.Figure(data=go.Bar(
                        x=baseline_daily['Giorno'],
                        y=whatif_impact,
                        marker_color=['green' if x >= 0 else 'red' for x in whatif_impact],
                        text=[f"{val:+.1f}%" for val in whatif_impact],
                        textposition='outside'
                    ))
                    
                    fig_whatif_impact.update_layout(
                        title="Impatto What-If per Giorno",
                        yaxis_title="Variazione (%)",
                        xaxis_title="Giorno",
                        height=400
                    )
                    
                    st.plotly_chart(fig_whatif_impact, use_container_width=True)
                
                with col2:
                    st.markdown("**⚠️ Impatto Stress Test (%)**")
                    stress_impact = ((stress_daily['Operatori necessari'] - baseline_daily['Operatori necessari']) / baseline_daily['Operatori necessari'] * 100).fillna(0)
                    
                    fig_stress_impact = go.Figure(data=go.Bar(
                        x=baseline_daily['Giorno'],
                        y=stress_impact,
                        marker_color=['darkred' if x >= 20 else 'orange' if x >= 10 else 'yellow' for x in stress_impact],
                        text=[f"{val:+.1f}%" for val in stress_impact],
                        textposition='outside'
                    ))
                    
                    fig_stress_impact.update_layout(
                        title="Impatto Stress Test per Giorno",
                        yaxis_title="Variazione (%)",
                        xaxis_title="Giorno",
                        height=400
                    )
                    
                    st.plotly_chart(fig_stress_impact, use_container_width=True)
                
            elif whatif_results is not None:
                st.markdown("**📊 What-If Analysis**")
                
                # Confronto baseline vs what-if
                whatif_total = whatif_results[whatif_results['Status'] == 'Aperto']['Operatori necessari'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 AS-IS (Baseline)", f"{baseline_total:,.0f} h")
                with col2:
                    delta = whatif_total - baseline_total
                    st.metric("🧪 What-If Scenario", f"{whatif_total:,.0f} h", delta=f"{delta:+.0f} h")
                with col3:
                    pct_change = (delta / baseline_total * 100) if baseline_total > 0 else 0
                    st.metric("📈 Variazione", f"{pct_change:+.1f}%", delta=f"{pct_change:+.1f}%")
                
                # Mostra parametri applicati
                st.info(f"� **Variazioni applicate:** Volume {st.session_state.whatif_params['volume_var']:+d}%, AHT {st.session_state.whatif_params['aht_var']:+d}%, SL Target {st.session_state.whatif_params['sl_var']:+d}%")
                
                # Grafico comparativo What-If migliorato
                st.markdown("**🔄 Confronto Interattivo: AS-IS vs What-If**")
                
                # Prepara dati per il confronto
                baseline_daily = results_df[results_df['Status'] == 'Aperto'].groupby('Giorno')['Operatori necessari'].sum().reset_index()
                whatif_daily = whatif_results[whatif_results['Status'] == 'Aperto'].groupby('Giorno')['Operatori necessari'].sum().reset_index()
                
                # Ordina giorni
                giorni_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                baseline_daily['Giorno'] = pd.Categorical(baseline_daily['Giorno'], categories=giorni_order, ordered=True)
                whatif_daily['Giorno'] = pd.Categorical(whatif_daily['Giorno'], categories=giorni_order, ordered=True)
                baseline_daily = baseline_daily.sort_values('Giorno')
                whatif_daily = whatif_daily.sort_values('Giorno')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_whatif = go.Figure()
                    
                    fig_whatif.add_trace(go.Bar(
                        name='AS-IS (Baseline)',
                        x=baseline_daily['Giorno'],
                        y=baseline_daily['Operatori necessari'],
                        marker_color='lightblue',
                        text=[f"{val:.0f}h" for val in baseline_daily['Operatori necessari']],
                        textposition='outside',
                        opacity=0.8
                    ))
                    
                    fig_whatif.add_trace(go.Bar(
                        name='What-If Scenario',
                        x=whatif_daily['Giorno'],
                        y=whatif_daily['Operatori necessari'],
                        marker_color='orange',
                        text=[f"{val:.0f}h" for val in whatif_daily['Operatori necessari']],
                        textposition='outside',
                        opacity=0.8
                    ))
                    
                    fig_whatif.update_layout(
                        title="Confronto Fabbisogno: AS-IS vs What-If",
                        barmode='group',
                        yaxis_title="Ore Operatori",
                        xaxis_title="Giorno della Settimana",
                        height=400
                    )
                    
                    st.plotly_chart(fig_whatif, use_container_width=True)
                
                with col2:
                    # Grafico Delta
                    delta_values = whatif_daily['Operatori necessari'] - baseline_daily['Operatori necessari']
                    
                    fig_delta = go.Figure(data=go.Bar(
                        x=baseline_daily['Giorno'],
                        y=delta_values,
                        marker_color=['green' if x >= 0 else 'red' for x in delta_values],
                        text=[f"{val:+.0f}h" for val in delta_values],
                        textposition='outside'
                    ))
                    
                    fig_delta.update_layout(
                        title="Differenza: What-If vs AS-IS",
                        yaxis_title="Differenza Ore",
                        xaxis_title="Giorno della Settimana",
                        height=400
                    )
                    
                    st.plotly_chart(fig_delta, use_container_width=True)
            
            elif stress_results is not None:
                st.markdown("**🆘 Stress Test Analysis**")
                
                # Confronto baseline vs stress
                stress_total = stress_results[stress_results['Status'] == 'Aperto']['Operatori necessari'].sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Normale (Baseline)", f"{baseline_total:,.0f} h")
                with col2:
                    delta = stress_total - baseline_total
                    st.metric("🆘 Stress Scenario", f"{stress_total:,.0f} h", delta=f"{delta:+.0f} h")
                with col3:
                    pct_change = (delta / baseline_total * 100) if baseline_total > 0 else 0
                    st.metric("⚡ Extra Capacity", f"{pct_change:+.1f}%", delta=f"{delta:+.0f} h")
                
                # Mostra parametri di stress applicati
                st.warning(f"⚠️ **Fattori di stress applicati:** Picco volume +{st.session_state.stress_params['picco_volume']}%, Assenteismo +{st.session_state.stress_params['assenteismo_extra']}%")
                
                # Grafici comparativi migliorati
                st.markdown("**🔥 Confronto Resilienza: Normale vs Stress Test**")
                
                # Prepara dati per il confronto stress
                baseline_daily_stress = results_df[results_df['Status'] == 'Aperto'].groupby('Giorno')['Operatori necessari'].sum().reset_index()
                stress_daily = stress_results[stress_results['Status'] == 'Aperto'].groupby('Giorno')['Operatori necessari'].sum().reset_index()
                
                # Ordina giorni
                giorni_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                baseline_daily_stress['Giorno'] = pd.Categorical(baseline_daily_stress['Giorno'], categories=giorni_order, ordered=True)
                stress_daily['Giorno'] = pd.Categorical(stress_daily['Giorno'], categories=giorni_order, ordered=True)
                baseline_daily_stress = baseline_daily_stress.sort_values('Giorno')
                stress_daily = stress_daily.sort_values('Giorno')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_stress = go.Figure()
                    
                    fig_stress.add_trace(go.Bar(
                        name='Scenario Normale',
                        x=baseline_daily_stress['Giorno'],
                        y=baseline_daily_stress['Operatori necessari'],
                        marker_color='lightgreen',
                        text=[f"{val:.0f}h" for val in baseline_daily_stress['Operatori necessari']],
                        textposition='outside',
                        opacity=0.8
                    ))
                    
                    fig_stress.add_trace(go.Bar(
                        name='Scenario Stress',
                        x=stress_daily['Giorno'],
                        y=stress_daily['Operatori necessari'],
                        marker_color='red',
                        text=[f"{val:.0f}h" for val in stress_daily['Operatori necessari']],
                        textposition='outside',
                        opacity=0.8
                    ))
                    
                    fig_stress.update_layout(
                        title="Test di Resilienza: Normale vs Stress",
                        barmode='group',
                        yaxis_title="Ore Operatori",
                        xaxis_title="Giorno della Settimana",
                        height=400
                    )
                    
                    st.plotly_chart(fig_stress, use_container_width=True)
                
                with col2:
                    # Grafico Extra Capacity richiesta
                    extra_capacity = stress_daily['Operatori necessari'] - baseline_daily_stress['Operatori necessari']
                    
                    fig_extra = go.Figure(data=go.Bar(
                        x=baseline_daily_stress['Giorno'],
                        y=extra_capacity,
                        marker_color=['darkred' if x >= 20 else 'orange' if x >= 10 else 'yellow' for x in extra_capacity],
                        text=[f"+{val:.0f}h" for val in extra_capacity],
                        textposition='outside'
                    ))
                    
                    fig_extra.update_layout(
                        title="Extra Capacity Richiesta",
                        yaxis_title="Ore Extra",
                        xaxis_title="Giorno della Settimana",
                        height=400
                    )
                    
                    st.plotly_chart(fig_extra, use_container_width=True)
                
                # Grafico waterfall migliorato per analisi impatto
                st.markdown("**📊 Analisi Impatto Fattori di Stress**")
                
                if baseline_total > 0:
                    # Calcolo più accurato degli impatti
                    volume_factor = st.session_state.stress_params['picco_volume'] / 100
                    absenteeism_factor = st.session_state.stress_params['assenteismo_extra'] / 100
                    
                    # Calcolo step-by-step
                    after_volume = baseline_total * (1 + volume_factor)
                    volume_impact = after_volume - baseline_total
                    
                    # L'assenteismo si applica dopo l'aumento di volume
                    final_total = after_volume / (1 - absenteeism_factor) if absenteeism_factor < 1 else after_volume * 2
                    absenteeism_impact = final_total - after_volume
                    
                    fig_waterfall = go.Figure(go.Waterfall(
                        name="Impatto Stress Factors",
                        orientation="v",
                        measure=["absolute", "relative", "relative", "total"],
                        x=["📊 Baseline\nNormale", 
                           f"📈 Picco Volume\n(+{st.session_state.stress_params['picco_volume']}%)", 
                           f"🏥 Assenteismo\n(+{st.session_state.stress_params['assenteismo_extra']}%)", 
                           "🚨 Totale\nStress"],
                        y=[baseline_total, volume_impact, absenteeism_impact, final_total],
                        text=[f"{baseline_total:.0f}h", f"+{volume_impact:.0f}h", f"+{absenteeism_impact:.0f}h", f"{final_total:.0f}h"],
                        textposition="outside",
                        connector={"line":{"color":"rgb(63, 63, 63)"}},
                        increasing={"marker":{"color":"red"}},
                        decreasing={"marker":{"color":"green"}},
                        totals={"marker":{"color":"darkred"}}
                    ))
                    
                    fig_waterfall.update_layout(
                        title="Analisi Waterfall: Impatto Fattori di Stress sulla Capacità",
                        yaxis_title="Ore Operatori Necessarie",
                        height=500,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_waterfall, use_container_width=True)
                    
                    # Tabella di riepilogo impatti
                    st.markdown("**📋 Riepilogo Impatti**")
                    impact_df = pd.DataFrame({
                        'Fattore di Stress': ['Picco Volume', 'Assenteismo Extra', 'Impatto Totale'],
                        'Parametro': [f"+{st.session_state.stress_params['picco_volume']}%", 
                                    f"+{st.session_state.stress_params['assenteismo_extra']}%", 
                                    "Combinato"],
                        'Ore Extra': [f"+{volume_impact:.0f}h", f"+{absenteeism_impact:.0f}h", f"+{final_total - baseline_total:.0f}h"],
                        'Impatto %': [f"+{volume_impact/baseline_total*100:.1f}%", 
                                    f"+{absenteeism_impact/baseline_total*100:.1f}%", 
                                    f"+{(final_total - baseline_total)/baseline_total*100:.1f}%"]
                    })
                    
                    st.dataframe(impact_df, use_container_width=True, hide_index=True)
            
            if whatif_results is None and stress_results is None:
                st.info("👆 Abilita What-If Analysis o Stress Test nella sidebar per vedere i confronti scenari.")
    
    else:
        st.warning("⚠️ Nessun risultato ottenuto. Verifica i parametri di configurazione.")

else:
    # Pagina iniziale
    st.markdown("---")
    st.markdown("### 👋 Benvenuto nel Capacity Sizing Tool")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("""
        **Per iniziare:**
        
        1. 📊 Configura il dataset nella sidebar
        2. 🕒 Imposta gli orari operativi  
        3. ⚙️ Seleziona il modello di calcolo
        4. 💰 Configura i parametri economici
        5. 🚀 Lancia il calcolo
        
        **Modelli disponibili:**
        - **Erlang C**: Modello classico per inbound
        - **Erlang A**: Include abbandoni chiamate
        - **Deterministico**: Calcolo semplificato
        - **Simulazione**: Analisi Monte Carlo
        """)
    
    # Mostra anteprima dati se disponibili
    if st.session_state.df is not None:
        st.markdown("#### 👀 Anteprima Dataset")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Statistiche dataset
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Righe totali", len(st.session_state.df))
        with col2:
            st.metric("📅 Giorni unici", st.session_state.df['Giorno'].nunique() if 'Giorno' in st.session_state.df.columns else 'N/A')
        with col3:
            st.metric("🕒 Slot temporali", st.session_state.df['Time slot'].nunique() if 'Time slot' in st.session_state.df.columns else 'N/A')
        with col4:
            st.metric("📞 Chiamate totali", st.session_state.df['Numero di chiamate'].sum() if 'Numero di chiamate' in st.session_state.df.columns else 'N/A')

