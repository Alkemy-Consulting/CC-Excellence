import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="Analisi Ritardi Operatori", layout="wide")
st.title("Analisi Ritardi Operatori Customer Service")

# Sidebar upload
with st.sidebar:
    st.header("Carica i file")
    file_turni = st.file_uploader("File Turnistica", type=["xlsx"])
    file_consuntivo = st.file_uploader("File Consuntivo (Effettivo)", type=["csv"])
    file_mapping = st.file_uploader("File Trascodifica", type=["xlsx"])
    tolleranza = st.number_input("Tolleranza in minuti", min_value=1, max_value=60, value=15)

# Funzione per trasformare orario (stringa) in oggetto time
def parse_time(val):
    try:
        return pd.to_datetime(val, format="%H:%M:%S", errors='coerce').time()
    except:
        return None

# Pulizia e merge solo se tutti i file sono caricati
if file_turni and file_consuntivo and file_mapping:
    # --- Caricamento file ---
    df_turni = pd.read_excel(file_turni, engine="openpyxl")
    df_cons = pd.read_csv(file_consuntivo)
    df_map = pd.read_excel(file_mapping, engine="openpyxl")

    # --- Pulizia nomi ---
    df_map.columns = df_map.columns.str.strip()
    df_turni.columns = df_turni.columns.str.strip()
    df_cons.columns = df_cons.columns.str.strip()

    df_map["ID file turnistica_clean"] = df_map["ID file turnistica"].astype(str).str.lower().str.strip()
    df_map["ID HubSpot_clean"] = df_map["ID HubSpot"].astype(str).str.lower().str.strip()

    df_turni["Operatore_clean"] = df_turni["Operatore"].astype(str).str.lower().str.strip()
    df_cons["Operatore_clean"] = df_cons["Agent"].astype(str).str.lower().str.strip()

    # --- Merge mapping ---
    df_turni = df_turni.merge(df_map[["ID file turnistica_clean", "ID HubSpot_clean"]],
                              how="left", left_on="Operatore_clean", right_on="ID file turnistica_clean")
    df_cons = df_cons.merge(df_map[["ID file turnistica_clean", "ID HubSpot_clean"]],
                            how="left", left_on="Operatore_clean", right_on="ID HubSpot_clean")

    # --- Parse date e orari ---
    df_turni["Data"] = pd.to_datetime(df_turni["Data"], errors='coerce')
    df_cons["Data"] = pd.to_datetime(df_cons["Data"], errors='coerce')
    df_turni["Ingresso_HHMM"] = df_turni["Ingresso"].apply(parse_time)
    df_cons["FirstActivityStart_HHMM"] = pd.to_timedelta(df_cons["FirstActivityStart"].astype(float), unit="s").apply(lambda x: (datetime(1900, 1, 1) + x).time())

    df_turni["Smart"] = df_turni["Smart"].fillna(0)

    # --- Join per confronto ---
    df_joined = df_cons.merge(
        df_turni[["Data", "ID HubSpot_clean", "Ingresso_HHMM", "Smart"]],
        how="left",
        left_on=["Data", "Operatore_clean"],
        right_on=["Data", "ID HubSpot_clean"]
    )

    # --- Calcolo ritardi ---
    df_joined["Ingresso_previsto"] = pd.to_datetime(df_joined["Ingresso_HHMM"].astype(str), errors='coerce')
    df_joined["Ingresso_effettivo"] = pd.to_datetime(df_joined["FirstActivityStart_HHMM"].astype(str), errors='coerce')
    df_joined["Deviazione_minuti"] = (df_joined["Ingresso_effettivo"] - df_joined["Ingresso_previsto"]).dt.total_seconds() / 60

    # --- Flag fuori orario ---
    df_valida = df_joined[df_joined["Deviazione_minuti"].notna()].copy()
    df_valida["Smart_flag"] = df_valida["Smart"].fillna(0).astype(int)
    df_valida["Fuori_orario"] = df_valida["Deviazione_minuti"].abs() > tolleranza

    # --- Filtro dinamico per mese ---
    mesi_disponibili = df_valida["Data"].dt.month.unique()
    mese_selezionato = st.selectbox("Seleziona il mese da analizzare", sorted(mesi_disponibili))
    df_valida = df_valida[df_valida["Data"].dt.month == mese_selezionato]

    st.subheader("Deviazione Media per Operatore")
    media_op = df_valida.groupby(["Operatore", "Smart_flag"])["Deviazione_minuti"].mean().unstack()
    media_op.columns = ["Presenza", "Smart Working"]
    st.dataframe(media_op.round(2))

    st.subheader("Ritardi Gravi (> 60 min) > 3 volte")
    gravi = df_valida[df_valida["Deviazione_minuti"] > 60]
    count_gravi = gravi.groupby("Operatore").size()
    seriali = count_gravi[count_gravi > 3].reset_index()
    seriali.columns = ["Operatore", "Occorrenze"]
    st.dataframe(seriali)

    st.subheader("Distribuzione Fuori Orario")
    dist = df_valida.groupby(["Smart_flag", "Fuori_orario"]).size().unstack().fillna(0)
    dist.index = dist.index.map({0: "Presenza", 1: "Smart Working"})
    st.bar_chart(dist)

    # --- Download file unificato ---
    df_export = df_valida[[
        "Operatore", "Data", "FirstActivityStart_HHMM", "Ingresso_HHMM", "Deviazione_minuti", "Smart_flag"
    ]].rename(columns={
        "FirstActivityStart_HHMM": "Ingresso Effettivo",
        "Ingresso_HHMM": "Ingresso Previsto",
        "Deviazione_minuti": "Deviazione (min)",
        "Smart_flag": "Smart Working (1=Si,0=No)"
    })

    buffer = io.BytesIO()
    df_export.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button("Scarica file unificato Excel", data=buffer, file_name="analisi_ritardi_operatori.xlsx")
