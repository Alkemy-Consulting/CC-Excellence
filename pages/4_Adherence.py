import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import matplotlib.pyplot as plt
import plotly.express as px

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
    df_cons["Operatore"] = df_cons["Agent"]

    # --- Merge mapping ---
    df_turni = df_turni.merge(df_map[["ID file turnistica_clean", "ID HubSpot_clean", "Nome Cognome"]],
                              how="left", left_on="Operatore_clean", right_on="ID file turnistica_clean")
    df_cons = df_cons.merge(df_map[["ID file turnistica_clean", "ID HubSpot_clean", "Nome Cognome"]],
                            how="left", left_on="Operatore_clean", right_on="ID HubSpot_clean")

    # --- Parse date e orari ---
    df_turni["Data"] = pd.to_datetime(df_turni["Data"], errors='coerce')
    df_cons["Data"] = pd.to_datetime(df_cons["Date"], errors='coerce')
    df_turni["Ingresso_HHMM"] = df_turni["Ingresso"].apply(parse_time)

    if "Schedule — First Activity Start" in df_cons.columns:
        df_cons["FirstActivityStart_HHMM"] = pd.to_timedelta(df_cons["Schedule — First Activity Start"].astype(float), unit="s").apply(lambda x: (datetime(1900, 1, 1) + x).time())
    else:
        st.error("Colonna 'Schedule — First Activity Start' non trovata nel file consuntivo.")
        st.stop()

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

    # --- Escludi operatori specifici ---
    da_escludere = ["facchetti", "ciceri", "bellandi", "morise"]
    df_valida = df_valida[~df_valida["Operatore_clean"].str.contains('|'.join(da_escludere), case=False, na=False)]

    # --- Filtro dinamico per mese ---
    mesi_disponibili = df_valida["Data"].dt.month.unique()
    mese_selezionato = st.selectbox("Seleziona il mese da analizzare", sorted(mesi_disponibili))
    df_valida_mese = df_valida[df_valida["Data"].dt.month == mese_selezionato]

    st.subheader("Deviazione Media per Operatore")
    media_op = df_valida_mese.groupby(["Nome Cognome", "Smart_flag"])["Deviazione_minuti"].mean().unstack()
    media_op.columns = ["Presenza", "Smart Working"]
    st.dataframe(media_op.round(2))

    fig_media_op = px.bar(
        media_op.reset_index().melt(id_vars="Nome Cognome", value_name="Deviazione media (min)", var_name="Modalità"),
        x="Nome Cognome",
        y="Deviazione media (min)",
        color="Modalità",
        barmode="group",
        title="Deviazione Media per Operatore"
    )
    st.plotly_chart(fig_media_op, use_container_width=True)

    # --- Metriche riepilogative ---
    st.subheader("Metriche sintetiche")
    tot_pres = df_valida_mese[df_valida_mese["Smart_flag"] == 0]
    tot_smart = df_valida_mese[df_valida_mese["Smart_flag"] == 1]
    perc_pres = (tot_pres["Fuori_orario"].sum() / len(tot_pres)) * 100 if len(tot_pres) > 0 else 0
    perc_smart = (tot_smart["Fuori_orario"].sum() / len(tot_smart)) * 100 if len(tot_smart) > 0 else 0
    media_pres = tot_pres["Deviazione_minuti"].mean() if len(tot_pres) > 0 else 0
    media_smart = tot_smart["Deviazione_minuti"].mean() if len(tot_smart) > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric("% Ritardi in Presenza", f"{perc_pres:.1f}%")
        st.metric("Ritardo medio in Presenza", f"{media_pres:.1f} min")
    with col2:
        st.metric("% Ritardi in Smart Working", f"{perc_smart:.1f}%")
        st.metric("Ritardo medio in Smart Working", f"{media_smart:.1f} min")

    # --- Trendline percentuale fuori orario ---
    df_valida["Mese"] = df_valida["Data"].dt.month
    df_trend = df_valida.groupby(["Mese", "Smart_flag"]).agg(
        Totale=("Deviazione_minuti", "count"),
        Ritardi=("Fuori_orario", "sum")
    ).reset_index()
    df_trend["Percentuale Ritardi"] = (df_trend["Ritardi"] / df_trend["Totale"]) * 100
    df_trend["Modalità"] = df_trend["Smart_flag"].map({0: "Presenza", 1: "Smart Working"})

    fig_trend = px.line(
        df_trend,
        x="Mese",
        y="Percentuale Ritardi",
        color="Modalità",
        markers=True,
        labels={"Percentuale Ritardi": "% Ritardi"},
        title="Trend % Ritardi Mensili"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Download file unificato ---
    df_export = df_valida[[
        "Operatore", "Nome Cognome", "Data", "FirstActivityStart_HHMM", "Ingresso_HHMM", "Deviazione_minuti", "Smart_flag"
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
