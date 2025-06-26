import streamlit as st
import pandas as pd
import plotly.express as px


def run_exploratory_analysis(df):
    st.subheader("📊 Analisi Esplorativa")

    st.markdown("**📌 Dimensioni del dataset:**")
    st.write(f"{df.shape[0]} righe × {df.shape[1]} colonne")

    st.markdown("**🔍 Prime righe del dataset:**")
    st.dataframe(df.head())

    st.markdown("**📈 Statistiche descrittive:**")
    st.dataframe(df.describe())

    st.markdown("**📅 Andamento nel tempo:**")
    if "ds" in df.columns and "y" in df.columns:
        fig = px.line(df, x="ds", y="y", title="Serie temporale")
        st.plotly_chart(fig)
    else:
        st.warning("Il dataset non contiene le colonne 'ds' e 'y' per visualizzare la serie storica.")
