import streamlit as st

# Configurazione generale dell'app
st.set_page_config(page_title="Contact Center Tool", page_icon="📞", layout="wide")

# Titolo principale
st.title("📞 Contact Center Tool")

# Menu laterale
menu = st.sidebar.radio("Seleziona una funzione:", ["Forecasting", "Capacity Planning", "WFM", "Quality Assurance"])

# Routing verso le funzioni specifiche
if menu == "Forecasting":
    from pages import forecasting
    forecasting.show()

elif menu == "Capacity Planning":
    from pages import capacity_planning
    capacity_planning.show()

elif menu == "WFM":
    from pages import wfm
    wfm.show()

elif menu == "Quality Assurance":
    from pages import quality_assurance
    quality_assurance.show()
