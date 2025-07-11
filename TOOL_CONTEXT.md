# TOOL_CONTEXT.md — Context for GitHub Copilot

Questo file raccoglie **tutte** le linee guida tecniche, UX/UI e architettura di progetto necessarie perché GitHub Copilot generi codice conforme agli standard di CC-Excellence.

---

## 1. Principi Generali e Requisiti Utente

- **Parametri in Sidebar**: **Tutti** i parametri di configurazione UX/UI devono essere esposti nella sidebar di Streamlit. Non utilizzare input nella parte centrale.
- **Data Preparation nella Sidebar**: tutte le attività di preprocessing dei dati e le chiamate alle funzioni di forecasting o sizing devono essere effettuate nella sidebar; la pagina principale deve essere utilizzata esclusivamente per la visualizzazione dei risultati.
- **Collapsed UI**: Tutti gli `st.expander` nella sidebar devono essere chiusi di default per un’interfaccia più pulita.
- **Auto–Tuning di Default**: Ogni modello deve prevedere un meccanismo di auto–tuning automatico come impostazione predefinita. Utenti esperti possono passare a configurazione manuale avanzata.
- **Help Contestuale**: Ogni widget avrà un tooltip che spiega:
  - Cosa fa
  - Range consigliato
  - Suggerimento ottimale
- **Livello di dettaglio UI**: Mostrare sempre tutte le impostazioni, incluse quelle più tecniche, mantenendo il toggle Auto–Tuning per semplificare.
- **Dataset di Esempio**: Inserire un uploader per dataset di esempio “clicca-e-guarda” in sidebar, **non caricato di default**.
- **Esecuzione su Click**: Ogni modello parte solo dopo `st.button("Esegui <Model>")` e visualizza risultati nella parte centrale.
- **Modifiche Mirate**: Per ogni patch o bugfix, modificare solo la parte richiesta senza toccare altro, salvo esplicita indicazione.
- **Coerenza Grafica**: Elementi ripetuti (pulsanti, slider) devono avere stile, dimensioni e colori identici in tutti i moduli.

---

## 2. Architettura di Progetto

```
root/
├── app.py                  # Entry point Streamlit
├── modules/
│   ├── __init__.py
│   ├── config.py           # Costanti, default, tooltip texts
│   ├── data_utils.py       # Preprocessing, imputazione, outlier, freq inference
│   ├── ui_components.py    # Widget helpers (sidebar, buttons, expanders)
│   ├── forecast_engine.py  # Orchestrazione, session_state, click logic
│   ├── prophet_module.py   # Prophet con auto-tuning, backtest
│   ├── arima_module.py     # ARIMA/SARIMA con auto_arima, backtest
│   ├── holtwinters_module.py # Holt-Winters con parametri ottimizzati, backtest
│   ├── capacity_sizing.py  # Capacity sizing con Erlang C e scenario analysis
│   └── render_utils.py     # Plotly charts, tabelle, export Excel/PDF
├── tests/                  # Unit & integration tests
├── requirements.txt        # Dipendenze Python
├── README.md               # Documentazione utente e guida rapida
└── TOOL_CONTEXT.md         # Questo file di contesto
```

- **app.py**: carica `TOOL_CONTEXT.md`, imposta layout, importa moduli.
- **modules/config.py**: centralizza parametri e range per sidebar.
- **modules/data_utils.py**: funzioni pure per data cleaning e metriche.
- **modules/ui_components.py**: wrappers per creare widget coerenti.
- **modules/forecast_engine.py**: gestisce il flusso di esecuzione su click.
- **modules/*_module.py**: ogni modulo implementa `run_<model>` con interfaccia uniforme.
- **modules/render_utils.py**: componenti comuni per grafici ed export.
- **tests/**: copertura di tutte le funzioni chiave e UI components.

---

## 3. Centralizzazione degli Elementi Ripetuti

- Applicare **principio DRY**: estrarre helper per elementi riutilizzati (es. `compute_metrics`, `create_execute_button`, `export_to_excel`).
- Garantisce coerenza, riduce bug e facilita testing.

---

## 4. UX/UI in Sidebar (Pattern)

1. **Data Source**: file uploader (CSV/XLSX) + preview + statistiche.
2. **Data Cleaning**: imputazione, outlier detection con expander.
3. **Modello**: toggle Auto–Tuning + parametri manuali avanzati (tooltip inclusi).
4. **Execution**: `st.button("Esegui <Model>")` inside sidebar.
5. **Export**: CSV/Excel, PDF via `reportlab` o `pdfkit`.
6. **Dataset Esempio**: switch per caricare dataset di esempio.

---

## 5. Librerie Affidabili

- Prophet, pmdarima, statsmodels, numpy, pandas, plotly, holidays.

---

## 6. Best Practices di Sviluppo

- **Type hints**, **relative imports**, **docstring** in Google style.
- **Tests** in `tests/` per validazione input e output.
- **Refactoring**: estrarre componenti comuni.

---

## 7. Testing & CI

- Test unitari, integration tests e snapshot UI.
- CI con GitHub Actions per linting, type check, test.

---

## 8. Deploy & Manutenzione

- Streamlit Cloud / Docker.
- Aggiornamenti modulari con versioning semantico.

---

## 🔄 Latest Changes

### UI Layout & Data Analysis Enhancement (Current Session)
- **Moved Analytics to Main Page**: Comprehensive dataset overview, quality assessment, and time series analysis moved from sidebar to main content area
- **Enhanced Pre-Analysis**: Added detailed statistical summary, distribution analysis, outlier detection, and trend analysis with multiple moving averages
- **Vertical Layout**: All elements displayed in vertical order for better readability
- **Data Preparation in Sidebar**: All data processing and preparation logic kept in sidebar, main page only for visualization
- **Quality Scoring**: Implemented comprehensive data quality scoring system with issue detection
- **Time Series Decomposition**: Added advanced time series analysis including frequency detection and trend decomposition

---

**Ultimo Aggiornamento**: sessione corrente

**Versione**: v1.2.0
