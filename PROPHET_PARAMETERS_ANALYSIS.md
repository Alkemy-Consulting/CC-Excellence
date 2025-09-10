# 📊 ANALISI COMPLETA DEI PARAMETRI PROPHET

## 🎯 PANORAMICA

Il modulo Prophet riceve due tipologie di parametri:
1. **`model_config`** - Parametri specifici del modello Prophet (da UI)
2. **`base_config`** - Parametri di base per il forecasting (da sezione Forecast Settings)

## 📋 PARAMETRI MODEL_CONFIG (Prophet-specifici)

### 🤖 **AUTO-TUNING PARAMETERS**

#### 1. `auto_tune` (boolean)
- **Funzione**: Abilita/disabilita l'ottimizzazione automatica dei parametri
- **Valore default**: `True`
- **Effetto**: Quando attivo, disabilita i controlli manuali per i parametri core
- **Utilizzo nel codice**: ❌ **NON IMPLEMENTATO** (placeholder nel codice)
- **Ridondanza**: Nessuna

#### 2. `tuning_horizon` (int)
- **Funzione**: Orizzonte temporale per l'auto-tuning (in giorni)
- **Range**: 7-90 giorni
- **Valore default**: 30
- **Effetto**: Dovrebbe definire quanti giorni usare per ottimizzare i parametri
- **Utilizzo nel codice**: ❌ **NON IMPLEMENTATO**
- **Ridondanza**: ⚠️ **POTENZIALMENTE RIDONDANTE** con `cv_horizon`

### 📊 **CROSS-VALIDATION PARAMETERS**

#### 3. `enable_cross_validation` (boolean)
- **Funzione**: Abilita la cross-validation per valutazione del modello
- **Valore default**: `False`
- **Effetto**: Dovrebbe eseguire time-series cross-validation
- **Utilizzo nel codice**: ❌ **NON IMPLEMENTATO** (placeholder nel codice)
- **Ridondanza**: Nessuna

#### 4. `cv_horizon` (int)
- **Funzione**: Orizzonte di previsione per ogni fold della CV
- **Range**: 7-60 giorni
- **Valore default**: 30
- **Effetto**: Quanti giorni prevedere in ogni fold di cross-validation
- **Utilizzo nel codice**: ❌ **NON IMPLEMENTATO**
- **Ridondanza**: ⚠️ **RIDONDANTE** con `tuning_horizon` e `forecast_periods`

#### 5. `cv_folds` (int)
- **Funzione**: Numero di fold per la cross-validation
- **Range**: 3-10
- **Valore default**: 5
- **Effetto**: In quante parti dividere i dati per la CV
- **Utilizzo nel codice**: ❌ **NON IMPLEMENTATO**
- **Ridondanza**: Nessuna

### 🎉 **HOLIDAY PARAMETERS**

#### 6. `add_holidays` (boolean)
- **Funzione**: Aggiunge effetti delle festività al modello
- **Valore default**: `False`
- **Effetto**: Include festività nazionali nel modello Prophet
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

#### 7. `holidays_country` (string)
- **Funzione**: Specifica il paese per le festività
- **Opzioni**: ['US', 'CA', 'UK', 'DE', 'FR', 'IT', 'ES', 'AU', 'JP']
- **Valore default**: 'US'
- **Effetto**: Definisce quale calendario festività usare
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

### 🔧 **CORE PROPHET PARAMETERS**

#### 8. `changepoint_prior_scale` (float)
- **Funzione**: Controlla la flessibilità del trend
- **Range**: 0.001-0.5
- **Valore default**: 0.05
- **Effetto**: Più alto = trend più flessibile, più changepoint
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

#### 9. `seasonality_prior_scale` (float)
- **Funzione**: Controlla la forza della stagionalità
- **Range**: 0.01-100.0
- **Valore default**: 10.0
- **Effetto**: Più alto = stagionalità più forte
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

#### 10. `seasonality_mode` (string)
- **Funzione**: Modalità di stagionalità
- **Opzioni**: ['additive', 'multiplicative']
- **Valore default**: 'additive'
- **Effetto**: Definisce come la stagionalità influenza il trend
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

#### 11. `uncertainty_samples` (int)
- **Funzione**: Numero di campioni per stima dell'incertezza
- **Range**: 100-2000
- **Valore default**: 1000
- **Effetto**: Più campioni = intervalli di confidenza più accurati ma calcolo più lento
- **Utilizzo nel codice**: ❌ **NON UTILIZZATO** (Prophet usa parametro `mcmc_samples`)
- **Ridondanza**: ⚠️ **PARAMETRO ERRATO** - dovrebbe essere `mcmc_samples`

### 📊 **SEASONALITY CONFIGURATION**

#### 12. `yearly_seasonality` (mixed)
- **Funzione**: Abilita/disabilita stagionalità annuale
- **Opzioni**: ['auto', True, False]
- **Valore default**: 'auto'
- **Effetto**: Cattura pattern che si ripetono ogni anno
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

#### 13. `weekly_seasonality` (mixed)
- **Funzione**: Abilita/disabilita stagionalità settimanale
- **Opzioni**: ['auto', True, False]
- **Valore default**: 'auto'
- **Effetto**: Cattura pattern che si ripetono ogni settimana
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

#### 14. `daily_seasonality` (mixed)
- **Funzione**: Abilita/disabilita stagionalità giornaliera
- **Opzioni**: ['auto', True, False]
- **Valore default**: 'auto'
- **Effetto**: Cattura pattern che si ripetono ogni giorno (per dati orari)
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

### 📈 **GROWTH MODEL**

#### 15. `growth` (string)
- **Funzione**: Tipo di modello di crescita
- **Opzioni**: ['linear', 'logistic']
- **Valore default**: 'linear'
- **Effetto**: Linear = crescita illimitata, Logistic = crescita con saturazione
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Nessuna

## 📋 PARAMETRI BASE_CONFIG

### 16. `forecast_periods` (int)
- **Funzione**: Numero di periodi futuri da prevedere
- **Valore default**: 30
- **Effetto**: Quanti punti temporali prevedere nel futuro
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: ⚠️ **RIDONDANTE** con `horizon` dalla sezione Forecast Settings

### 17. `confidence_interval` (float)
- **Funzione**: Livello di confidenza per gli intervalli
- **Valore default**: 0.95
- **Effetto**: Ampiezza degli intervalli di confidenza (95% = intervalli più larghi)
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: ⚠️ **RIDONDANTE** con `confidence_level` dalla sezione Forecast Settings

### 18. `train_size` (float)
- **Funzione**: Proporzione di dati per training
- **Valore default**: 0.8 (80%)
- **Effetto**: Quanto dei dati storici usare per allenare vs valutare
- **Utilizzo nel codice**: ✅ **IMPLEMENTATO** correttamente
- **Ridondanza**: Possibilmente ridondante con parametri di backtesting

## 🚨 PROBLEMI IDENTIFICATI

### ❌ **PARAMETRI NON IMPLEMENTATI**
1. `auto_tune` - Funzionalità non implementata
2. `tuning_horizon` - Non utilizzato
3. `enable_cross_validation` - Non implementata
4. `cv_horizon` - Non utilizzato  
5. `cv_folds` - Non utilizzato

### ⚠️ **PARAMETRI RIDONDANTI**
1. **`tuning_horizon` vs `cv_horizon` vs `forecast_periods`** - Tre parametri per orizzonte temporale
2. **`confidence_interval` vs `confidence_level`** - Due parametri per stesso concetto
3. **`uncertainty_samples`** - Parametro errato, dovrebbe essere `mcmc_samples`

### 🔧 **PARAMETRI FORECAST SETTINGS NON UTILIZZATI**
Dalla sezione "5. Forecast Settings" molti parametri non vengono passati a Prophet:
- `frequency` - Non utilizzato
- `aggregation` - Non utilizzato  
- `include_confidence` - Non utilizzato
- `interval_width` - Non utilizzato
- `enable_backtesting` - Non utilizzato
- `backtest_strategy` - Non utilizzato
- Tutti i parametri di backtesting avanzato

## 💡 RACCOMANDAZIONI

### 1. **RIMUOVERE PARAMETRI RIDONDANTI**
```python
# ❌ RIMUOVERE:
tuning_horizon  # Usa solo cv_horizon o forecast_periods
confidence_interval vs confidence_level  # Unificare in uno solo
interval_width  # Se non implementato
```

### 2. **CORREGGERE PARAMETRO ERRATO**
```python
# ❌ ATTUALE:
uncertainty_samples

# ✅ CORRETTO:
mcmc_samples  # Parametro Prophet corretto
```

### 3. **IMPLEMENTARE O RIMUOVERE**
- Implementare auto_tune e cross_validation
- Oppure rimuovere dalla UI se non implementate

### 4. **UNIFICARE CONFIGURAZIONI**
- Usare solo `base_config` per parametri comuni
- Evitare duplicazioni tra sezioni diverse

## ✅ PARAMETRI EFFETTIVAMENTE UTILIZZATI

**Solo questi 10 parametri hanno effetto reale:**
1. `add_holidays` ✅
2. `holidays_country` ✅
3. `changepoint_prior_scale` ✅
4. `seasonality_prior_scale` ✅
5. `seasonality_mode` ✅
6. `yearly_seasonality` ✅
7. `weekly_seasonality` ✅
8. `daily_seasonality` ✅
9. `growth` ✅
10. `forecast_periods` (da base_config) ✅
11. `confidence_interval` (da base_config) ✅
12. `train_size` (da base_config) ✅

**Gli altri 6+ parametri sono inutili** perché non implementati o ridondanti.
