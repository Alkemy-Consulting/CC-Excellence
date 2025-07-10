"""
Configurazione centrale per il modulo di forecasting
Contiene costanti, label e configurazioni condivise
"""

from typing import Dict, List, Any
import pandas as pd

# Configurazioni generali
DEFAULT_HORIZON = 30
MIN_DATA_POINTS = 20
DEFAULT_CI_LEVEL = 0.8

# Mapping formati data
DATE_FORMATS = {
    "aaaa-mm-gg": "%Y-%m-%d",
    "gg/mm/aaaa": "%d/%m/%Y", 
    "gg/mm/aa": "%d/%m/%y",
    "mm/gg/aaaa": "%m/%d/%Y",
    "gg.mm.aaaa": "%d.%m.%Y",
    "aaaa/mm/gg": "%Y/%m/%d"
}

# Mapping frequenze temporali
FREQUENCY_MAP = {
    "Daily": "D",
    "Weekly": "W", 
    "Monthly": "M",
    "Quarterly": "Q",
    "Yearly": "Y"
}

# Mapping periodi stagionali automatici
SEASONAL_PERIODS_MAP = {
    "D": 7,    # Giornaliero -> settimanale
    "W": 52,   # Settimanale -> annuale 
    "M": 12,   # Mensile -> annuale
    "Q": 4,    # Trimestrale -> annuale
    "Y": 1     # Annuale -> no stagionalità
}

# Opzioni gestione missing values
MISSING_HANDLING_OPTIONS = [
    "Forward Fill",
    "Backward Fill", 
    "Interpolazione lineare",
    "Zero Fill"
]

# Metodi aggregazione
AGGREGATION_METHODS = ["sum", "mean", "max", "min", "median"]

# Metriche disponibili
AVAILABLE_METRICS = ["MAPE", "MAE", "MSE", "RMSE", "SMAPE"]

# Paesi supportati per festività
HOLIDAY_COUNTRIES = {
    None: "Nessuno",
    'IT': "Italia", 
    'US': "Stati Uniti",
    'UK': "Regno Unito",
    'DE': "Germania", 
    'FR': "Francia",
    'ES': "Spagna"
}

# Colonne che potrebbero contenere date (per auto-detection)
DATE_COLUMN_NAMES = [
    'date', 'data', 'time', 'timestamp', 'datetime', 
    'day', 'giorno', 'periodo', 'period'
]

# Colonne che potrebbero contenere valori target (per auto-detection)
VALUE_COLUMN_NAMES = [
    'value', 'valore', 'volume', 'count', 'amount', 'qty', 'quantity',
    'calls', 'chiamate', 'revenue', 'ricavi', 'sales', 'vendite'
]

# Configurazioni modelli
PROPHET_DEFAULTS = {
    'seasonality_mode': 'additive',
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'interval_width': 0.8,
    'uncertainty_samples': 1000,
    'daily_seasonality': 'auto',
    'weekly_seasonality': 'auto',
    'yearly_seasonality': 'auto',
    'growth': 'linear',
    'mcmc_samples': 0,
    'holidays_country': None,
    'custom_seasonalities': []
}

# Model labels and configurations
MODEL_LABELS = {
    'prophet': 'Prophet',
    'arima': 'ARIMA',
    'sarima': 'SARIMA', 
    'holtwinters': 'Holt-Winters'
}

# SARIMA default parameters
SARIMA_DEFAULTS = {
    'p': 1,
    'd': 1, 
    'q': 1,
    'P': 1,
    'D': 1,
    'Q': 1,
    'seasonal_periods': 12,
    'auto_tune': True
}

# ARIMA default parameters
ARIMA_DEFAULTS = {
    'p': 1,
    'd': 1,
    'q': 1,
    'auto_tune': True
}

# Forecast default parameters
FORECAST_DEFAULTS = {
    'periods': 30,
    'confidence_interval': 0.95,
    'train_size': 0.8
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'height': 500,
    'width': 800,
    'colors': {
        'historical': 'blue',
        'forecast': 'red',
        'confidence': 'rgba(255,0,0,0.2)'
    }
}

# Error messages
ERROR_MESSAGES = {
    'no_data': 'No data provided for forecasting',
    'insufficient_data': 'Insufficient data points for forecasting',
    'invalid_date_column': 'Invalid or missing date column',
    'invalid_target_column': 'Invalid or missing target column',
    'model_fit_failed': 'Model fitting failed',
    'forecast_failed': 'Forecast generation failed'
}

HOLT_WINTERS_DEFAULTS = {
    'trend_type': 'add',
    'seasonal_type': 'add', 
    'damped_trend': True,
    'seasonal_periods': 12,
    'use_custom': False,
    'smoothing_level': 0.2,
    'smoothing_trend': 0.1,
    'smoothing_seasonal': 0.1
}

HOLTWINTERS_DEFAULTS = HOLT_WINTERS_DEFAULTS  # Alias per compatibilità

# Descrizioni modelli
MODEL_DESCRIPTIONS = {
    "Prophet": "🔮 Advanced time series forecasting with automatic seasonality detection and holiday effects",
    "ARIMA": "📊 Classic autoregressive integrated moving average model for stationary time series",
    "SARIMA": "🔄 Seasonal ARIMA model that captures both trend and seasonal patterns",
    "Holt-Winters": "❄️ Exponential smoothing method ideal for data with trend and seasonality",
    "Auto-Select": "🤖 Automatically tests multiple models and selects the best performer"
}

# Tooltip dettagliati per parametri
PARAMETER_TOOLTIPS = {
    'prophet': {
        'seasonality_mode': "Additive: constant seasonal amplitude; Multiplicative: seasonal amplitude scales with trend",
        'changepoint_prior_scale': "Controls trend flexibility. Higher values = more flexible trend, lower = more stable",
        'seasonality_prior_scale': "Controls seasonality strength. Higher values = stronger seasonal patterns",
        'uncertainty_samples': "Number of samples for uncertainty estimation. More samples = more accurate intervals but slower computation"
    },
    'arima': {
        'p': "Autoregressive order: number of past observations to include in the model",
        'd': "Degree of differencing: number of times to difference the data to make it stationary", 
        'q': "Moving average order: number of past forecast errors to include in the model"
    },
    'holt_winters': {
        'trend': "How to model the trend component: additive (linear) or multiplicative (exponential)",
        'seasonal': "How to model seasonality: additive (constant) or multiplicative (growing with trend)",
        'damped_trend': "Whether to dampen the trend in long-term forecasts for more conservative predictions",
        'seasonal_periods': "Number of periods in one seasonal cycle (e.g., 7 for daily data with weekly seasonality)"
    }
}

# Definizioni delle metriche
METRICS_DEFINITIONS = {
    'MAPE': "Mean Absolute Percentage Error - Average absolute percentage difference between actual and predicted values",
    'MAE': "Mean Absolute Error - Average absolute difference between actual and predicted values", 
    'MSE': "Mean Squared Error - Average squared difference between actual and predicted values",
    'RMSE': "Root Mean Squared Error - Square root of MSE, in same units as original data",
    'SMAPE': "Symmetric Mean Absolute Percentage Error - Symmetric version of MAPE, bounded between 0-200%"
}

# UI Labels e tooltip
UI_LABELS = {
    'upload_file': "📂 Carica file dati",
    'sample_data': "Usa dataset di esempio",
    'date_column': "Colonna data",
    'target_column': "Colonna target (valore da prevedere)",
    'frequency': "Frequenza temporale",
    'horizon': "Orizzonte previsionale (periodi)",
    'confidence_interval': "Livello intervallo di confidenza (%)",
    'missing_handling': "Gestione valori mancanti",
    'outlier_detection': "Rilevamento outlier",
    'external_regressors': "Regressori esterni"
}

TOOLTIPS = {
    'date_column': "La colonna che contiene le date/timestamp",
    'target_column': "La variabile numerica da prevedere",
    'frequency': "Aggrega i dati a questo livello temporale",
    'horizon': "Numero di periodi futuri da prevedere",
    'confidence_interval': "Ampiezza delle bande di confidenza del forecast",
    'missing_handling': "Come trattare i valori mancanti nei dati",
    'outlier_detection': "Identifica e rimuovi valori anomali",
    'external_regressors': "Variabili aggiuntive che influenzano il target"
}

# Configurazioni grafici
PLOT_CONFIG = {
    'height': 500,
    'colors': {
        'historical': '#1f77b4',
        'forecast': '#ff7f0e', 
        'confidence': 'rgba(255, 127, 14, 0.2)',
        'trend': '#2ca02c',
        'outliers': '#d62728'
    },
    'line_styles': {
        'historical': 'solid',
        'forecast': 'dash',
        'trend': 'dot'
    }
}

# Configurazioni export
EXPORT_CONFIG = {
    'excel_sheet_names': {
        'data': 'Dati Storici',
        'forecast': 'Previsioni', 
        'metrics': 'Metriche',
        'config': 'Configurazione'
    },
    'pdf_config': {
        'title': 'Report Forecasting',
        'author': 'CC-Excellence',
        'subject': 'Analisi Previsionale'
    }
}
