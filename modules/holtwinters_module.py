import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Tuple, Optional

def holt_winters_forecast(
    series: pd.Series,
    forecast_periods: int = 12,
    seasonal_periods: int = 12,
    trend: str = 'add',
    seasonal: str = 'add',
    damped_trend: bool = True,
    initialization_method: str = 'estimated',
    smoothing_level: Optional[float] = None,
    smoothing_trend: Optional[float] = None,
    smoothing_seasonal: Optional[float] = None,
    optimized: bool = True
) -> Tuple[pd.Series, pd.Series, dict]:
    """
    Applica il modello Holt-Winters alla serie temporale fornita e restituisce
    i valori adattati, le previsioni future e i parametri del modello.

    Parameters:
    - series: Serie temporale di input (pd.Series).
    - forecast_periods: Numero di periodi futuri da prevedere.
    - seasonal_periods: Periodo stagionale (es. 12 per dati mensili).
    - trend: Tipo di componente trend ('add' o 'mul').
    - seasonal: Tipo di componente stagionale ('add' o 'mul').
    - damped_trend: Se True, applica un fattore di smorzamento al trend.
    - initialization_method: Metodo di inizializzazione ('estimated', 'heuristic', 'legacy-heuristic', 'known').
    - smoothing_level: Valore di α (livello), se None viene ottimizzato automaticamente.
    - smoothing_trend: Valore di β (trend), se None viene ottimizzato automaticamente.
    - smoothing_seasonal: Valore di γ (stagionalità), se None viene ottimizzato automaticamente.
    - optimized: Se True, ottimizza automaticamente i parametri non specificati.

    Returns:
    - fitted_values: Valori adattati alla serie storica.
    - forecast_values: Previsioni per i periodi futuri.
    - model_params: Dizionario contenente i parametri del modello.
    """
    # Verifica che la serie sia univoca e ordinata
    series = series.sort_index()
    series = series.asfreq(pd.infer_freq(series.index))

    # Costruzione del modello
    model = ExponentialSmoothing(
        series,
        trend=trend,
        damped_trend=damped_trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        initialization_method=initialization_method
    )

    # Adattamento del modello
    fit = model.fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal,
        optimized=optimized
    )

    # Estrazione dei valori adattati e delle previsioni
    fitted_values = fit.fittedvalues
    forecast_values = fit.forecast(forecast_periods)

    # Estrazione dei parametri del modello
    model_params = {
        'smoothing_level (α)': fit.model.params.get('smoothing_level', None),
        'smoothing_trend (β)': fit.model.params.get('smoothing_trend', None),
        'smoothing_seasonal (γ)': fit.model.params.get('smoothing_seasonal', None),
        'damping_trend (ϕ)': fit.model.params.get('damping_trend', None),
        'initial_level': fit.model.params.get('initial_level', None),
        'initial_trend': fit.model.params.get('initial_trend', None),
        'initial_seasonal': fit.model.params.get('initial_seasonal', None)
    }

    return fitted_values, forecast_values, model_params
