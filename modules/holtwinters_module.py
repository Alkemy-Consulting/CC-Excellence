 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/modules/holtwinters_module.py b/modules/holtwinters_module.py
index 574027da17e1a639e9eae329f4a6f4703fb8c960..6e6b0106d19ed4cc42d8f4404464e0958aeca0fc 100644
--- a/modules/holtwinters_module.py
+++ b/modules/holtwinters_module.py
@@ -1,78 +1,100 @@
 import pandas as pd
 import numpy as np
 from statsmodels.tsa.holtwinters import ExponentialSmoothing
 from typing import Tuple, Optional
 
+
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
-    """
-    Applica il modello Holt-Winters alla serie temporale fornita e restituisce
-    i valori adattati, le previsioni future e i parametri del modello.
+    """Apply the Holt-Winters model to a time series and return fitted values,
+    forecasts and model parameters.
 
-    Parameters:
-    - series: Serie temporale di input (pd.Series).
-    - forecast_periods: Numero di periodi futuri da prevedere.
-    - seasonal_periods: Periodo stagionale (es. 12 per dati mensili).
-    - trend: Tipo di componente trend ('add' o 'mul').
-    - seasonal: Tipo di componente stagionale ('add' o 'mul').
-    - damped_trend: Se True, applica un fattore di smorzamento al trend.
-    - initialization_method: Metodo di inizializzazione ('estimated', 'heuristic', 'legacy-heuristic', 'known').
-    - smoothing_level: Valore di α (livello), se None viene ottimizzato automaticamente.
-    - smoothing_trend: Valore di β (trend), se None viene ottimizzato automaticamente.
-    - smoothing_seasonal: Valore di γ (stagionalità), se None viene ottimizzato automaticamente.
-    - optimized: Se True, ottimizza automaticamente i parametri non specificati.
+    Parameters
+    ----------
+    series : pd.Series
+        Input time series indexed by date.
+    forecast_periods : int, default 12
+        Number of future periods to forecast.
+    seasonal_periods : int, default 12
+        Length of the seasonal cycle.
+    trend : {'add', 'mul'}, default 'add'
+        Type of trend component.
+    seasonal : {'add', 'mul'}, default 'add'
+        Type of seasonal component.
+    damped_trend : bool, default True
+        Whether to apply trend damping.
+    initialization_method : str, default 'estimated'
+        Initialization method used by ``ExponentialSmoothing``.
+    smoothing_level : float, optional
+        Fixed value for the level component (``alpha``).
+    smoothing_trend : float, optional
+        Fixed value for the trend component (``beta``).
+    smoothing_seasonal : float, optional
+        Fixed value for the seasonal component (``gamma``).
+    optimized : bool, default True
+        If ``True`` any parameter not specified is automatically optimized.
 
-    Returns:
-    - fitted_values: Valori adattati alla serie storica.
-    - forecast_values: Previsioni per i periodi futuri.
-    - model_params: Dizionario contenente i parametri del modello.
+    Returns
+    -------
+    fitted_values : pd.Series
+        In-sample predictions from the fitted model.
+    forecast_values : pd.Series
+        Forecasts for ``forecast_periods`` ahead.
+    model_params : dict
+        Dictionary with model parameters and fit statistics.
     """
-    # Verifica che la serie sia univoca e ordinata
+    # Ensure the series is ordered and has an inferred frequency
     series = series.sort_index()
     series = series.asfreq(pd.infer_freq(series.index))
 
-    # Costruzione del modello
+    # Build and fit the model
     model = ExponentialSmoothing(
         series,
         trend=trend,
         damped_trend=damped_trend,
         seasonal=seasonal,
         seasonal_periods=seasonal_periods,
-        initialization_method=initialization_method
+        initialization_method=initialization_method,
     )
 
-    # Adattamento del modello
     fit = model.fit(
         smoothing_level=smoothing_level,
         smoothing_trend=smoothing_trend,
         smoothing_seasonal=smoothing_seasonal,
-        optimized=optimized
+        optimized=optimized,
     )
 
-    # Estrazione dei valori adattati e delle previsioni
     fitted_values = fit.fittedvalues
     forecast_values = fit.forecast(forecast_periods)
 
-    # Estrazione dei parametri del modello
+    # Compute basic fit statistics
+    with np.errstate(divide='ignore', invalid='ignore'):
+        mape = np.mean(np.abs((series - fitted_values) / series)) * 100
+    rmse = np.sqrt(np.mean(np.square(series - fitted_values)))
+
     model_params = {
-        'smoothing_level (α)': fit.model.params.get('smoothing_level', None),
-        'smoothing_trend (β)': fit.model.params.get('smoothing_trend', None),
-        'smoothing_seasonal (γ)': fit.model.params.get('smoothing_seasonal', None),
-        'damping_trend (ϕ)': fit.model.params.get('damping_trend', None),
-        'initial_level': fit.model.params.get('initial_level', None),
-        'initial_trend': fit.model.params.get('initial_trend', None),
-        'initial_seasonal': fit.model.params.get('initial_seasonal', None)
+        'smoothing_level (α)': fit.params.get('smoothing_level'),
+        'smoothing_trend (β)': fit.params.get('smoothing_trend'),
+        'smoothing_seasonal (γ)': fit.params.get('smoothing_seasonal'),
+        'damping_trend (ϕ)': fit.params.get('damping_trend'),
+        'initial_level': fit.params.get('initial_level'),
+        'initial_trend': fit.params.get('initial_trend'),
+        'initial_seasonal': fit.params.get('initial_seasonal'),
+        'aic': fit.aic,
+        'bic': fit.bic,
+        'mape': mape,
+        'rmse': rmse,
     }
 
     return fitted_values, forecast_values, model_params
 
EOF
)
