import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred, metrics):
    """Calcola un dizionario di metriche da una lista di nomi."""
    out = {}
    # evitiamo divisioni per zero
    mask = y_true != 0
    for m in metrics:
        if m == "MAE":
            out["MAE"] = mean_absolute_error(y_true, y_pred)
        elif m == "RMSE":
            out["RMSE"] = mean_squared_error(y_true, y_pred, squared=False)
        elif m == "MAPE":
            out["MAPE"] = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        elif m == "SMAPE":
            # symmetric MAPE
            denom = (np.abs(y_true) + np.abs(y_pred))
            mask2 = denom != 0
            out["SMAPE"] = np.mean(2 * np.abs(y_true[mask2] - y_pred[mask2]) / denom[mask2]) * 100
        # aggiungere altre metriche se servono
    return out