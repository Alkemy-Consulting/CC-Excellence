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

def compute_all_metrics(y_true, y_pred):
    """Calcola tutte le metriche standard (MAE, MSE, RMSE, MAPE, SMAPE)."""
    # Rimuovi i valori nulli o infiniti per evitare errori di calcolo
    valid_indices = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_indices]
    y_pred = y_pred[valid_indices]

    if len(y_true) == 0:
        return {k: np.nan for k in ["MAE", "MSE", "RMSE", "MAPE", "SMAPE"]}

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Calcolo robusto di MAPE e SMAPE per evitare divisione per zero
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan

    denominator = np.abs(y_true) + np.abs(y_pred)
    non_zero_denom_mask = denominator != 0
    if np.any(non_zero_denom_mask):
        smape = np.mean(
            2 * np.abs(y_pred[non_zero_denom_mask] - y_true[non_zero_denom_mask]) / denominator[non_zero_denom_mask]
        ) * 100
    else:
        smape = np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape
    }