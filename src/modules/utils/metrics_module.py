import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_metrics(y_true, y_pred, metrics):
    """
    Calcola un dizionario di metriche da una lista di nomi.
    Gestisce edge cases e valori invalidi.
    """
    # Validazione input
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devono avere la stessa lunghezza")
    
    # Rimuovi valori non finiti
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid_mask):
        return {m: np.nan for m in metrics}
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    out = {}
    
    for m in metrics:
        try:
            if m == "MAE":
                out["MAE"] = mean_absolute_error(y_true_valid, y_pred_valid)
            elif m == "RMSE":
                out["RMSE"] = mean_squared_error(y_true_valid, y_pred_valid, squared=False)
            elif m == "MSE":
                out["MSE"] = mean_squared_error(y_true_valid, y_pred_valid)
            elif m == "MAPE":
                # Maschera per evitare divisione per zero
                non_zero_mask = y_true_valid != 0
                if np.any(non_zero_mask):
                    mape_values = np.abs((y_true_valid[non_zero_mask] - y_pred_valid[non_zero_mask]) / y_true_valid[non_zero_mask])
                    out["MAPE"] = np.mean(mape_values) * 100
                else:
                    out["MAPE"] = np.nan
            elif m == "SMAPE":
                # Symmetric MAPE più robusta
                denominator = np.abs(y_true_valid) + np.abs(y_pred_valid)
                non_zero_denom_mask = denominator != 0
                if np.any(non_zero_denom_mask):
                    smape_values = 2 * np.abs(y_true_valid[non_zero_denom_mask] - y_pred_valid[non_zero_denom_mask]) / denominator[non_zero_denom_mask]
                    out["SMAPE"] = np.mean(smape_values) * 100
                else:
                    out["SMAPE"] = np.nan
            else:
                out[m] = np.nan  # Metrica non riconosciuta
        except Exception:
            out[m] = np.nan  # Errore nel calcolo
            
    return out

def compute_all_metrics(y_true, y_pred):
    """
    Calcola tutte le metriche standard con gestione robusta degli errori.
    """
    # Conversione e validazione
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {k: np.nan for k in ["MAE", "MSE", "RMSE", "MAPE", "SMAPE"]}
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Dimensioni incompatibili: y_true={len(y_true)}, y_pred={len(y_pred)}")

    # Rimuovi valori non finiti
    valid_indices = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if not np.any(valid_indices):
        return {k: np.nan for k in ["MAE", "MSE", "RMSE", "MAPE", "SMAPE"]}

    y_true_clean = y_true[valid_indices]
    y_pred_clean = y_pred[valid_indices]

    try:
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        mse = mean_squared_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mse)
    except Exception:
        mae = mse = rmse = np.nan

    # Calcolo robusto di MAPE
    try:
        non_zero_mask = y_true_clean != 0
        if np.any(non_zero_mask):
            mape_values = np.abs((y_true_clean[non_zero_mask] - y_pred_clean[non_zero_mask]) / y_true_clean[non_zero_mask])
            # Rimuovi valori infiniti che potrebbero risultare da divisioni per numeri molto piccoli
            mape_values = mape_values[np.isfinite(mape_values)]
            mape = np.mean(mape_values) * 100 if len(mape_values) > 0 else np.nan
        else:
            mape = np.nan
    except Exception:
        mape = np.nan

    # Calcolo robusto di SMAPE
    try:
        denominator = np.abs(y_true_clean) + np.abs(y_pred_clean)
        non_zero_denom_mask = denominator > 1e-10  # Soglia per evitare problemi numerici
        if np.any(non_zero_denom_mask):
            smape_values = 2 * np.abs(y_pred_clean[non_zero_denom_mask] - y_true_clean[non_zero_denom_mask]) / denominator[non_zero_denom_mask]
            # Rimuovi valori infiniti
            smape_values = smape_values[np.isfinite(smape_values)]
            smape = np.mean(smape_values) * 100 if len(smape_values) > 0 else np.nan
        else:
            smape = np.nan
    except Exception:
        smape = np.nan

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "SMAPE": smape
    }