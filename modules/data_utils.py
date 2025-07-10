"""
Utilità per la gestione e pre-processing dei dati nel modulo forecasting
Include funzioni per: rilevamento automatico, gestione missing, outlier detection, validazione
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import warnings
from .config import (
    DATE_COLUMN_NAMES, VALUE_COLUMN_NAMES, DATE_FORMATS, 
    SEASONAL_PERIODS_MAP, MISSING_HANDLING_OPTIONS
)

def generate_sample_data() -> pd.DataFrame:
    """
    Genera un DataFrame di esempio con trend, stagionalità e rumore
    
    Returns:
        pd.DataFrame: Dataset di esempio
    """
    today = datetime.today()
    dates = pd.date_range(start=today - timedelta(days=730), end=today, freq='D')
    
    # Trend crescente
    trend = np.linspace(50, 150, len(dates))
    
    # Stagionalità annuale e settimanale
    annual_seasonality = 25 * (1 + np.sin(np.arange(len(dates)) * 2 * np.pi / 365.25))
    weekly_seasonality = 15 * (1 + np.sin(np.arange(len(dates)) * 2 * np.pi / 7))
    
    # Rumore
    noise = np.random.normal(0, 15, len(dates))
    
    # Combinazione
    volume = trend + annual_seasonality + weekly_seasonality + noise
    volume = np.maximum(0, volume)  # Assicura che non ci siano valori negativi
    
    return pd.DataFrame({
        'date': dates,
        'volume': volume.round(2)
    })

def detect_file_format(uploaded_file) -> str:
    """
    Rileva automaticamente il formato del file caricato
    
    Args:
        uploaded_file: File uploadato tramite Streamlit
        
    Returns:
        str: Formato rilevato ('csv' o 'xlsx')
    """
    filename = uploaded_file.name.lower()
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        return 'xlsx'
    elif filename.endswith('.csv'):
        return 'csv'
    else:
        # Default fallback
        return 'csv'

def load_data_file(uploaded_file, delimiter: str = ",", date_format: Optional[str] = None) -> pd.DataFrame:
    """
    Carica un file di dati con gestione automatica del formato
    
    Args:
        uploaded_file: File uploadato
        delimiter: Delimitatore per CSV (ignorato per Excel)
        date_format: Formato data opzionale
        
    Returns:
        pd.DataFrame: DataFrame caricato
    """
    try:
        file_format = detect_file_format(uploaded_file)
        
        if file_format == 'xlsx':
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file, delimiter=delimiter)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Errore nel caricamento del file: {str(e)}")
        return pd.DataFrame()

def auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Rileva automaticamente le colonne data e target basandosi sui nomi e tipi
    
    Args:
        df: DataFrame da analizzare
        
    Returns:
        Tuple[str, str]: (nome_colonna_data, nome_colonna_target)
    """
    date_col = None
    target_col = None
    
    # Rileva colonna data
    for col in df.columns:
        col_lower = col.lower()
        # Check per nome
        if any(name in col_lower for name in DATE_COLUMN_NAMES):
            date_col = col
            break
        # Check per tipo datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        # Check se può essere convertito in data
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(), errors='raise')
                date_col = col
                break
            except:
                continue
    
    # Rileva colonna target (numerica)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        for col in numeric_cols:
            col_lower = col.lower()
            if any(name in col_lower for name in VALUE_COLUMN_NAMES):
                target_col = col
                break
        
        # Se non trovato per nome, prendi la prima colonna numerica che non è la data
        if target_col is None:
            for col in numeric_cols:
                if col != date_col:
                    target_col = col
                    break
    
    return date_col, target_col

def get_data_statistics(df: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
    """
    Calcola statistiche base del dataset
    
    Args:
        df: DataFrame
        date_col: Nome colonna data
        target_col: Nome colonna target
        
    Returns:
        Dict: Statistiche del dataset
    """
    stats = {}
    
    try:
        # Statistiche generali
        stats['total_records'] = len(df)
        
        # Statistiche temporali
        if date_col and date_col in df.columns:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            
            min_date = df_temp[date_col].min()
            max_date = df_temp[date_col].max()
            
            if pd.notna(min_date) and pd.notna(max_date):
                stats['date_range_days'] = (max_date - min_date).days
                stats['min_date'] = min_date
                stats['max_date'] = max_date
            else:
                stats['date_range_days'] = 0
                stats['min_date'] = None
                stats['max_date'] = None
            
            stats['duplicate_dates'] = df_temp[date_col].duplicated().sum()
            
            # Rileva frequenza
            try:
                inferred_freq = pd.infer_freq(df_temp[date_col].dropna().sort_values())
                stats['inferred_frequency'] = inferred_freq
            except:
                stats['inferred_frequency'] = None
        
        # Statistiche target
        if target_col and target_col in df.columns:
            target_series = pd.to_numeric(df[target_col], errors='coerce')
            
            stats['min_value'] = target_series.min() if not target_series.empty else 0
            stats['max_value'] = target_series.max() if not target_series.empty else 0
            stats['mean_value'] = target_series.mean() if not target_series.empty else 0
            stats['std_value'] = target_series.std() if not target_series.empty else 0
            stats['missing_values'] = target_series.isnull().sum()
            stats['missing_percentage'] = (stats['missing_values'] / len(df)) * 100 if len(df) > 0 else 0
            stats['negative_count'] = (target_series < 0).sum()
            stats['zero_count'] = (target_series == 0).sum()
    
    except Exception as e:
        st.warning(f"⚠️ Errore nel calcolo delle statistiche: {str(e)}")
        # Return default stats on error
        stats = {
            'total_records': len(df) if df is not None else 0,
            'date_range_days': 0,
            'min_value': 0,
            'max_value': 0,
            'mean_value': 0,
            'std_value': 0,
            'missing_values': 0,
            'missing_percentage': 0,
            'duplicate_dates': 0
        }
    
    return stats

def handle_missing_values(df: pd.DataFrame, target_col: str, method: str) -> pd.DataFrame:
    """
    Gestisce i valori mancanti secondo il metodo specificato
    
    Args:
        df: DataFrame
        target_col: Colonna target
        method: Metodo di imputazione
        
    Returns:
        pd.DataFrame: DataFrame con valori mancanti gestiti
    """
    df_clean = df.copy()
    
    if method == "Forward Fill":
        df_clean[target_col] = df_clean[target_col].ffill()
    elif method == "Backward Fill":
        df_clean[target_col] = df_clean[target_col].bfill()
    elif method == "Interpolazione lineare":
        df_clean[target_col] = df_clean[target_col].interpolate(method='linear')
    elif method == "Zero Fill":
        df_clean[target_col] = df_clean[target_col].fillna(0)
    
    return df_clean

def detect_outliers(df: pd.DataFrame, target_col: str, method: str = "iqr") -> Dict[str, Any]:
    """
    Rileva outlier usando il metodo IQR
    
    Args:
        df: DataFrame
        target_col: Colonna target
        method: Metodo di rilevamento (default: "iqr")
        
    Returns:
        Dict[str, Any]: Statistiche outlier
    """
    outlier_stats = {
        'count': 0,
        'percentage': 0.0,
        'indices': [],
        'values': []
    }
    
    try:
        values = pd.to_numeric(df[target_col], errors='coerce').dropna()
        
        if method == "iqr":
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Trova outlier nel DataFrame originale
            numeric_series = pd.to_numeric(df[target_col], errors='coerce')
            outlier_mask = (numeric_series < lower_bound) | (numeric_series > upper_bound)
            outlier_indices = df[outlier_mask].index.tolist()
            
            outlier_stats['count'] = len(outlier_indices)
            outlier_stats['percentage'] = (len(outlier_indices) / len(df)) * 100
            outlier_stats['indices'] = outlier_indices
            outlier_stats['values'] = df.loc[outlier_indices, target_col].tolist()
            outlier_stats['bounds'] = {'lower': lower_bound, 'upper': upper_bound}
    
    except Exception as e:
        st.warning(f"⚠️ Errore nel rilevamento outlier: {str(e)}")
    
    return outlier_stats

def create_outlier_boxplot(df: pd.DataFrame, target_col: str) -> go.Figure:
    """
    Crea un boxplot per visualizzare gli outlier
    
    Args:
        df: DataFrame
        target_col: Colonna target
        
    Returns:
        go.Figure: Grafico boxplot
    """
    try:
        values = pd.to_numeric(df[target_col], errors='coerce').dropna()
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=values,
            name=target_col,
            boxpoints='outliers',
            marker=dict(color='lightblue'),
            line=dict(color='darkblue')
        ))
        
        fig.update_layout(
            title=f"Distribuzione e Outlier - {target_col}",
            yaxis_title=target_col,
            height=400,
            showlegend=False
        )
        
        return fig
    
    except Exception as e:
        st.warning(f"⚠️ Errore nella creazione del boxplot: {str(e)}")
        return go.Figure()

def validate_data_quality(df: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, Any]:
    """
    Valida la qualità dei dati e restituisce warning/errori
    
    Args:
        df: DataFrame
        date_col: Colonna data
        target_col: Colonna target
        
    Returns:
        Dict: Risultati validazione
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'recommendations': []
    }
    
    try:
        # Controllo dimensioni dataset
        if len(df) < 20:
            validation['errors'].append("Dataset troppo piccolo (< 20 righe)")
            validation['is_valid'] = False
        elif len(df) < 50:
            validation['warnings'].append("Dataset piccolo (< 50 righe) - risultati potrebbero essere instabili")
        
        # Controllo colonna data
        if date_col:
            date_series = pd.to_datetime(df[date_col], errors='coerce')
            null_dates = date_series.isnull().sum()
            if null_dates > 0:
                validation['warnings'].append(f"{null_dates} date non valide trovate")
        
        # Controllo colonna target
        if target_col:
            target_series = pd.to_numeric(df[target_col], errors='coerce')
            null_targets = target_series.isnull().sum()
            zero_targets = (target_series == 0).sum()
            negative_targets = (target_series < 0).sum()
            
            if null_targets > len(df) * 0.3:
                validation['errors'].append(f"Troppi valori mancanti nel target ({null_targets}/{len(df)})")
                validation['is_valid'] = False
            elif null_targets > 0:
                validation['warnings'].append(f"{null_targets} valori mancanti nel target")
            
            if zero_targets > len(df) * 0.5:
                validation['warnings'].append(f"Molti valori zero nel target ({zero_targets}/{len(df)})")
            
            if negative_targets > 0:
                validation['warnings'].append(f"{negative_targets} valori negativi nel target")
            
            # Controllo varianza
            if target_series.var() == 0:
                validation['errors'].append("Target ha varianza zero - forecasting non possibile")
                validation['is_valid'] = False
            elif target_series.std() < target_series.mean() * 0.01:
                validation['warnings'].append("Target ha varianza molto bassa - forecasting potrebbe essere poco utile")
        
        # Raccomandazioni
        if len(validation['warnings']) > 0:
            validation['recommendations'].append("Considera di pulire i dati prima del forecasting")
        
        if len(df) > 1000:
            validation['recommendations'].append("Dataset grande - considera di aggregare temporalmente per migliorare performance")
    
    except Exception as e:
        validation['errors'].append(f"Errore nella validazione: {str(e)}")
        validation['is_valid'] = False
    
    return validation

def infer_seasonal_periods(df: pd.DataFrame, date_col: str, freq: str) -> int:
    """
    Inferisce automaticamente i periodi stagionali basandosi sulla frequenza
    
    Args:
        df: DataFrame
        date_col: Colonna data
        freq: Frequenza temporale
        
    Returns:
        int: Numero di periodi stagionali
    """
    try:
        # Usa il mapping di default
        if freq in SEASONAL_PERIODS_MAP:
            return SEASONAL_PERIODS_MAP[freq]
        
        # Se non mappato, prova a inferire dalla serie temporale
        if len(df) > 1 and date_col in df.columns:
            dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            if len(dates) > 1:
                inferred_freq = pd.infer_freq(dates)
                if inferred_freq:
                    if 'D' in inferred_freq:
                        return 7  # Settimanale per dati giornalieri
                    elif 'W' in inferred_freq:
                        return 52  # Annuale per dati settimanali
                    elif 'M' in inferred_freq:
                        return 12  # Annuale per dati mensili
                    elif 'Q' in inferred_freq:
                        return 4   # Annuale per dati trimestrali
        
        # Default fallback
        return 12
    
    except Exception:
        return 12

def clean_data(df: pd.DataFrame, cleaning_preferences: Dict[str, Any], target_col: str) -> pd.DataFrame:
    """
    Applica le preferenze di pulizia ai dati
    
    Args:
        df: DataFrame
        cleaning_preferences: Dizionario con preferenze di pulizia
        target_col: Colonna target
        
    Returns:
        pd.DataFrame: DataFrame pulito
    """
    df_clean = df.copy()
    
    try:
        # Rimuovi zeri se richiesto
        if cleaning_preferences.get('remove_zeros', False):
            df_clean = df_clean[pd.to_numeric(df_clean[target_col], errors='coerce') != 0]
        
        # Gestisci valori negativi
        if cleaning_preferences.get('remove_negatives', False):
            numeric_values = pd.to_numeric(df_clean[target_col], errors='coerce')
            df_clean[target_col] = numeric_values.clip(lower=0)
        
        # Sostituisci outlier
        if cleaning_preferences.get('replace_outliers', False):
            numeric_values = pd.to_numeric(df_clean[target_col], errors='coerce')
            Q1 = numeric_values.quantile(0.25)
            Q3 = numeric_values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Sostituisci outlier con mediana
            median_val = numeric_values.median()
            outlier_mask = (numeric_values < lower_bound) | (numeric_values > upper_bound)
            df_clean.loc[outlier_mask, target_col] = median_val
        
        # Gestisci valori mancanti
        if 'nan_handling' in cleaning_preferences:
            method = cleaning_preferences['nan_handling']
            df_clean = handle_missing_values(df_clean, target_col, method)
    
    except Exception as e:
        st.warning(f"⚠️ Errore nella pulizia dei dati: {str(e)}")
    
    return df_clean

def aggregate_data(df: pd.DataFrame, date_col: str, target_col: str, 
                  freq: str, aggregation_method: str) -> pd.DataFrame:
    """
    Aggrega i dati secondo la frequenza e metodo specificati
    
    Args:
        df: DataFrame
        date_col: Colonna data
        target_col: Colonna target
        freq: Frequenza di aggregazione
        aggregation_method: Metodo di aggregazione
        
    Returns:
        pd.DataFrame: DataFrame aggregato
    """
    try:
        df_agg = df.copy()
        
        # Converti colonna data
        if not pd.api.types.is_datetime64_any_dtype(df_agg[date_col]):
            df_agg[date_col] = pd.to_datetime(df_agg[date_col], errors='coerce')
        
        # Rimuovi righe con date non valide
        initial_len = len(df_agg)
        df_agg = df_agg.dropna(subset=[date_col])
        
        if len(df_agg) < initial_len:
            st.warning(f"⚠️ Rimosse {initial_len - len(df_agg)} righe con date non valide")
        
        if df_agg.empty:
            st.error("❌ Nessun dato valido rimasto dopo la pulizia delle date")
            return pd.DataFrame()
        
        # Aggrega
        df_agg = df_agg.set_index(date_col).resample(freq).agg({
            target_col: aggregation_method
        }).reset_index()
        
        return df_agg
    
    except Exception as e:
        st.error(f"❌ Errore durante l'aggregazione: {str(e)}")
        return pd.DataFrame()

def get_external_regressor_candidates(df: pd.DataFrame, date_col: str, target_col: str) -> List[str]:
    """
    Identifica colonne che potrebbero essere usate come regressori esterni
    
    Args:
        df: DataFrame
        date_col: Colonna data
        target_col: Colonna target
        
    Returns:
        List[str]: Lista nomi colonne candidate
    """
    candidates = []
    
    for col in df.columns:
        if col != date_col and col != target_col:
            # Controlla se è numerica o può essere convertita
            try:
                pd.to_numeric(df[col], errors='raise')
                candidates.append(col)
            except:
                # Controlla se è categorica con pochi valori (potenziale dummy)
                unique_vals = df[col].nunique()
                if unique_vals <= 10 and unique_vals > 1:
                    candidates.append(f"{col} (categorica)")
    
    return candidates

def get_missing_value_stats(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Calcola statistiche sui valori mancanti
    
    Returns:
        Dict: Statistiche missing values
    """
    missing_count = df[target_col].isna().sum()
    total_count = len(df)
    
    return {
        'count': missing_count,
        'percentage': (missing_count / total_count * 100) if total_count > 0 else 0,
        'total_records': total_count
    }

def get_regressor_candidates(df: pd.DataFrame, target_col: str, 
                           min_correlation: float = 0.1) -> List[Dict[str, Any]]:
    """
    Identifica automaticamente potenziali regressori basati sulla correlazione
    
    Args:
        df: DataFrame
        target_col: Colonna target
        min_correlation: Soglia minima di correlazione
        
    Returns:
        List[Dict]: Lista di candidati con metadati
    """
    candidates = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numeric_cols:
        if col != target_col:
            try:
                correlation = df[col].corr(df[target_col])
                if abs(correlation) >= min_correlation:
                    candidates.append({
                        'column': col,
                        'correlation': correlation,
                        'missing_pct': df[col].isna().sum() / len(df) * 100,
                        'variance': df[col].var()
                    })
            except:
                continue
    
    # Ordina per correlazione assoluta decrescente
    candidates.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return candidates

def handle_outliers_data(df: pd.DataFrame, target_col: str, method: str) -> pd.DataFrame:
    """
    Gestisce gli outlier secondo il metodo specificato
    
    Args:
        df: DataFrame
        target_col: Colonna target
        method: Metodo di gestione outlier
        
    Returns:
        pd.DataFrame: DataFrame con outlier gestiti
    """
    df_clean = df.copy()
    
    # Identifica outlier usando IQR
    Q1 = df_clean[target_col].quantile(0.25)
    Q3 = df_clean[target_col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (df_clean[target_col] < lower_bound) | (df_clean[target_col] > upper_bound)
    
    if method == "Replace with median":
        median_value = df_clean[target_col].median()
        df_clean.loc[outlier_mask, target_col] = median_value
    elif method == "Replace with mean":
        mean_value = df_clean[target_col].mean()
        df_clean.loc[outlier_mask, target_col] = mean_value
    elif method == "Remove outliers":
        df_clean = df_clean[~outlier_mask]
    elif method == "Winsorize (clip)":
        df_clean[target_col] = df_clean[target_col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean
