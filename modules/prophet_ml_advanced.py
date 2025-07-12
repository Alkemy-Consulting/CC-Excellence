"""
Advanced Machine Learning Features for Prophet Forecasting
PHASE 5: Ensemble methods, feature engineering, and model selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from prophet import Prophet
import logging

from .prophet_core import ProphetForecaster, ProphetForecastResult
from .prophet_performance import PerformanceMonitor, performance_optimized

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class MLFeatureConfig:
    """Configuration for ML feature engineering"""
    lag_features: List[int] = None  # [1, 7, 14, 30]
    rolling_windows: List[int] = None  # [7, 14, 30]
    diff_features: List[int] = None  # [1, 7]
    fourier_order: int = 10
    enable_trends: bool = True
    enable_seasonality: bool = True
    enable_holidays: bool = True
    enable_external_regressors: bool = False
    
    def __post_init__(self):
        if self.lag_features is None:
            self.lag_features = [1, 7, 14, 30]
        if self.rolling_windows is None:
            self.rolling_windows = [7, 14, 30]
        if self.diff_features is None:
            self.diff_features = [1, 7]

@dataclass
class EnsembleModelResult:
    """Result from ensemble model prediction"""
    ensemble_forecast: pd.DataFrame
    individual_predictions: Dict[str, pd.DataFrame]
    model_weights: Dict[str, float]
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, np.ndarray]
    model_configs: Dict[str, Dict]

class AdvancedFeatureEngineer:
    """Advanced feature engineering for time series forecasting"""
    
    def __init__(self, config: MLFeatureConfig = None):
        self.config = config or MLFeatureConfig()
        self.feature_names = []
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=20)
        
    def engineer_features(self, df: pd.DataFrame, date_col: str, 
                         target_col: str) -> pd.DataFrame:
        """Create advanced features for time series modeling"""
        
        # Ensure proper datetime index
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col])
        df_work = df_work.set_index(date_col).sort_index()
        
        features_df = pd.DataFrame(index=df_work.index)
        features_df['target'] = df_work[target_col]
        
        # Lag features
        for lag in self.config.lag_features:
            features_df[f'lag_{lag}'] = df_work[target_col].shift(lag)
            
        # Rolling window features
        for window in self.config.rolling_windows:
            features_df[f'rolling_mean_{window}'] = df_work[target_col].rolling(window).mean()
            features_df[f'rolling_std_{window}'] = df_work[target_col].rolling(window).std()
            features_df[f'rolling_min_{window}'] = df_work[target_col].rolling(window).min()
            features_df[f'rolling_max_{window}'] = df_work[target_col].rolling(window).max()
            
        # Difference features
        for diff in self.config.diff_features:
            features_df[f'diff_{diff}'] = df_work[target_col].diff(diff)
            
        # Time-based features
        features_df['hour'] = features_df.index.hour
        features_df['day'] = features_df.index.day
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['year'] = features_df.index.year
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['day_of_year'] = features_df.index.dayofyear
        features_df['is_weekend'] = (features_df.index.dayofweek >= 5).astype(int)
        
        # Fourier features for seasonality
        if self.config.fourier_order > 0:
            for i in range(1, self.config.fourier_order + 1):
                features_df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * features_df.index.dayofyear / 365.25)
                features_df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * features_df.index.dayofyear / 365.25)
                
        # Trend features
        if self.config.enable_trends:
            features_df['linear_trend'] = np.arange(len(features_df))
            features_df['quadratic_trend'] = features_df['linear_trend'] ** 2
            
        # Technical indicators
        features_df['target_momentum'] = features_df['target'].pct_change()
        features_df['target_acceleration'] = features_df['target_momentum'].diff()
        
        # Volatility features
        features_df['volatility_7'] = features_df['target'].rolling(7).std()
        features_df['volatility_30'] = features_df['target'].rolling(30).std()
        
        # Remove rows with NaN values
        features_df = features_df.dropna()
        
        # Store feature names
        self.feature_names = [col for col in features_df.columns if col != 'target']
        
        logger.info(f"Feature engineering completed: {len(self.feature_names)} features created")
        
        return features_df
    
    def select_features(self, features_df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """Select best features using statistical methods"""
        
        feature_cols = [col for col in features_df.columns if col != target_col]
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Remove constant features
        constant_features = X.columns[X.var() == 0]
        X = X.drop(columns=constant_features)
        
        # Select best features
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_feature_names = X.columns[self.feature_selector.get_support()]
        
        result_df = pd.DataFrame(X_selected, 
                               columns=selected_feature_names, 
                               index=features_df.index)
        result_df[target_col] = y
        
        logger.info(f"Feature selection completed: {len(selected_feature_names)} features selected")
        
        return result_df

class EnsembleForecaster:
    """Ensemble forecasting with multiple models"""
    
    def __init__(self, enable_prophet: bool = True, enable_ml_models: bool = True):
        self.enable_prophet = enable_prophet
        self.enable_ml_models = enable_ml_models
        self.models = {}
        self.model_weights = {}
        self.feature_engineer = AdvancedFeatureEngineer()
        self.performance_monitor = PerformanceMonitor()
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize ensemble models"""
        models = {}
        
        if self.enable_prophet:
            models['prophet'] = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
        if self.enable_ml_models:
            models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            models['linear_regression'] = LinearRegression()
            models['ridge'] = Ridge(alpha=1.0)
            models['lasso'] = Lasso(alpha=0.1)
            
        self.models = models
        logger.info(f"Initialized {len(models)} models for ensemble")
        
        return models
    
    @performance_optimized
    def fit_ensemble(self, df: pd.DataFrame, date_col: str, target_col: str,
                    train_size: float = 0.8) -> Dict[str, Any]:
        """Fit ensemble models on training data"""
        
        with self.performance_monitor.monitor_execution("ensemble_training"):
            
            # Initialize models
            self.initialize_models()
            
            # Prepare data
            df_work = df.copy()
            df_work[date_col] = pd.to_datetime(df_work[date_col])
            df_work = df_work.sort_values(date_col)
            
            # Split data
            split_idx = int(len(df_work) * train_size)
            train_df = df_work.iloc[:split_idx]
            val_df = df_work.iloc[split_idx:]
            
            # Engineer features for ML models
            if self.enable_ml_models:
                features_df = self.feature_engineer.engineer_features(train_df, date_col, target_col)
                features_df = self.feature_engineer.select_features(features_df)
                
                feature_cols = [col for col in features_df.columns if col != 'target']
                X_train = features_df[feature_cols]
                y_train = features_df['target']
                
                # Scale features
                X_train_scaled = self.feature_engineer.scaler.fit_transform(X_train)
            
            # Train models
            training_results = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'prophet':
                        # Train Prophet
                        prophet_df = train_df[[date_col, target_col]].rename(
                            columns={date_col: 'ds', target_col: 'y'}
                        )
                        model.fit(prophet_df)
                        training_results[model_name] = {'status': 'success', 'type': 'prophet'}
                        
                    else:
                        # Train ML models
                        model.fit(X_train_scaled, y_train)
                        training_results[model_name] = {'status': 'success', 'type': 'ml'}
                        
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    training_results[model_name] = {'status': 'error', 'error': str(e)}
            
            # Validate models and calculate weights
            model_scores = self._validate_models(val_df, date_col, target_col)
            self.model_weights = self._calculate_weights(model_scores)
            
            logger.info(f"Ensemble training completed: {len(training_results)} models")
            logger.info(f"Model weights: {self.model_weights}")
            
            return {
                'training_results': training_results,
                'model_weights': self.model_weights,
                'validation_scores': model_scores
            }
    
    def _validate_models(self, val_df: pd.DataFrame, date_col: str, 
                        target_col: str) -> Dict[str, float]:
        """Validate models on validation set"""
        scores = {}
        
        # Engineer features for validation
        if self.enable_ml_models:
            val_features = self.feature_engineer.engineer_features(val_df, date_col, target_col)
            val_features = self.feature_engineer.select_features(val_features)
            
            feature_cols = [col for col in val_features.columns if col != 'target']
            X_val = val_features[feature_cols]
            y_val = val_features['target']
            X_val_scaled = self.feature_engineer.scaler.transform(X_val)
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'prophet':
                    # Predict with Prophet
                    future = model.make_future_dataframe(periods=len(val_df))
                    forecast = model.predict(future)
                    val_pred = forecast['yhat'].iloc[-len(val_df):].values
                    val_true = val_df[target_col].values
                    
                else:
                    # Predict with ML models
                    val_pred = model.predict(X_val_scaled)
                    val_true = y_val.values
                
                # Calculate RMSE score
                rmse = np.sqrt(mean_squared_error(val_true, val_pred))
                scores[model_name] = rmse
                
            except Exception as e:
                logger.error(f"Error validating {model_name}: {e}")
                scores[model_name] = float('inf')  # Worst possible score
        
        return scores
    
    def _calculate_weights(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate ensemble weights based on validation scores"""
        # Inverse of RMSE scores (lower RMSE = higher weight)
        valid_scores = {k: v for k, v in scores.items() if v != float('inf')}
        
        if not valid_scores:
            return {}
        
        # Calculate inverse scores
        inverse_scores = {k: 1.0 / v for k, v in valid_scores.items()}
        total_inverse = sum(inverse_scores.values())
        
        # Normalize to get weights
        weights = {k: v / total_inverse for k, v in inverse_scores.items()}
        
        return weights
    
    @performance_optimized
    def predict_ensemble(self, df: pd.DataFrame, date_col: str, target_col: str,
                        forecast_periods: int = 30) -> EnsembleModelResult:
        """Generate ensemble predictions"""
        
        with self.performance_monitor.monitor_execution("ensemble_prediction"):
            
            individual_predictions = {}
            
            # Generate predictions from each model
            for model_name, model in self.models.items():
                try:
                    if model_name == 'prophet':
                        # Prophet prediction
                        prophet_df = df[[date_col, target_col]].rename(
                            columns={date_col: 'ds', target_col: 'y'}
                        )
                        future = model.make_future_dataframe(periods=forecast_periods)
                        forecast = model.predict(future)
                        individual_predictions[model_name] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                        
                    else:
                        # ML model prediction
                        # Create future features
                        extended_df = self._extend_dataframe(df, date_col, forecast_periods)
                        features_df = self.feature_engineer.engineer_features(extended_df, date_col, target_col)
                        
                        feature_cols = [col for col in features_df.columns if col != 'target']
                        X_future = features_df[feature_cols].iloc[-forecast_periods:]
                        X_future_scaled = self.feature_engineer.scaler.transform(X_future)
                        
                        predictions = model.predict(X_future_scaled)
                        
                        # Create prediction dataframe
                        future_dates = pd.date_range(
                            start=df[date_col].max() + pd.Timedelta(days=1),
                            periods=forecast_periods,
                            freq='D'
                        )
                        
                        pred_df = pd.DataFrame({
                            'ds': future_dates,
                            'yhat': predictions,
                            'yhat_lower': predictions * 0.95,  # Simple confidence interval
                            'yhat_upper': predictions * 1.05
                        })
                        
                        individual_predictions[model_name] = pred_df
                        
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
                    continue
            
            # Calculate ensemble forecast
            ensemble_forecast = self._calculate_ensemble_forecast(individual_predictions)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_ensemble_metrics(df, date_col, target_col)
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            result = EnsembleModelResult(
                ensemble_forecast=ensemble_forecast,
                individual_predictions=individual_predictions,
                model_weights=self.model_weights,
                performance_metrics=performance_metrics,
                feature_importance=feature_importance,
                model_configs={name: self._get_model_config(name, model) 
                             for name, model in self.models.items()}
            )
            
            logger.info(f"Ensemble prediction completed: {len(individual_predictions)} models")
            
            return result
    
    def _extend_dataframe(self, df: pd.DataFrame, date_col: str, periods: int) -> pd.DataFrame:
        """Extend dataframe with future dates for feature engineering"""
        last_date = df[date_col].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        # Create future dataframe with NaN values for target
        future_df = pd.DataFrame({date_col: future_dates})
        for col in df.columns:
            if col != date_col:
                future_df[col] = np.nan
        
        return pd.concat([df, future_df], ignore_index=True)
    
    def _calculate_ensemble_forecast(self, individual_predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate weighted ensemble forecast"""
        if not individual_predictions:
            return pd.DataFrame()
        
        # Initialize ensemble forecast with the first prediction structure
        first_pred = list(individual_predictions.values())[0]
        ensemble_forecast = first_pred.copy()
        ensemble_forecast['yhat'] = 0.0
        ensemble_forecast['yhat_lower'] = 0.0
        ensemble_forecast['yhat_upper'] = 0.0
        
        # Calculate weighted average
        for model_name, prediction in individual_predictions.items():
            weight = self.model_weights.get(model_name, 0.0)
            if weight > 0:
                ensemble_forecast['yhat'] += weight * prediction['yhat']
                ensemble_forecast['yhat_lower'] += weight * prediction['yhat_lower']
                ensemble_forecast['yhat_upper'] += weight * prediction['yhat_upper']
        
        return ensemble_forecast
    
    def _calculate_ensemble_metrics(self, df: pd.DataFrame, date_col: str, target_col: str) -> Dict[str, float]:
        """Calculate ensemble performance metrics"""
        # This would typically involve backtesting
        # For now, return placeholder metrics
        return {
            'ensemble_rmse': 0.0,
            'ensemble_mae': 0.0,
            'ensemble_r2': 0.0,
            'model_diversity': len(self.models),
            'weight_entropy': self._calculate_weight_entropy()
        }
    
    def _calculate_weight_entropy(self) -> float:
        """Calculate entropy of model weights (higher = more diverse)"""
        weights = list(self.model_weights.values())
        if not weights:
            return 0.0
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        
        return entropy
    
    def _get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from ML models"""
        importance = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance[model_name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance[model_name] = np.abs(model.coef_)
        
        return importance
    
    def _get_model_config(self, model_name: str, model: Any) -> Dict[str, Any]:
        """Get model configuration"""
        if model_name == 'prophet':
            return {
                'yearly_seasonality': model.yearly_seasonality,
                'weekly_seasonality': model.weekly_seasonality,
                'daily_seasonality': model.daily_seasonality,
                'changepoint_prior_scale': model.changepoint_prior_scale,
                'seasonality_prior_scale': model.seasonality_prior_scale
            }
        else:
            return getattr(model, 'get_params', lambda: {})()

class HyperparameterOptimizer:
    """Automated hyperparameter optimization using Optuna"""
    
    def __init__(self, n_trials: int = 50, timeout: int = 3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        self.best_params = {}
        
    def optimize_prophet_params(self, df: pd.DataFrame, date_col: str, 
                              target_col: str) -> Dict[str, Any]:
        """Optimize Prophet hyperparameters"""
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
                'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 100, log=True),
                'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 100, log=True),
                'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
                'yearly_seasonality': trial.suggest_categorical('yearly_seasonality', [True, False]),
                'weekly_seasonality': trial.suggest_categorical('weekly_seasonality', [True, False])
            }
            
            # Cross-validation with time series split
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(df):
                train_df = df.iloc[train_idx]
                val_df = df.iloc[val_idx]
                
                try:
                    # Train Prophet with trial parameters
                    model = Prophet(**params)
                    prophet_df = train_df[[date_col, target_col]].rename(
                        columns={date_col: 'ds', target_col: 'y'}
                    )
                    model.fit(prophet_df)
                    
                    # Predict on validation set
                    future = model.make_future_dataframe(periods=len(val_df))
                    forecast = model.predict(future)
                    
                    # Calculate RMSE
                    val_pred = forecast['yhat'].iloc[-len(val_df):].values
                    val_true = val_df[target_col].values
                    rmse = np.sqrt(mean_squared_error(val_true, val_pred))
                    scores.append(rmse)
                    
                except Exception as e:
                    logger.warning(f"Trial failed: {e}")
                    scores.append(float('inf'))
            
            return np.mean(scores)
        
        # Create and run study
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        
        logger.info(f"Hyperparameter optimization completed:")
        logger.info(f"  Best score: {self.study.best_value:.4f}")
        logger.info(f"  Best params: {self.best_params}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study': self.study
        }

# Factory functions
def create_feature_engineer(config: MLFeatureConfig = None) -> AdvancedFeatureEngineer:
    """Create feature engineer instance"""
    return AdvancedFeatureEngineer(config)

def create_ensemble_forecaster(enable_prophet: bool = True, 
                             enable_ml_models: bool = True) -> EnsembleForecaster:
    """Create ensemble forecaster instance"""
    return EnsembleForecaster(enable_prophet, enable_ml_models)

def create_hyperparameter_optimizer(n_trials: int = 50, 
                                   timeout: int = 3600) -> HyperparameterOptimizer:
    """Create hyperparameter optimizer instance"""
    return HyperparameterOptimizer(n_trials, timeout)
