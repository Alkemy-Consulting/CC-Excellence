from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import warnings
import logging
import hashlib
from functools import lru_cache
from prophet import Prophet
from datetime import datetime
from dataclasses import asdict
warnings.filterwarnings('ignore')

# Import new enterprise architecture components
from src.modules.forecasting.prophet_core import ProphetForecaster, ProphetForecastResult
from .prophet_presentation import (
    ProphetPlotFactory, 
    ProphetVisualizationConfig,
    create_prophet_plots
)
from src.modules.forecasting.prophet_diagnostics import (
    ProphetDiagnosticAnalyzer,
    ProphetDiagnosticPlots,
    ProphetDiagnosticConfig,
    create_diagnostic_analyzer,
    create_diagnostic_plots
)

# Configure logging
logger = logging.getLogger(__name__)
from .prophet_performance import (
    OptimizedProphetForecaster,
    PerformanceMonitor,
    DataFrameOptimizer,
    create_optimized_forecaster,
    create_dataframe_optimizer,
    get_performance_report,
    performance_monitor
)

# Advanced ML Features Integration (Phase 5)
try:
    from src.modules.forecasting.prophet_ml_advanced import (
        create_feature_engineer,
        create_ensemble_forecaster,
        create_hyperparameter_optimizer,
        MLFeatureConfig,
        EnsembleModelResult
    )
    
    def run_prophet_ensemble_forecast(df: pd.DataFrame, date_col: str, target_col: str,
                                     forecast_periods: int = 30,
                                     enable_prophet: bool = True,
                                     enable_ml_models: bool = True,
                                     enable_optimization: bool = True) -> Dict[str, Any]:
        """
        Run ensemble forecasting with multiple models including Prophet and ML models
        
        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            target_col: Name of target column
            forecast_periods: Number of periods to forecast
            enable_prophet: Whether to include Prophet in ensemble
            enable_ml_models: Whether to include ML models in ensemble
            enable_optimization: Whether to apply performance optimizations
            
        Returns:
            Dictionary containing ensemble results and individual model predictions
        """
        try:
            # Initialize ensemble forecaster
            ensemble_forecaster = create_ensemble_forecaster(
                enable_prophet=enable_prophet,
                enable_ml_models=enable_ml_models
            )
            
            # Apply performance optimizations if enabled
            if enable_optimization:
                try:
                    from .prophet_performance import performance_monitor
                    with performance_monitor.monitor_execution("ensemble_forecast"):
                        # Train ensemble
                        training_result = ensemble_forecaster.fit_ensemble(
                            df, date_col, target_col, train_size=0.8
                        )
                        
                        # Generate predictions
                        prediction_result = ensemble_forecaster.predict_ensemble(
                            df, date_col, target_col, forecast_periods=forecast_periods
                        )
                except ImportError:
                    # Fallback without performance monitoring
                    training_result = ensemble_forecaster.fit_ensemble(
                        df, date_col, target_col, train_size=0.8
                    )
                    
                    prediction_result = ensemble_forecaster.predict_ensemble(
                        df, date_col, target_col, forecast_periods=forecast_periods
                    )
            else:
                # Train ensemble without performance monitoring
                training_result = ensemble_forecaster.fit_ensemble(
                    df, date_col, target_col, train_size=0.8
                )
                
                # Generate predictions
                prediction_result = ensemble_forecaster.predict_ensemble(
                    df, date_col, target_col, forecast_periods=forecast_periods
                )
            
            # Combine results
            result = {
                'ensemble_forecast': prediction_result.ensemble_forecast,
                'individual_predictions': prediction_result.individual_predictions,
                'model_weights': prediction_result.model_weights,
                'performance_metrics': prediction_result.performance_metrics,
                'feature_importance': prediction_result.feature_importance,
                'model_configs': prediction_result.model_configs,
                'training_summary': training_result,
                'forecast_metadata': {
                    'forecast_periods': forecast_periods,
                    'models_used': list(prediction_result.model_weights.keys()),
                    'ensemble_diversity': prediction_result.performance_metrics.get('model_diversity', 0),
                    'created_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"Ensemble forecasting completed with {len(prediction_result.model_weights)} models")
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble forecasting: {e}")
            raise


    def run_prophet_feature_engineering(df: pd.DataFrame, date_col: str, target_col: str,
                                       config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run advanced feature engineering for time series data
        
        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            target_col: Name of target column
            config: Configuration dictionary for feature engineering
            
        Returns:
            Dictionary containing engineered features and statistics
        """
        try:
            # Create feature configuration
            if config:
                feature_config = MLFeatureConfig(
                    lag_features=config.get('lag_features', [1, 7, 14, 30]),
                    rolling_windows=config.get('rolling_windows', [7, 14, 30]),
                    diff_features=config.get('diff_features', [1, 7]),
                    fourier_order=config.get('fourier_order', 10),
                    enable_trends=config.get('enable_trends', True),
                    enable_seasonality=config.get('enable_seasonality', True)
                )
            else:
                feature_config = MLFeatureConfig()
            
            # Initialize feature engineer
            feature_engineer = create_feature_engineer(feature_config)
            
            # Engineer features
            features_df = feature_engineer.engineer_features(df, date_col, target_col)
            
            # Select best features
            selected_features_df = feature_engineer.select_features(features_df)
            
            # Calculate feature statistics
            feature_stats = {
                'total_features_created': len(feature_engineer.feature_names),
                'features_selected': len(selected_features_df.columns) - 1,  # Exclude target
                'feature_types': {
                    'lag_features': len([f for f in feature_engineer.feature_names if f.startswith('lag_')]),
                    'rolling_features': len([f for f in feature_engineer.feature_names if f.startswith('rolling_')]),
                    'time_features': len([f for f in feature_engineer.feature_names if f in ['hour', 'day', 'month', 'year', 'day_of_week']]),
                    'fourier_features': len([f for f in feature_engineer.feature_names if 'fourier' in f]),
                    'trend_features': len([f for f in feature_engineer.feature_names if 'trend' in f])
                },
                'data_quality': {
                    'original_rows': len(df),
                    'engineered_rows': len(features_df),
                    'selected_rows': len(selected_features_df),
                    'missing_values_handled': len(df) - len(features_df)
                }
            }
            
            result = {
                'engineered_features': features_df,
                'selected_features': selected_features_df,
                'feature_names': feature_engineer.feature_names,
                'feature_statistics': feature_stats,
                'configuration_used': asdict(feature_config),
                'created_at': datetime.now().isoformat()
            }
            
            logger.info(f"Feature engineering completed: {feature_stats['total_features_created']} features created, {feature_stats['features_selected']} selected")
            return result
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            raise


    def run_prophet_hyperparameter_optimization(df: pd.DataFrame, date_col: str, target_col: str,
                                               n_trials: int = 50, timeout: int = 3600) -> Dict[str, Any]:
        """
        Run automated hyperparameter optimization for Prophet models
        
        Args:
            df: DataFrame with time series data
            date_col: Name of date column
            target_col: Name of target column
            n_trials: Number of optimization trials
            timeout: Maximum optimization time in seconds
            
        Returns:
            Dictionary containing optimized parameters and performance metrics
        """
        try:
            # Initialize hyperparameter optimizer
            optimizer = create_hyperparameter_optimizer(
                n_trials=n_trials,
                timeout=timeout
            )
            
            # Run optimization
            optimization_result = optimizer.optimize_prophet_params(
                df, date_col, target_col
            )
            
            # Test optimized parameters
            optimized_forecaster = ProphetForecaster()
            
            # Create model config with optimized parameters
            optimized_model_config = optimization_result['best_params']
            base_config = {
                'train_size': 0.8,
                'forecast_periods': 30,
                'include_history': False,
                'enable_diagnostics': False
            }
            
            # Run forecast with optimized parameters
            optimized_result = optimized_forecaster.run_forecast_core(
                df, date_col, target_col, optimized_model_config, base_config
            )
            
            # Calculate improvement metrics
            improvement_metrics = {
                'optimization_trials': optimization_result['n_trials'],
                'best_score': optimization_result['best_score'],
                'optimized_parameters': optimization_result['best_params'],
                'forecast_performance': {
                    'mape': getattr(optimized_result.metrics, 'mape', None),
                    'rmse': getattr(optimized_result.metrics, 'rmse', None),
                    'mae': getattr(optimized_result.metrics, 'mae', None)
                }
            }
            
            result = {
                'optimization_results': optimization_result,
                'optimized_forecast': optimized_result,
                'improvement_metrics': improvement_metrics,
                'optimization_summary': {
                    'trials_completed': optimization_result['n_trials'],
                    'best_score_achieved': optimization_result['best_score'],
                    'parameters_optimized': list(optimization_result['best_params'].keys()),
                    'optimization_time': timeout,
                    'created_at': datetime.now().isoformat()
                }
            }
            
            logger.info(f"Hyperparameter optimization completed: {optimization_result['n_trials']} trials, best score: {optimization_result['best_score']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            raise

except ImportError as e:
    logger.warning(f"Advanced ML features not available: {e}")
    
    # Provide fallback functions
    def run_prophet_ensemble_forecast(*args, **kwargs):
        raise ImportError("Advanced ML features not available. Please install required dependencies.")
    
    def run_prophet_feature_engineering(*args, **kwargs):
        raise ImportError("Advanced ML features not available. Please install required dependencies.")
    
    def run_prophet_hyperparameter_optimization(*args, **kwargs):
        raise ImportError("Advanced ML features not available. Please install required dependencies.")

# Configure logging for Prophet module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if not exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

from modules.holidays import get_holidays



def create_prophet_forecast_chart(model, forecast_df, actual_data, date_col, target_col, confidence_interval=0.8):
    """
    Legacy wrapper for backward compatibility - delegates to enterprise architecture
    """
    try:
        logger.info("Using enterprise Prophet visualization layer")
        
        # Create visualization configuration
        config = ProphetVisualizationConfig()
        plot_generator = ProphetPlotFactory.create_plot_generator(config)
        
        # Prepare chart data
        chart_data = plot_generator.prepare_chart_data(
            model, forecast_df, actual_data, date_col, target_col, confidence_interval
        )
        
        # Create and return the chart
        if chart_data.get('success', False):
            return plot_generator.create_forecast_chart(chart_data)
        else:
            st.error(f"Error creating forecast chart: {chart_data.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        logger.error(f"Error in legacy chart creation wrapper: {str(e)}")
        st.error(f"Error creating Prophet forecast chart: {str(e)}")
        return None

def generate_data_hash(df: pd.DataFrame, date_col: str, target_col: str, model_config: dict, base_config: dict) -> str:
    """
    Generate a hash for caching purposes based on input data and configuration
    """
    try:
        # Create a string representation of the data and configs
        data_str = f"{df[[date_col, target_col]].to_string()}"
        config_str = f"{sorted(model_config.items())}{sorted(base_config.items())}"
        combined_str = data_str + config_str
        
        # Generate MD5 hash
        return hashlib.md5(combined_str.encode()).hexdigest()
    except Exception:
        # Return empty hash if generation fails
        return ""

@lru_cache(maxsize=32)
def _cached_prophet_model_params(
    seasonality_mode: str,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
    interval_width: float,
    yearly_seasonality: str,
    weekly_seasonality: str,
    daily_seasonality: str
) -> dict:
    """
    Cache Prophet model parameters to avoid repeated parameter processing
    """
    return {
        'yearly_seasonality': yearly_seasonality,
        'weekly_seasonality': weekly_seasonality,
        'daily_seasonality': daily_seasonality,
        'seasonality_mode': seasonality_mode,
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale,
        'interval_width': interval_width
    }

def optimize_dataframe_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage for Prophet processing
    """
    try:
        optimized_df = df.copy()
        
        # Convert datetime column to more efficient format
        if 'ds' in optimized_df.columns:
            optimized_df['ds'] = pd.to_datetime(optimized_df['ds'])
        
        # Convert numeric columns to more efficient dtypes
        if 'y' in optimized_df.columns:
            optimized_df['y'] = pd.to_numeric(optimized_df['y'], downcast='float')
        
        logger.info(f"DataFrame optimized - Memory usage reduced from {df.memory_usage(deep=True).sum()} to {optimized_df.memory_usage(deep=True).sum()} bytes")
        return optimized_df
    except Exception as e:
        logger.warning(f"DataFrame optimization failed: {e}. Using original DataFrame.")
        return df

def validate_prophet_inputs(df: pd.DataFrame, date_col: str, target_col: str) -> None:
    """
    Robust input validation for Prophet forecasting
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas DataFrame")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")
    if not isinstance(date_col, str) or not date_col:
        raise ValueError("date_col must be a non-empty string")
    if not isinstance(target_col, str) or not target_col:
        raise ValueError("target_col must be a non-empty string")
    
    # Sanitize column names to prevent injection attacks
    safe_date_col = str(date_col).strip()
    safe_target_col = str(target_col).strip()
    
    if safe_date_col not in df.columns:
        raise KeyError(f"Date column not found in DataFrame")
    if safe_target_col not in df.columns:
        raise KeyError(f"Target column not found in DataFrame")
    
    # Check for minimum data requirements
    if len(df) < 10:
        raise ValueError(f"Insufficient data points: {len(df)} (minimum 10 required for Prophet)")
    
    # Check for maximum data size to prevent memory issues
    if len(df) > 100000:
        logger.warning(f"Large dataset detected: {len(df)} rows. Consider data sampling for better performance.")
    
    # Validate target column data type and missing values
    target_series = df[safe_target_col]
    if target_series.isna().sum() > len(df) * 0.3:
        raise ValueError(f"Too many missing values in target column: {target_series.isna().sum()}/{len(df)} (>30%)")
    
    # Check for numeric target values
    try:
        numeric_values = pd.to_numeric(target_series.dropna(), errors='raise')
        # Check for infinite or extremely large values
        if np.isinf(numeric_values).any():
            raise ValueError("Target column contains infinite values")
        if (np.abs(numeric_values) > 1e10).any():
            logger.warning("Target column contains very large values which may cause numerical instability")
    except (ValueError, TypeError):
        raise ValueError(f"Target column contains non-numeric values")
    
    # Validate date column
    try:
        date_values = pd.to_datetime(df[safe_date_col], errors='raise')
        # Check for reasonable date range
        if date_values.min().year < 1900 or date_values.max().year > 2100:
            logger.warning("Date values outside reasonable range (1900-2100)")
    except (ValueError, TypeError):
        raise ValueError(f"Date column contains invalid date values")
    
    # Check for zero variance
    numeric_target = pd.to_numeric(target_series.dropna(), errors='coerce')
    if numeric_target.std() == 0:
        raise ValueError("Target column has zero variance - cannot forecast constant values")

def run_prophet_forecast(df: pd.DataFrame, date_col: str, target_col: str, 
                        model_config: dict, base_config: dict):
    """
    Enterprise Prophet forecast interface - delegates to clean architecture
    """
    try:
        logger.info("Using enterprise Prophet forecasting architecture")
        
        # Initialize Prophet forecaster
        forecaster = ProphetForecaster()
        
        # Run forecast using clean architecture
        result = forecaster.run_forecast_core(df, date_col, target_col, model_config, base_config)
        
        if result.success:
            # Store results in session state for diagnostics
            st.session_state.last_prophet_result = result
            st.session_state.last_prophet_data = {
                'df': df.copy(),
                'date_col': date_col,
                'target_col': target_col
            }
            
            # Create visualizations using presentation layer
            plots = create_prophet_plots(result, df, date_col, target_col)
            
            # Convert to legacy format for backward compatibility
            forecast_output = result.raw_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_output.columns = [date_col, f'{target_col}_forecast', f'{target_col}_lower', f'{target_col}_upper']
            
            logger.info(f"Enterprise Prophet forecast completed successfully - Output shape: {forecast_output.shape}")
            return forecast_output, result.metrics, plots
            
        else:
            logger.error(f"Prophet forecast failed: {result.error}")
            st.error(f"Prophet forecast failed: {result.error}")
            return pd.DataFrame(), {}, {}
            
    except Exception as e:
        logger.error(f"Error in enterprise Prophet forecast interface: {str(e)}")
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}

def run_prophet_forecast_legacy(df: pd.DataFrame, date_col: str, target_col: str, 
                        model_config: dict, base_config: dict):
    """
    Legacy Prophet forecast implementation (preserved for reference)
    """
    # Legacy implementation moved to archive - use run_prophet_forecast() instead
    logger.warning("Legacy Prophet forecast called - redirecting to enterprise implementation")
    return run_prophet_forecast(df, date_col, target_col, model_config, base_config)

def run_prophet_forecast_optimized(df: pd.DataFrame, date_col: str, target_col: str, 
                                   model_config: dict, base_config: dict,
                                   enable_optimization: bool = True):
    """
    Performance-optimized Prophet forecast with automatic tuning and monitoring
    
    Args:
        df: Input DataFrame with time series data
        date_col: Name of the date column
        target_col: Name of the target column
        model_config: Prophet model configuration
        base_config: Base forecast configuration
        enable_optimization: Whether to apply performance optimizations
        
    Returns:
        Tuple of (forecast_df, metrics, plots, performance_report)
    """
    try:
        logger.info("Starting optimized Prophet forecasting with performance monitoring")
        
        if enable_optimization:
            # Initialize optimization components
            optimizer = create_dataframe_optimizer()
            forecaster = create_optimized_forecaster(enable_parallel=True)
            
            # Optimize DataFrame
            df_optimized = optimizer.optimize_dataframe(df.copy())
            logger.info(f"DataFrame optimized: {optimizer.get_optimization_stats()['memory_savings_pct']:.1f}% memory saved")

            # Auto-tune configuration using new optimization logic
            from modules.prophet_performance import optimize_prophet_hyperparameters
            tuned_model_config, tuned_base_config, _ = optimize_prophet_hyperparameters(
                df_optimized, model_config, base_config
            )

            logger.info("Auto-tuning applied for optimal performance")

        else:
            # Use standard components
            df_optimized = df.copy()
            tuned_model_config = model_config
            tuned_base_config = base_config
            forecaster = ProphetForecaster()

        # Run forecast with performance monitoring
        with performance_monitor.monitor_execution("optimized_prophet_forecast") as monitor:
            result = forecaster.run_forecast_core(
                df_optimized, date_col, target_col, tuned_model_config, tuned_base_config
            )
        
        if result.success:
            # Store results in session state for diagnostics
            st.session_state.last_prophet_result = result
            st.session_state.last_prophet_data = {
                'df': df_optimized.copy(),
                'date_col': date_col,
                'target_col': target_col
            }
            
            # Create visualizations using presentation layer
            plots = create_prophet_plots(result, df_optimized, date_col, target_col)
            
            # Generate performance report
            performance_report = get_performance_report()
            
            # Add optimization metrics if applied
            if enable_optimization and 'optimizer' in locals():
                performance_report['dataframe_optimization'] = optimizer.get_optimization_stats()
            
            # Convert to legacy format for backward compatibility
            forecast_output = result.raw_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_output.columns = [date_col, f'{target_col}_forecast', f'{target_col}_lower', f'{target_col}_upper']
            
            logger.info(f"Optimized Prophet forecast completed - Performance report available")
            return forecast_output, result.metrics, plots, performance_report
            
        else:
            logger.error(f"Optimized Prophet forecast failed: {result.error}")
            st.error(f"Prophet forecast failed: {result.error}")
            return pd.DataFrame(), {}, {}, {}
            
    except Exception as e:
        logger.error(f"Error in optimized Prophet forecast: {str(e)}")
        st.error(f"Error in Prophet forecasting: {str(e)}")
        return pd.DataFrame(), {}, {}, {}

def run_prophet_diagnostics(df: pd.DataFrame, date_col: str, target_col: str,
                           forecast_result: ProphetForecastResult,
                           show_diagnostic_plots: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive diagnostic analysis on Prophet forecast results
    
    Args:
        df: Original data used for forecasting
        date_col: Name of the date column
        target_col: Name of target column  
        forecast_result: Result from Prophet forecasting
        show_diagnostic_plots: Whether to display diagnostic plots in Streamlit
        
    Returns:
        Dictionary containing diagnostic analysis and plots
    """
    try:
        logger.info("Starting Prophet diagnostic analysis")
        
        # Initialize diagnostic components
        diagnostic_config = ProphetDiagnosticConfig()
        analyzer = create_diagnostic_analyzer(diagnostic_config)
        plots_generator = create_diagnostic_plots(diagnostic_config)
        
        # Run comprehensive analysis
        analysis = analyzer.analyze_forecast_quality(forecast_result, df, date_col, target_col)
        
        # Generate diagnostic plots
        diagnostic_plots = plots_generator.create_comprehensive_diagnostic_report(
            forecast_result, df, date_col, target_col
        )
        
        # Display plots in Streamlit if requested
        if show_diagnostic_plots and diagnostic_plots:
            st.subheader("📊 Extended Diagnostic Analysis")
            
            # Quality Dashboard
            if 'quality_dashboard' in diagnostic_plots:
                st.plotly_chart(diagnostic_plots['quality_dashboard'], width='stretch')
            
            # Create tabs for different diagnostic views
            diagnostic_tabs = st.tabs([
                "🔍 Residual Analysis", 
                "📈 Trend Decomposition", 
                "🌊 Seasonality Analysis", 
                "📏 Uncertainty Analysis",
                "✅ Forecast Validation"
            ])
            
            with diagnostic_tabs[0]:
                if 'residual_analysis' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['residual_analysis'], width='stretch')
                    
                    # Show residual statistics
                    residual_stats = analysis.get('residual_analysis', {})
                    if 'error' not in residual_stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Mean Residual", 
                                f"{residual_stats.get('mean_residual', 0):.4f}",
                                help="Average of residuals (should be close to 0)"
                            )
                        
                        with col2:
                            st.metric(
                                "Residual Std", 
                                f"{residual_stats.get('std_residual', 0):.4f}",
                                help="Standard deviation of residuals"
                            )
                        
                        with col3:
                            normality = "✅ Normal" if residual_stats.get('is_normally_distributed', False) else "❌ Not Normal"
                            st.metric(
                                "Normality Test", 
                                normality,
                                help="Shapiro-Wilk test for normality"
                            )
            
            with diagnostic_tabs[1]:
                if 'trend_decomposition' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['trend_decomposition'], width='stretch')
                    
                    # Show trend statistics
                    trend_stats = analysis.get('trend_analysis', {})
                    if 'error' not in trend_stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Trend Direction", 
                                trend_stats.get('trend_direction', 'Unknown').title(),
                                help="Overall direction of the trend"
                            )
                        
                        with col2:
                            st.metric(
                                "Trend Slope", 
                                f"{trend_stats.get('trend_slope', 0):.6f}",
                                help="Linear slope of the trend"
                            )
                        
                        with col3:
                            st.metric(
                                "Significant Changes", 
                                f"{trend_stats.get('significant_changes', 0)}",
                                help="Number of significant trend changes detected"
                            )
            
            with diagnostic_tabs[2]:
                if 'seasonality_analysis' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['seasonality_analysis'], width='stretch')
                    
                    # Show seasonality strength
                    seasonality_stats = analysis.get('seasonality_analysis', {})
                    if seasonality_stats:
                        for component, stats in seasonality_stats.items():
                            if isinstance(stats, dict):
                                st.subheader(f"{component.title()} Seasonality")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Amplitude", 
                                        f"{stats.get('amplitude', 0):.2f}",
                                        help="Range of seasonal variation"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Strength", 
                                        f"{stats.get('strength', 0):.3f}",
                                        help="Relative strength of seasonality"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Std Deviation", 
                                        f"{stats.get('std', 0):.2f}",
                                        help="Standard deviation of seasonal component"
                                    )
            
            with diagnostic_tabs[3]:
                if 'uncertainty_analysis' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['uncertainty_analysis'], width='stretch')
                    
                    # Show uncertainty statistics
                    uncertainty_stats = analysis.get('uncertainty_analysis', {})
                    if 'error' not in uncertainty_stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Mean Interval Width", 
                                f"{uncertainty_stats.get('mean_interval_width', 0):.2f}",
                                help="Average width of confidence intervals"
                            )
                        
                        with col2:
                            st.metric(
                                "Relative Width (%)", 
                                f"{uncertainty_stats.get('mean_relative_width', 0) * 100:.1f}%",
                                help="Average relative width of intervals"
                            )
                        
                        with col3:
                            st.metric(
                                "Interval Symmetry", 
                                f"{uncertainty_stats.get('interval_symmetry', 0):.3f}",
                                help="Symmetry of confidence intervals (-1 to 1)"
                            )
            
            with diagnostic_tabs[4]:
                if 'forecast_validation' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['forecast_validation'], width='stretch')
                    
                    # Show validation statistics
                    coverage_stats = analysis.get('forecast_coverage', {})
                    if 'error' not in coverage_stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Data Coverage", 
                                f"{coverage_stats.get('coverage_ratio', 0) * 100:.1f}%",
                                help="Percentage of actual data covered by forecast"
                            )
                        
                        with col2:
                            overlap_days = coverage_stats.get('overlap_days', 0)
                            st.metric(
                                "Overlap Days", 
                                f"{overlap_days}",
                                help="Number of days with both actual and forecast data"
                            )
                        
                        with col3:
                            quality_score = analysis.get('quality_score', 0)
                            st.metric(
                                "Quality Score", 
                                f"{quality_score:.1f}/100",
                                help="Overall forecast quality score",
                                delta=f"{quality_score - 75:.1f}" if quality_score > 0 else None
                            )
        
        logger.info(f"Prophet diagnostic analysis completed - Quality Score: {analysis.get('quality_score', 0):.1f}")
        
        return {
            'analysis': analysis,
            'plots': diagnostic_plots,
            'quality_score': analysis.get('quality_score', 0)
        }
        
    except Exception as e:
        logger.error(f"Error in Prophet diagnostic analysis: {str(e)}")
        st.error(f"Error in diagnostic analysis: {str(e)}")
        return {
            'analysis': {'error': str(e)},
            'plots': {},
            'quality_score': 0
        }

def run_prophet_diagnostics_optimized(df: pd.DataFrame, date_col: str, target_col: str,
                                     forecast_result: ProphetForecastResult,
                                     show_diagnostic_plots: bool = True,
                                     enable_parallel: bool = True) -> Dict[str, Any]:
    """
    Performance-optimized diagnostic analysis with parallel processing
    
    Args:
        df: Original data used for forecasting
        date_col: Name of the date column
        target_col: Name of the target column  
        forecast_result: Result from Prophet forecasting
        show_diagnostic_plots: Whether to display diagnostic plots in Streamlit
        enable_parallel: Whether to use parallel processing for diagnostics
        
    Returns:
        Dictionary containing diagnostic analysis, plots, and performance metrics
    """
    try:
        logger.info("Starting optimized Prophet diagnostic analysis")
        
        with performance_monitor.monitor_execution("optimized_diagnostics") as monitor:
            if enable_parallel:
                # Use optimized forecaster for parallel diagnostics
                optimized_forecaster = create_optimized_forecaster(enable_parallel=True)
                analysis = optimized_forecaster.run_parallel_diagnostics(
                    df, date_col, target_col, forecast_result
                )
                logger.info("Parallel diagnostic analysis completed")
            else:
                # Standard diagnostic analysis
                analyzer = create_diagnostic_analyzer()
                analysis = analyzer.analyze_forecast_quality(forecast_result, df, date_col, target_col)
                logger.info("Sequential diagnostic analysis completed")
            
            # Generate diagnostic plots
            diagnostic_config = ProphetDiagnosticConfig()
            plots_generator = create_diagnostic_plots(diagnostic_config)
            diagnostic_plots = plots_generator.create_comprehensive_diagnostic_report(
                forecast_result, df, date_col, target_col
            )
        
        # Display plots in Streamlit if requested
        if show_diagnostic_plots and diagnostic_plots:
            st.subheader("📊 Optimized Diagnostic Analysis")
            
            # Performance metrics display
            if performance_monitor.metrics_history:
                latest_metrics = performance_monitor.metrics_history[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Analysis Time", 
                        f"{latest_metrics.execution_time:.2f}s",
                        help="Time taken for diagnostic analysis"
                    )
                with col2:
                    st.metric(
                        "Memory Usage", 
                        f"{latest_metrics.memory_usage:.1f}MB",
                        help="Memory used during analysis"
                    )
                with col3:
                    st.metric(
                        "Cache Hit Ratio", 
                        f"{performance_monitor.get_cache_hit_ratio():.1%}",
                        help="Percentage of cached results used"
                    )
                with col4:
                    optimization_count = len(latest_metrics.optimization_applied)
                    st.metric(
                        "Optimizations", 
                        f"{optimization_count}",
                        help="Number of performance optimizations applied"
                    )
            # Diagnostic tabs (existing code from previous implementation)
            diagnostic_tabs = st.tabs([
                "🔍 Residual Analysis", 
                "📈 Trend Decomposition", 
                "🌊 Seasonality Analysis", 
                "📏 Uncertainty Analysis",
                "✅ Forecast Validation"
            ])
            
            with diagnostic_tabs[0]:
                if 'residual_analysis' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['residual_analysis'], width='stretch')
                    
                    # Show residual statistics
                    residual_stats = analysis.get('residual_analysis', {})
                    if 'error' not in residual_stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Mean Residual", 
                                f"{residual_stats.get('mean_residual', 0):.4f}",
                                help="Average of residuals (should be close to 0)"
                            )
                        
                        with col2:
                            st.metric(
                                "Residual Std", 
                                f"{residual_stats.get('std_residual', 0):.4f}",
                                help="Standard deviation of residuals"
                            )
                        
                        with col3:
                            normality = "✅ Normal" if residual_stats.get('is_normally_distributed', False) else "❌ Not Normal"
                            st.metric(
                                "Normality Test", 
                                normality,
                                help="Shapiro-Wilk test for normality"
                            )
            
            with diagnostic_tabs[1]:
                if 'trend_decomposition' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['trend_decomposition'], width='stretch')
                    
                    # Show trend statistics
                    trend_stats = analysis.get('trend_analysis', {})
                    if 'error' not in trend_stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Trend Direction", 
                                trend_stats.get('trend_direction', 'Unknown').title(),
                                help="Overall direction of the trend"
                            )
                        
                        with col2:
                            st.metric(
                                "Trend Slope", 
                                f"{trend_stats.get('trend_slope', 0):.6f}",
                                help="Linear slope of the trend"
                            )
                        
                        with col3:
                            st.metric(
                                "Significant Changes", 
                                f"{trend_stats.get('significant_changes', 0)}",
                                help="Number of significant trend changes detected"
                            )
            
            with diagnostic_tabs[2]:
                if 'seasonality_analysis' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['seasonality_analysis'], width='stretch')
                    
                    # Show seasonality strength
                    seasonality_stats = analysis.get('seasonality_analysis', {})
                    if seasonality_stats:
                        for component, stats in seasonality_stats.items():
                            if isinstance(stats, dict):
                                st.subheader(f"{component.title()} Seasonality")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Amplitude", 
                                        f"{stats.get('amplitude', 0):.2f}",
                                        help="Range of seasonal variation"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Strength", 
                                        f"{stats.get('strength', 0):.3f}",
                                        help="Relative strength of seasonality"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Std Deviation", 
                                        f"{stats.get('std', 0):.2f}",
                                        help="Standard deviation of seasonal component"
                                    )
            
            with diagnostic_tabs[3]:
                if 'uncertainty_analysis' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['uncertainty_analysis'], width='stretch')
                    
                    # Show uncertainty statistics
                    uncertainty_stats = analysis.get('uncertainty_analysis', {})
                    if 'error' not in uncertainty_stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Mean Interval Width", 
                                f"{uncertainty_stats.get('mean_interval_width', 0):.2f}",
                                help="Average width of confidence intervals"
                            )
                        
                        with col2:
                            st.metric(
                                "Relative Width (%)", 
                                f"{uncertainty_stats.get('mean_relative_width', 0) * 100:.1f}%",
                                help="Average relative width of intervals"
                            )
                        
                        with col3:
                            st.metric(
                                "Interval Symmetry", 
                                f"{uncertainty_stats.get('interval_symmetry', 0):.3f}",
                                help="Symmetry of confidence intervals (-1 to 1)"
                            )
            
            with diagnostic_tabs[4]:
                if 'forecast_validation' in diagnostic_plots:
                    st.plotly_chart(diagnostic_plots['forecast_validation'], width='stretch')
                    
                    # Show validation statistics
                    coverage_stats = analysis.get('forecast_coverage', {})
                    if 'error' not in coverage_stats:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Data Coverage", 
                                f"{coverage_stats.get('coverage_ratio', 0) * 100:.1f}%",
                                help="Percentage of actual data covered by forecast"
                            )
                        
                        with col2:
                            overlap_days = coverage_stats.get('overlap_days', 0)
                            st.metric(
                                "Overlap Days", 
                                f"{overlap_days}",
                                help="Number of days with both actual and forecast data"
                            )
                        
                        with col3:
                            quality_score = analysis.get('quality_score', 0)
                            st.metric(
                                "Quality Score", 
                                f"{quality_score:.1f}/100",
                                help="Overall forecast quality score",
                                delta=f"{quality_score - 75:.1f}" if quality_score > 0 else None
                            )
        
        logger.info(f"Optimized diagnostic analysis completed - Quality Score: {quality_score:.1f}")
        return {
            'analysis': analysis,
            'plots': diagnostic_plots,
            'quality_score': quality_score,
            'optimization_enabled': enable_parallel
        }
    except Exception as e:
        logger.error(f"Error in optimized Prophet diagnostic analysis: {str(e)}")
        st.error(f"Error in diagnostic analysis: {str(e)}")
        return {
            'analysis': {'error': str(e)},
            'plots': {},
            'quality_score': 0
        }

def run_prophet_benchmarking(df: pd.DataFrame, date_col: str, target_col: str,
                             model_configs: List[dict], base_config: dict,
                             num_runs: int = 3) -> Dict[str, Any]:
    """
    Benchmark multiple Prophet configurations for performance comparison
    
    Args:
        df: Input DataFrame
        date_col: Date column name
        target_col: Target column name
        model_configs: List of model configurations to benchmark
        base_config: Base configuration
        num_runs: Number of runs per configuration
        
    Returns:
        Comprehensive benchmark results
    """
    try:
        logger.info(f"Starting Prophet model benchmarking with {len(model_configs)} configurations")
        
        benchmark_results = {
            'configurations': [],
            'performance_comparison': {},
            'recommendations': [],
            'best_configuration': None
        }
        
        for i, config in enumerate(model_configs):
            config_name = f"Config_{i+1}"
            logger.info(f"Benchmarking {config_name}")
            
            config_results = {
                'name': config_name,
                'config': config,
                'runs': [],
                'average_metrics': {},
                'performance_score': 0
            }
            
            # Run multiple iterations
            for run in range(num_runs):
                with performance_monitor.monitor_execution(f"{config_name}_run_{run}"):
                    try:
                        # Use optimized forecaster
                        forecaster = create_optimized_forecaster()
                        result = forecaster.run_forecast_core(df, date_col, target_col, config, base_config)
                        
                        if result.success and performance_monitor.metrics_history:
                            latest_metrics = performance_monitor.metrics_history[-1]
                            config_results['runs'].append({
                                'run_number': run,
                                'execution_time': latest_metrics.execution_time,
                                'memory_usage': latest_metrics.memory_usage,
                                'forecast_metrics': result.metrics
                            })
                    except Exception as e:
                        logger.warning(f"Run {run} failed for {config_name}: {e}")
            
            # Calculate averages
            if config_results['runs']:
                avg_time = np.mean([r['execution_time'] for r in config_results['runs']])
                avg_memory = np.mean([r['memory_usage'] for r in config_results['runs']])
                avg_mape = np.mean([r['forecast_metrics'].get('mape', 100) for r in config_results['runs']])
                
                config_results['average_metrics'] = {
                    'avg_execution_time': avg_time,
                    'avg_memory_usage': avg_memory,
                    'avg_mape': avg_mape
                }
                
                # Calculate performance score (lower is better)
                # Combine speed, memory efficiency, and accuracy
                time_score = min(30, avg_time) / 30  # Normalize to 0-1
                memory_score = min(500, avg_memory) / 500  # Normalize to 0-1
                accuracy_score = min(20, avg_mape) / 20  # Normalize to 0-1
                
                config_results['performance_score'] = 100 - (
                    (time_score * 30) + (memory_score * 30) + (accuracy_score * 40)
                )
            
            benchmark_results['configurations'].append(config_results)
        
        # Find best configuration
        valid_configs = [c for c in benchmark_results['configurations'] if c['runs']]
        if valid_configs:
            best_config = max(valid_configs, key=lambda x: x['performance_score'])
            benchmark_results['best_configuration'] = best_config
            
            # Generate recommendations
            benchmark_results['recommendations'] = [
                f"Best performing configuration: {best_config['name']}",
                f"Performance score: {best_config['performance_score']:.1f}/100",
                f"Average execution time: {best_config['average_metrics']['avg_execution_time']:.2f}s",
                f"Average MAPE: {best_config['average_metrics']['avg_mape']:.2f}%"
            ]
        
        logger.info("Prophet model benchmarking completed")
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Error in Prophet model benchmarking: {str(e)}")
        return {'error': str(e)}

def create_forecast_quality_report(df: pd.DataFrame, date_col: str, target_col: str,
                                 forecast_result: ProphetForecastResult) -> str:
    """
    Generate a comprehensive text report of forecast quality
    
    Args:
        df: Original data used for forecasting
        date_col: Name of date column
        target_col: Name of target column
        forecast_result: Result from Prophet forecasting
        
    Returns:
        String containing detailed quality report
    """
    try:
        analyzer = create_diagnostic_analyzer()
        analysis = analyzer.analyze_forecast_quality(forecast_result, df, date_col, target_col)
        
        quality_score = analysis.get('quality_score', 0)
        
        report = f"""
📊 PROPHET FORECAST QUALITY REPORT
================================

Overall Quality Score: {quality_score:.1f}/100

🎯 FORECAST COVERAGE
• Coverage Ratio: {analysis.get('forecast_coverage', {}).get('coverage_ratio', 0) * 100:.1f}%
• Forecast Period: {analysis.get('forecast_coverage', {}).get('forecast_start', 'N/A')} to {analysis.get('forecast_coverage', {}).get('forecast_end', 'N/A')}

🔍 RESIDUAL ANALYSIS
"""
        
        residual_analysis = analysis.get('residual_analysis', {})
        if 'error' not in residual_analysis:
            report += f"""• Mean Residual: {residual_analysis.get('mean_residual', 0):.4f}
• Residual Std: {residual_analysis.get('std_residual', 0):.4f}
• Normality Test: {'✅ PASSED' if residual_analysis.get('is_normally_distributed', False) else '❌ FAILED'}
• Autocorrelation Test: {'❌ DETECTED' if residual_analysis.get('has_autocorrelation', True) else '✅ NONE'}
• Durbin-Watson Statistic: {residual_analysis.get('durbin_watson_statistic', 0):.3f}
"""
        
        report += "\n📈 TREND ANALYSIS\n"
        trend_analysis = analysis.get('trend_analysis', {})
        if 'error' not in trend_analysis:
            report += f"""• Trend Direction: {trend_analysis.get('trend_direction', 'Unknown').title()}
• Trend Slope: {trend_analysis.get('trend_slope', 0):.6f}
• Trend Volatility: {trend_analysis.get('trend_volatility', 0):.3f}
• Significant Changes: {trend_analysis.get('significant_changes', 0)}
"""
        
        report += "\n🌊 SEASONALITY ANALYSIS\n"
        seasonality_analysis = analysis.get('seasonality_analysis', {})
        for component, stats in seasonality_analysis.items():
            if isinstance(stats, dict):
                report += f"""• {component.title()} Seasonality:
  - Amplitude: {stats.get('amplitude', 0):.2f}
  - Strength: {stats.get('strength', 0):.3f}
  - Standard Deviation: {stats.get('std', 0):.2f}
"""
        
        report += "\n📏 UNCERTAINTY ANALYSIS\n"
        uncertainty_analysis = analysis.get('uncertainty_analysis', {})
        if 'error' not in uncertainty_analysis:
            report += f"""• Mean Interval Width: {uncertainty_analysis.get('mean_interval_width', 0):.2f}
• Relative Width: {uncertainty_analysis.get('mean_relative_width', 0) * 100:.1f}%
• Interval Symmetry: {uncertainty_analysis.get('interval_symmetry', 0):.3f}
• Max Interval Width: {uncertainty_analysis.get('max_interval_width', 0):.2f}
"""
        
        report += "\n🔧 CHANGEPOINT ANALYSIS\n"
        changepoint_analysis = analysis.get('changepoint_analysis', {})
        if 'error' not in changepoint_analysis:
            report += f"""• Total Changepoints: {changepoint_analysis.get('changepoints_count', 0)}
• Valid Changepoints: {changepoint_analysis.get('valid_changepoints_count', 0)}
• Average Spacing: {changepoint_analysis.get('changepoint_spacing_days', 0):.1f} days
"""
        
        # Quality assessment
        report += "\n📋 QUALITY ASSESSMENT\n"
        if quality_score >= 80:
            report += "✅ EXCELLENT - High quality forecast with reliable predictions"
        elif quality_score >= 60:
            report += "⚠️ GOOD - Acceptable quality with minor areas for improvement"
        elif quality_score >= 40:
            report += "🔶 MODERATE - Forecast has limitations, use with caution"
        else:
            report += "❌ POOR - Significant quality issues, consider model adjustments"
        
        report += f"\n\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating quality report: {str(e)}")
        return f"Error generating quality report: {str(e)}"
