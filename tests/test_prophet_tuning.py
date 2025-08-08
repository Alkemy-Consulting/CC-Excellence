import sys
import os
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ProphetTuningTest")

# Step 1: Load the dataset
file_path = "/home/filipporaimondi/Projects/CC-Excellence/data/DB_FORECAST.csv"
df = pd.read_csv(file_path)

# Log dataset info
logger.debug(f"Dataset loaded with {len(df)} rows and columns: {df.columns.tolist()}")

# Prepare data for Prophet
df = df.rename(columns={"DATA_CHIAMATA": "ds", "TOTALE_CHIAMATE_RICEVUTE": "y"})
df["ds"] = pd.to_datetime(df["ds"])

# Step 2: Define initial model configuration
initial_model_config = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 10.0,
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "seasonality_mode": "additive",
    "interval_width": 0.8,
}

base_config = {
    "forecast_periods": 30,
    "train_size": 0.8,
}

# Log initial configuration
logger.debug(f"Initial Model Configuration: {initial_model_config}")

# Step 3: Run auto-tuning with detailed logging
from modules.prophet_performance import optimize_prophet_hyperparameters

optimized_model_config, optimized_base_config, optimization_metrics = optimize_prophet_hyperparameters(
    df, initial_model_config, base_config
)
logger.debug("Auto-tuning completed successfully.")

# Log optimized configuration
logger.debug(f"Optimized Model Configuration: {optimized_model_config}")

# Log optimization metrics
logger.debug(f"Optimization Metrics: {optimization_metrics}")

# Step 4: Compare configurations
print("Initial Model Configuration:")
print(initial_model_config)
print("\nOptimized Model Configuration:")
print(optimized_model_config)

# Step 5: Display optimization metrics
print("\nOptimization Metrics:")
print(optimization_metrics)
