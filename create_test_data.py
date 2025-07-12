# Script per testare Prophet con dati reali CSV
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('/workspaces/CC-Excellence')

# Prima controlliamo se possiamo accedere al file CSV
csv_path = "/c:/Users/filippo.raimondi/OneDrive - Alkemy Spa/General - DentalPro/02. Data Model/04. Estrazioni SAS/DB_FORECAST.csv"

print("Trying to access CSV file...")
try:
    # Let's use a local sample first
    print("Creating sample data for testing...")
    
    # Create sample time series data similar to typical forecasting data
    dates = pd.date_range(start='2022-01-01', end='2024-06-30', freq='D')
    np.random.seed(42)
    
    # Simulate realistic business data with trend and seasonality
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    weekly = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    noise = np.random.normal(0, 5, len(dates))
    values = trend + seasonal + weekly + noise
    
    # Create DataFrame
    test_df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    print(f"✅ Sample data created: {len(test_df)} rows")
    print(f"Date range: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Value range: {test_df['value'].min():.2f} to {test_df['value'].max():.2f}")
    
    # Save sample data
    test_df.to_csv('/workspaces/CC-Excellence/test_data.csv', index=False)
    print("✅ Test data saved to test_data.csv")
    
except Exception as e:
    print(f"❌ Error creating test data: {e}")
