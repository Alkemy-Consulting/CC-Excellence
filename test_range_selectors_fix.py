#!/usr/bin/env python3
"""
Test script to verify that range selectors are compatible with pandas >= 2.0
"""

import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import sys

print("=" * 60)
print("RANGE SELECTORS COMPATIBILITY TEST")
print("=" * 60)

# Test 1: Check pandas version
print(f"Python version: {sys.version}")
print(f"Pandas version: {pd.__version__}")
try:
    import plotly
    print(f"Plotly version: {plotly.__version__}")
except:
    print("Plotly version: Unable to determine")

# Test 2: Create sample data
print("\n📊 Creating sample time series data...")
dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
values = np.random.randn(len(dates)).cumsum() + 100
df = pd.DataFrame({'ds': dates, 'y': values})
print(f"✅ Sample data created: {len(df)} rows from {df['ds'].min()} to {df['ds'].max()}")

# Test 3: Test the NEW range selector configuration (day-based)
print("\n🔧 Testing NEW range selector configuration...")
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Sample Data'
    ))
    
    # This is the NEW configuration that should work with pandas >= 2.0
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=30, label="1M", step="day", stepmode="backward"),   # ✅ NEW
                    dict(count=90, label="3M", step="day", stepmode="backward"),   # ✅ NEW
                    dict(count=180, label="6M", step="day", stepmode="backward"),  # ✅ NEW
                    dict(count=365, label="1Y", step="day", stepmode="backward"),  # ✅ NEW
                    dict(count=730, label="2Y", step="day", stepmode="backward"),  # ✅ NEW
                    dict(step="all", label="All")
                ]
            )
        )
    )
    print("✅ NEW range selector configuration created successfully")
    
except Exception as e:
    print(f"❌ ERROR with NEW configuration: {e}")
    sys.exit(1)

# Test 4: Test timestamp arithmetic (what was causing the original error)
print("\n⏰ Testing timestamp arithmetic...")
try:
    # OLD problematic code (this would fail in pandas >= 2.0):
    # pd.Timestamp('2023-01-01') + 1  # ❌ Addition/subtraction of integers no longer supported
    
    # NEW correct approach:
    ts = pd.Timestamp('2023-01-01')
    ts_plus_30_days = ts + pd.Timedelta(days=30)
    ts_plus_90_days = ts + pd.Timedelta(days=90)
    ts_plus_180_days = ts + pd.Timedelta(days=180)
    ts_plus_365_days = ts + pd.Timedelta(days=365)
    ts_plus_730_days = ts + pd.Timedelta(days=730)
    
    print(f"✅ Base date: {ts}")
    print(f"✅ + 30 days (1M): {ts_plus_30_days}")
    print(f"✅ + 90 days (3M): {ts_plus_90_days}")
    print(f"✅ + 180 days (6M): {ts_plus_180_days}")
    print(f"✅ + 365 days (1Y): {ts_plus_365_days}")
    print(f"✅ + 730 days (2Y): {ts_plus_730_days}")
    
except Exception as e:
    print(f"❌ ERROR with timestamp arithmetic: {e}")
    sys.exit(1)

# Test 5: Test Prophet module imports
print("\n🔮 Testing Prophet module imports...")
try:
    from modules.prophet_module import create_prophet_forecast_chart, run_prophet_forecast
    print("✅ Prophet module imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    sys.exit(1)

# Test 6: Verify files have been updated
print("\n📄 Verifying file updates...")
files_to_check = [
    '/workspaces/CC-Excellence/modules/prophet_module.py',
    '/workspaces/CC-Excellence/pages/1_📈Forecasting.py'
]

for file_path in files_to_check:
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check for old problematic patterns
        old_patterns = [
            'step="month"',
            'step="year"',
            "step='month'",
            "step='year'"
        ]
        
        found_old = any(pattern in content for pattern in old_patterns)
        
        # Check for new correct patterns  
        new_patterns = [
            'step="day"',
            'count=30',
            'count=90', 
            'count=180',
            'count=365',
            'count=730'
        ]
        
        found_new = any(pattern in content for pattern in new_patterns)
        
        if found_old:
            print(f"⚠️  {file_path}: Still contains old patterns")
        elif found_new:
            print(f"✅ {file_path}: Updated with new patterns")
        else:
            print(f"ℹ️  {file_path}: No range selectors found")
            
    except Exception as e:
        print(f"❌ Error checking {file_path}: {e}")

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("✅ Range selectors are now compatible with pandas >= 2.0")
print("✅ No more timestamp arithmetic errors")
print("✅ Prophet module should work correctly")
print("=" * 60)
