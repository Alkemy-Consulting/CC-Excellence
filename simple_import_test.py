#!/usr/bin/env python3
"""Simple import test for debugging"""

import sys
sys.path.append('/workspaces/CC-Excellence')

# Test basic imports first
try:
    import pandas as pd
    import numpy as np
    print("Basic imports OK")
except Exception as e:
    print(f"Basic import error: {e}")

# Test forecast engine
try:
    from modules.forecast_engine import run_enhanced_forecast
    print("Forecast engine import OK")
except Exception as e:
    print(f"Forecast engine error: {e}")

print("Test complete")
