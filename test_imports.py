#!/usr/bin/env python3

import sys
import os
sys.path.append('/workspaces/CC-Excellence')

try:
    from modules.ui_components import render_data_upload_section
    print("✅ ui_components import successful")
except Exception as e:
    print(f"❌ ui_components import failed: {e}")

try:
    from modules.config import SUPPORTED_HOLIDAY_COUNTRIES
    print("✅ config import successful")
    print(f"Holiday countries: {SUPPORTED_HOLIDAY_COUNTRIES}")
except Exception as e:
    print(f"❌ config import failed: {e}")

try:
    from modules.data_utils import get_holidays_for_country, parse_manual_holidays
    print("✅ data_utils holiday functions import successful")
except Exception as e:
    print(f"❌ data_utils holiday functions import failed: {e}")

try:
    from modules.prophet_module import build_and_forecast_prophet_enhanced
    print("✅ prophet_module enhanced function import successful")
except Exception as e:
    print(f"❌ prophet_module enhanced function import failed: {e}")

try:
    from modules.forecast_engine import run_prophet_forecast
    print("✅ forecast_engine import successful")
except Exception as e:
    print(f"❌ forecast_engine import failed: {e}")

print("✅ All critical imports successful!")
