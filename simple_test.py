#!/usr/bin/env python3
"""
Simple test to validate basic app components
"""

import sys
import os
sys.path.append('/workspaces/CC-Excellence')

def test_step_by_step():
    print("🔍 Testing CC-Excellence App Step by Step\n")
    
    # Step 1: Test Python environment
    print("1. Testing Python environment...")
    try:
        import pandas as pd
        import numpy as np
        print(f"   ✓ pandas {pd.__version__}")
        print(f"   ✓ numpy {np.__version__}")
    except Exception as e:
        print(f"   ❌ Error importing basic libraries: {e}")
        return
    
    # Step 2: Test config module
    print("\n2. Testing config module...")
    try:
        from modules.config import SUPPORTED_FILE_FORMATS, DEFAULT_FORECAST_HORIZONS
        print(f"   ✓ Config imported")
        print(f"   ✓ File formats: {len(SUPPORTED_FILE_FORMATS)}")
        print(f"   ✓ Forecast horizons: {DEFAULT_FORECAST_HORIZONS}")
    except Exception as e:
        print(f"   ❌ Config module error: {e}")
        return
    
    # Step 3: Test data utilities
    print("\n3. Testing data utilities...")
    try:
        from modules.data_utils import generate_sample_data, detect_file_format
        print("   ✓ Data utils imported")
        
        # Generate sample data
        df = generate_sample_data()
        print(f"   ✓ Sample data: {df.shape}")
        print(f"   ✓ Columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"   ❌ Data utils error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Test Streamlit components
    print("\n4. Testing Streamlit and UI components...")
    try:
        import streamlit as st
        print(f"   ✓ Streamlit {st.__version__}")
        
        # Test if we can import UI components (might fail without Streamlit context)
        from modules import ui_components
        print("   ✓ UI components module imported")
        
    except Exception as e:
        print(f"   ❌ Streamlit/UI error: {e}")
    
    # Step 5: Test forecast engine
    print("\n5. Testing forecast engine...")
    try:
        from modules.forecast_engine import run_enhanced_forecast
        print("   ✓ Forecast engine imported")
    except Exception as e:
        print(f"   ❌ Forecast engine error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Basic component test completed!")

if __name__ == "__main__":
    test_step_by_step()
