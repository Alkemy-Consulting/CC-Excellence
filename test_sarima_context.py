#!/usr/bin/env python3
"""
Test SARIMA module import in Streamlit context.
"""

def test_sarima_in_streamlit_context():
    """Test SARIMA import as it would happen in Streamlit app."""
    print("Testing SARIMA module import in Streamlit context...")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        # Import streamlit first to simulate app context
        import streamlit as st
        print("✅ Streamlit imported")
        
        # Now try to import the SARIMA module
        from modules.sarima_enhanced import run_sarima_forecast, SARIMA_AVAILABLE
        print(f"✅ SARIMA module imported successfully")
        print(f"SARIMA_AVAILABLE = {SARIMA_AVAILABLE}")
        
        if SARIMA_AVAILABLE:
            print("🎉 SARIMA is available and ready to use!")
            return True
        else:
            print("❌ SARIMA is marked as unavailable")
            return False
            
    except Exception as e:
        print(f"❌ Error importing SARIMA module: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_forecast_engine_sarima():
    """Test SARIMA through forecast engine."""
    print("\nTesting SARIMA through forecast engine...")
    
    try:
        import sys
        import os
        sys.path.insert(0, os.getcwd())
        
        # Import forecast engine
        from modules.forecast_engine import ENHANCED_MODELS_AVAILABLE
        print(f"Enhanced models available: {ENHANCED_MODELS_AVAILABLE}")
        
        # Try to check SARIMA specifically
        try:
            from modules.sarima_enhanced import SARIMA_AVAILABLE
            print(f"SARIMA available: {SARIMA_AVAILABLE}")
        except Exception as e:
            print(f"Error checking SARIMA: {e}")
            
        return ENHANCED_MODELS_AVAILABLE
        
    except Exception as e:
        print(f"❌ Error testing forecast engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing SARIMA in app context...\n")
    
    direct_test = test_sarima_in_streamlit_context()
    engine_test = test_forecast_engine_sarima()
    
    print(f"\n📊 Results:")
    print(f"   Direct SARIMA import: {'✅ PASS' if direct_test else '❌ FAIL'}")
    print(f"   Forecast engine test: {'✅ PASS' if engine_test else '❌ FAIL'}")
    
    if direct_test and engine_test:
        print("\n🎉 SARIMA works correctly in app context!")
    else:
        print("\n❌ SARIMA still has issues in app context.")

if __name__ == "__main__":
    main()
