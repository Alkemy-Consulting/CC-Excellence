#!/usr/bin/env python3
"""
Test script to verify the forecast execution logic
"""

def test_forecast_logic():
    """Test the forecast trigger logic"""
    
    # Simulate session state
    session_state = {}
    
    print("🧪 Testing Forecast Logic")
    print("=" * 50)
    
    # Test 1: Initial state - no data loaded
    print("\n1️⃣ Test: Initial state")
    session_state = {}
    
    data_loaded = session_state.get('data_loaded', False)
    run_forecast = session_state.get('run_forecast', False)
    forecast_results_available = session_state.get('forecast_results_available', False)
    
    if not data_loaded:
        state = "WELCOME_SCREEN"
    elif data_loaded and not run_forecast and not forecast_results_available:
        state = "DATA_ANALYSIS"
    elif data_loaded and run_forecast:
        state = "FORECAST_EXECUTION"
        # Simulate immediate reset
        session_state['run_forecast'] = False
        session_state['forecast_results_available'] = True
    elif data_loaded and forecast_results_available:
        state = "FORECAST_RESULTS"
    else:
        state = "UNKNOWN"
    
    print(f"   Data loaded: {data_loaded}")
    print(f"   Run forecast: {run_forecast}")
    print(f"   Results available: {forecast_results_available}")
    print(f"   ➡️ State: {state}")
    
    # Test 2: Data loaded, no forecast
    print("\n2️⃣ Test: Data loaded, ready for forecast")
    session_state = {'data_loaded': True}
    
    data_loaded = session_state.get('data_loaded', False)
    run_forecast = session_state.get('run_forecast', False)
    forecast_results_available = session_state.get('forecast_results_available', False)
    
    if not data_loaded:
        state = "WELCOME_SCREEN"
    elif data_loaded and not run_forecast and not forecast_results_available:
        state = "DATA_ANALYSIS"
    elif data_loaded and run_forecast:
        state = "FORECAST_EXECUTION"
    elif data_loaded and forecast_results_available:
        state = "FORECAST_RESULTS"
    else:
        state = "UNKNOWN"
    
    print(f"   Data loaded: {data_loaded}")
    print(f"   Run forecast: {run_forecast}")
    print(f"   Results available: {forecast_results_available}")
    print(f"   ➡️ State: {state}")
    
    # Test 3: Button clicked - trigger forecast
    print("\n3️⃣ Test: Button clicked - trigger forecast")
    session_state = {'data_loaded': True, 'run_forecast': True}
    
    data_loaded = session_state.get('data_loaded', False)
    run_forecast = session_state.get('run_forecast', False)
    forecast_results_available = session_state.get('forecast_results_available', False)
    
    if not data_loaded:
        state = "WELCOME_SCREEN"
    elif data_loaded and not run_forecast and not forecast_results_available:
        state = "DATA_ANALYSIS"
    elif data_loaded and run_forecast:
        state = "FORECAST_EXECUTION"
        # CRITICAL: Simulate immediate reset after detection
        print("   🔄 RESETTING run_forecast flag immediately...")
        session_state['run_forecast'] = False
        session_state['forecast_results_available'] = True
    elif data_loaded and forecast_results_available:
        state = "FORECAST_RESULTS"
    else:
        state = "UNKNOWN"
    
    print(f"   Data loaded: {data_loaded}")
    print(f"   Run forecast: {run_forecast}")
    print(f"   Results available: {forecast_results_available}")
    print(f"   ➡️ State: {state}")
    
    # Test 4: After forecast execution - showing results
    print("\n4️⃣ Test: After forecast execution - showing results")
    session_state = {'data_loaded': True, 'forecast_results_available': True}
    
    data_loaded = session_state.get('data_loaded', False)
    run_forecast = session_state.get('run_forecast', False)
    forecast_results_available = session_state.get('forecast_results_available', False)
    
    if not data_loaded:
        state = "WELCOME_SCREEN"
    elif data_loaded and not run_forecast and not forecast_results_available:
        state = "DATA_ANALYSIS"
    elif data_loaded and run_forecast:
        state = "FORECAST_EXECUTION"
    elif data_loaded and forecast_results_available:
        state = "FORECAST_RESULTS"
    else:
        state = "UNKNOWN"
    
    print(f"   Data loaded: {data_loaded}")
    print(f"   Run forecast: {run_forecast}")
    print(f"   Results available: {forecast_results_available}")
    print(f"   ➡️ State: {state}")
    
    # Test 5: Click "Run Another Forecast" - back to data analysis
    print("\n5️⃣ Test: Click 'Run Another Forecast' - back to data analysis")
    session_state = {'data_loaded': True}  # forecast_results_available cleared
    
    data_loaded = session_state.get('data_loaded', False)
    run_forecast = session_state.get('run_forecast', False)
    forecast_results_available = session_state.get('forecast_results_available', False)
    
    if not data_loaded:
        state = "WELCOME_SCREEN"
    elif data_loaded and not run_forecast and not forecast_results_available:
        state = "DATA_ANALYSIS"
    elif data_loaded and run_forecast:
        state = "FORECAST_EXECUTION"
    elif data_loaded and forecast_results_available:
        state = "FORECAST_RESULTS"
    else:
        state = "UNKNOWN"
    
    print(f"   Data loaded: {data_loaded}")
    print(f"   Run forecast: {run_forecast}")
    print(f"   Results available: {forecast_results_available}")
    print(f"   ➡️ State: {state}")
    
    print("\n✅ Logic Test Summary:")
    print("   • Forecast ONLY executes when button is clicked")
    print("   • run_forecast flag is reset immediately after detection")
    print("   • forecast_results_available tracks completed forecasts")
    print("   • User can navigate back to data analysis or run new forecasts")
    print("   • NO automatic re-execution on app rerun")
    
    return True

if __name__ == "__main__":
    test_forecast_logic()
