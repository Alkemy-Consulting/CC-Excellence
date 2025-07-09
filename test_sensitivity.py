#!/usr/bin/env python3
"""
Test script per verificare la funzionalità della tabella di sensitivity
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the modules from the pages directory
import importlib.util
import pandas as pd
import numpy as np

def test_sensitivity_function():
    """Test della funzione generate_erlang_sensitivity_table"""
    
    # Load the module dynamically 
    spec = importlib.util.spec_from_file_location("capacity_sizing", "pages/2_🧮Capacity Sizing.py")
    capacity_sizing = importlib.util.module_from_spec(spec)
    
    # Test parameters
    arrival_rate = 100  # 100 chiamate/ora
    aht = 300  # 300 secondi AHT
    service_level_target = 0.80  # 80% SL target
    answer_time_target = 20  # 20 secondi
    patience = 90  # 90 secondi pazienza
    model_type = "Erlang C"
    max_occupancy = 0.85  # 85% max occupancy
    
    print("🧪 Test della funzione generate_erlang_sensitivity_table")
    print(f"📊 Parametri: {arrival_rate} chiamate/ora, AHT {aht}s, SL {service_level_target:.0%}, Answer time {answer_time_target}s")
    print("-" * 80)
    
    try:
        # Execute the module to load all functions
        spec.loader.exec_module(capacity_sizing)
        
        # Test the sensitivity function
        sensitivity_df = capacity_sizing.generate_erlang_sensitivity_table(
            arrival_rate=arrival_rate,
            aht=aht,
            service_level_target=service_level_target,
            answer_time_target=answer_time_target,
            patience=patience,
            model_type=model_type,
            max_occupancy=max_occupancy
        )
        
        print("✅ Funzione eseguita con successo!")
        print(f"📋 Tabella generata con {len(sensitivity_df)} righe")
        print("\n📊 Prime 5 righe della tabella:")
        print(sensitivity_df.head().to_string(index=False))
        
        # Check for optimal configuration
        optimal_rows = sensitivity_df[sensitivity_df['Target Met'] == True]
        if not optimal_rows.empty:
            optimal_row = optimal_rows.iloc[0]
            print(f"\n✅ Configurazione ottimale trovata:")
            print(f"   👥 Agenti: {optimal_row['Number of Agents']}")
            print(f"   📈 Service Level: {optimal_row['Service Level %']:.1f}%")
            print(f"   📊 Occupancy: {optimal_row['Occupancy %']:.1f}%")
            print(f"   ⏱️ ASA: {optimal_row['ASA (seconds)']:.1f}s")
        else:
            print("\n⚠️ Nessuna configurazione ottimale nel range testato")
        
        print("\n🎯 Test completato con successo!")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sensitivity_function()
    exit(0 if success else 1)
