#!/usr/bin/env python3
"""
Test script per verificare la funzionalità del parametro max_occupancy
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the modules
import importlib.util
import pandas as pd
import numpy as np

def test_max_occupancy_parameter():
    """Test del parametro max_occupancy nelle funzioni di calcolo"""
    
    # Load the module dynamically 
    spec = importlib.util.spec_from_file_location("capacity_sizing", "pages/2_🧮Capacity Sizing.py")
    capacity_sizing = importlib.util.module_from_spec(spec)
    
    print("🧪 Test del parametro Max Occupancy")
    print("-" * 60)
    
    try:
        # Execute the module to load all functions
        spec.loader.exec_module(capacity_sizing)
        
        # Test parameters
        arrival_rate = 200  # 200 chiamate/ora (alto volume)
        aht = 300  # 300 secondi AHT
        service_level_target = 0.80  # 80% SL target
        answer_time_target = 20  # 20 secondi
        
        print(f"📊 Parametri test: {arrival_rate} chiamate/ora, AHT {aht}s, SL {service_level_target:.0%}")
        
        # Test con diversi valori di max_occupancy
        max_occupancy_values = [0.70, 0.80, 0.85, 0.90, 0.95]
        
        print("\n🎯 Test Erlang C con diversi Max Occupancy:")
        print(f"{'Max Occ':<8} {'Agenti':<8} {'SL Reale':<10} {'Occ Reale':<10} {'Status':<15}")
        print("-" * 55)
        
        for max_occ in max_occupancy_values:
            agents, sl, occ = capacity_sizing.erlang_c(
                arrival_rate, aht, service_level_target, answer_time_target, max_occ
            )
            
            # Verifica se i vincoli sono rispettati
            sl_ok = sl >= service_level_target
            occ_ok = occ <= max_occ
            status = "✅ OK" if sl_ok and occ_ok else "⚠️ Vincoli" if not sl_ok or not occ_ok else "❌ Fail"
            
            print(f"{max_occ:<8.0%} {agents:<8.0f} {sl:<10.1%} {occ:<10.1%} {status:<15}")
        
        print("\n🎯 Test Erlang A con Max Occupancy 85%:")
        patience = 90
        agents_a, sl_a, occ_a = capacity_sizing.erlang_a(
            arrival_rate, aht, patience, service_level_target, answer_time_target, 0.85
        )
        print(f"   Agenti: {agents_a:.0f}, SL: {sl_a:.1%}, Occupancy: {occ_a:.1%}")
        
        print("\n🎯 Test Simulazione con Max Occupancy 85%:")
        agents_sim, sl_sim, occ_sim = capacity_sizing.simulation_model(
            arrival_rate, aht, service_level_target, 100, 0.85  # 100 simulazioni per velocità
        )
        print(f"   Agenti: {agents_sim:.0f}, SL: {sl_sim:.1%}, Occupancy: {occ_sim:.1%}")
        
        # Test della tabella sensitivity
        print("\n🔬 Test Tabella Sensitivity:")
        sensitivity_df = capacity_sizing.generate_erlang_sensitivity_table(
            arrival_rate=100,  # Volume più gestibile per la tabella
            aht=aht,
            service_level_target=service_level_target,
            answer_time_target=answer_time_target,
            patience=None,
            model_type="Erlang C",
            max_occupancy=0.85
        )
        
        optimal_rows = sensitivity_df[sensitivity_df['Target Met'] == True]
        print(f"   Configurazioni ottimali trovate: {len(optimal_rows)}")
        
        if not optimal_rows.empty:
            best = optimal_rows.iloc[0]
            print(f"   Migliore: {best['Number of Agents']} agenti, SL {best['Service Level %']:.1f}%, Occ {best['Occupancy %']:.1f}%")
        
        print("\n✅ Test completato con successo!")
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_max_occupancy_parameter()
    exit(0 if success else 1)
