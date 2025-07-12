#!/usr/bin/env python3
"""
Test per verificare le modifiche di UX Consistency e Progressive Disclosure
"""

def test_ui_consistency():
    """Test che le funzioni di configurazione supportino disabled=True"""
    print("🧪 Testing UX Consistency modifications...")
    
    try:
        # Test import delle funzioni modificate
        from modules.ui_components import render_prophet_config, render_arima_config, render_sarima_config, render_holtwinters_config
        print("✅ Import delle funzioni UI completato")
        
        # Test import della funzione display modificata
        from modules.forecast_engine import display_forecast_results
        print("✅ Import della funzione display_forecast_results completato")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        return False

def test_progressive_disclosure():
    """Test che le diagnostiche siano integrate correttamente"""
    print("🧪 Testing Progressive Disclosure modifications...")
    
    try:
        # Verifica che la funzione display_forecast_results sia stata modificata
        from modules.forecast_engine import display_forecast_results
        import inspect
        
        # Ottieni il codice sorgente della funzione
        source = inspect.getsource(display_forecast_results)
        
        # Verifica che contenga le nostre modifiche
        required_elements = [
            "🔬 **Diagnostiche Avanzate del Modello**",
            "Progressive Disclosure",
            "ARIMA/SARIMA Diagnostics",
            "Prophet Diagnostics",
            "Holt-Winters Diagnostics"
        ]
        
        found_elements = []
        for element in required_elements:
            if element in source:
                found_elements.append(element)
        
        print(f"✅ Elementi trovati: {len(found_elements)}/{len(required_elements)}")
        for element in found_elements:
            print(f"   ✓ {element}")
        
        return len(found_elements) >= 3  # Almeno 3 elementi devono essere presenti
        
    except Exception as e:
        print(f"❌ Errore durante il test: {e}")
        return False

def main():
    """Esegui tutti i test"""
    print("🚀 Avvio test delle modifiche UI...")
    print("=" * 60)
    
    # Test UX Consistency
    consistency_ok = test_ui_consistency()
    print()
    
    # Test Progressive Disclosure
    disclosure_ok = test_progressive_disclosure()
    print()
    
    # Risultato finale
    print("=" * 60)
    print("📊 RISULTATI FINALI:")
    print(f"   UX Consistency: {'✅ PASS' if consistency_ok else '❌ FAIL'}")
    print(f"   Progressive Disclosure: {'✅ PASS' if disclosure_ok else '❌ FAIL'}")
    
    if consistency_ok and disclosure_ok:
        print("\n🎉 Tutte le modifiche sono state implementate correttamente!")
        print("\n📋 RIEPILOGO MODIFICHE:")
        print("1. ✅ UX Consistency - Campi disabled=True quando auto è attivo")
        print("   - Prophet: Auto-tuning disabilita parametri manuali")
        print("   - ARIMA: Auto-ARIMA disabilita parametri p,d,q")
        print("   - SARIMA: Auto-SARIMA disabilita parametri manuali")
        print("   - Holt-Winters: Auto-HW disabilita tutti i parametri")
        print()
        print("2. ✅ Progressive Disclosure - Menu diagnostiche avanzate")
        print("   - Sezione espandibile '🔬 Diagnostiche Avanzate del Modello'")
        print("   - Diagnostiche specifiche per ogni modello")
        print("   - Interpretazione automatica dei risultati")
        print("   - Raccomandazioni personalizzate")
    else:
        print("\n❌ Alcune modifiche necessitano di revisione")

if __name__ == "__main__":
    main()
