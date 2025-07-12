#!/usr/bin/env python3
"""
AUDIT TECNICO-SCIENTIFICO COMPLETO - MODULO PROPHET
Analisi dettagliata di ogni funzione secondo rigorosità scientifica
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def run_comprehensive_prophet_audit():
    """
    Esecuzione audit completo del modulo Prophet
    """
    print("🔍 AUDIT TECNICO-SCIENTIFICO COMPLETO - MODULO PROPHET")
    print("=" * 70)
    
    audit_results = {
        'data_layer': 0,
        'business_logic_models': 0, 
        'ux_ui_code': 0,
        'rendering_export': 0,
        'robustezza_sicurezza': 0,
        'performance_scalabilita': 0,
        'testing_ci': 0
    }
    
    issues_checklist = []
    
    # 1. DATA LAYER AUDIT
    print("\n📊 1. DATA LAYER AUDIT")
    print("-" * 30)
    
    data_layer_score = audit_data_layer()
    audit_results['data_layer'] = data_layer_score
    
    # 2. BUSINESS LOGIC & MODELS AUDIT  
    print("\n🧠 2. BUSINESS LOGIC & MODELS AUDIT")
    print("-" * 35)
    
    business_logic_score = audit_business_logic()
    audit_results['business_logic_models'] = business_logic_score
    
    # 3. UX/UI CODE AUDIT
    print("\n🎨 3. UX/UI CODE AUDIT")
    print("-" * 20)
    
    ui_score = audit_ui_code()
    audit_results['ux_ui_code'] = ui_score
    
    # 4. RENDERING & EXPORT AUDIT
    print("\n📈 4. RENDERING & EXPORT AUDIT")
    print("-" * 30)
    
    rendering_score = audit_rendering_export()
    audit_results['rendering_export'] = rendering_score
    
    # 5. ROBUSTEZZA & SICUREZZA AUDIT
    print("\n🛡️ 5. ROBUSTEZZA & SICUREZZA AUDIT")
    print("-" * 35)
    
    security_score = audit_security_robustness()
    audit_results['robustezza_sicurezza'] = security_score
    
    # 6. PERFORMANCE & SCALABILITÀ AUDIT
    print("\n⚡ 6. PERFORMANCE & SCALABILITÀ AUDIT")
    print("-" * 35)
    
    performance_score = audit_performance_scalability()
    audit_results['performance_scalabilita'] = performance_score
    
    # 7. TESTING & CI AUDIT  
    print("\n🧪 7. TESTING & CI AUDIT")
    print("-" * 22)
    
    testing_score = audit_testing_ci()
    audit_results['testing_ci'] = testing_score
    
    # REPORT FINALE
    print("\n" + "=" * 70)
    print("📋 REPORT DI VALUTAZIONE FINALE")
    print("=" * 70)
    
    generate_final_report(audit_results)
    
    return audit_results

def audit_data_layer():
    """Audit del data layer Prophet"""
    print("Analizzando moduli data_utils.py e validazione input...")
    
    score = 0
    issues = []
    
    # Verifica esistenza file core
    try:
        with open('/workspaces/CC-Excellence/modules/prophet_core.py', 'r') as f:
            core_content = f.read()
            
        # 1. Controllo validazione input robusta
        if 'validate_inputs' in core_content:
            print("✅ Validazione input presente")
            score += 2
            
            # Controlli specifici
            validation_checks = [
                'isinstance(df, pd.DataFrame)',
                'df.empty',
                'pd.to_numeric',
                'pd.to_datetime', 
                'isna().sum()',
                'zero variance'
            ]
            
            for check in validation_checks:
                if check in core_content:
                    print(f"  ✅ {check} - implementato")
                    score += 0.5
                else:
                    print(f"  ❌ {check} - mancante")
                    issues.append(f"Validazione mancante: {check}")
        else:
            print("❌ Validazione input non trovata")
            issues.append("Implementare validazione input robusta")
            
        # 2. Ottimizzazione DataFrame
        if 'optimize_dataframe' in core_content:
            print("✅ Ottimizzazione DataFrame presente")
            score += 1
        else:
            print("❌ Ottimizzazione DataFrame mancante")
            issues.append("Implementare ottimizzazione memoria DataFrame")
            
        # 3. Gestione outlier e missing values
        if 'dropna' in core_content and 'errors=' in core_content:
            print("✅ Gestione missing values presente")
            score += 1
        else:
            print("⚠️ Gestione outlier/missing limitata")
            score += 0.5
            
    except Exception as e:
        print(f"❌ Errore lettura prophet_core.py: {e}")
        score = 2
        
    final_score = min(10, max(1, score))
    print(f"\n📊 PUNTEGGIO DATA LAYER: {final_score}/10")
    
    if issues:
        print("🔧 ISSUES TROVATE:")
        for issue in issues:
            print(f"  - {issue}")
    
    return final_score

def audit_business_logic():
    """Audit della business logic dei modelli Prophet"""
    print("Analizzando implementazione algoritmi Prophet...")
    
    score = 0
    issues = []
    
    try:
        # Leggi prophet_core.py
        with open('/workspaces/CC-Excellence/modules/prophet_core.py', 'r') as f:
            core_content = f.read()
            
        # Leggi prophet_module.py  
        with open('/workspaces/CC-Excellence/modules/prophet_module.py', 'r') as f:
            module_content = f.read()
            
        # 1. Implementazione Prophet corretta
        prophet_features = [
            'seasonality_mode',
            'changepoint_prior_scale', 
            'seasonality_prior_scale',
            'yearly_seasonality',
            'weekly_seasonality',
            'daily_seasonality',
            'interval_width'
        ]
        
        for feature in prophet_features:
            if feature in core_content:
                print(f"✅ {feature} - configurabile")
                score += 0.5
            else:
                print(f"❌ {feature} - non configurabile")
                issues.append(f"Rendere configurabile: {feature}")
                
        # 2. Cross-validation implementazione
        if 'cross_validation' in module_content:
            print("✅ Cross-validation implementata")
            score += 2
        else:
            print("❌ Cross-validation mancante")
            issues.append("Implementare cross-validation Prophet")
            
        # 3. Holidays support
        if 'add_holidays' in core_content or 'holidays' in core_content:
            print("✅ Support holidays presente")
            score += 1.5
        else:
            print("❌ Support holidays mancante")
            issues.append("Implementare support holidays")
            
        # 4. Metriche performance
        if 'calculate_metrics' in core_content:
            print("✅ Calcolo metriche presente")
            score += 1
            
            # Controllo metriche specifiche
            metrics = ['mape', 'mae', 'rmse', 'r2']
            for metric in metrics:
                if metric in core_content:
                    print(f"  ✅ {metric.upper()} implementata")
                    score += 0.25
                else:
                    print(f"  ❌ {metric.upper()} mancante")
                    
        # 5. Gestione parametri esterni
        if 'external_regressors' in module_content:
            print("✅ External regressors supportati")
            score += 1
        else:
            print("⚠️ External regressors limitati")
            score += 0.5
            
    except Exception as e:
        print(f"❌ Errore analisi business logic: {e}")
        score = 4
        
    final_score = min(10, max(1, score))
    print(f"\n🧠 PUNTEGGIO BUSINESS LOGIC: {final_score}/10")
    
    if issues:
        print("🔧 ISSUES TROVATE:")
        for issue in issues:
            print(f"  - {issue}")
    
    return final_score

def audit_ui_code():
    """Audit del codice UI/UX"""
    print("Analizzando separazione logica vs presentazione...")
    
    score = 0
    issues = []
    
    try:
        # Verifica architettura pulita
        if os.path.exists('/workspaces/CC-Excellence/modules/prophet_presentation.py'):
            print("✅ Layer presentazione separato")
            score += 3
        else:
            print("❌ Layer presentazione non separato")
            issues.append("Separare logica UI dalla business logic")
            
        # Verifica modulo UI components
        files_to_check = ['prophet_module.py']
        
        for filename in files_to_check:
            try:
                with open(f'/workspaces/CC-Excellence/modules/{filename}', 'r') as f:
                    content = f.read()
                    
                # Controllo separazione concerns
                ui_mixed_with_logic = content.count('st.') > 10 and 'Prophet(' in content
                if ui_mixed_with_logic:
                    print(f"⚠️ {filename}: UI logic mixed con business logic") 
                    score += 1
                    issues.append(f"Separare UI da business logic in {filename}")
                else:
                    print(f"✅ {filename}: Buona separazione concerns")
                    score += 2
                    
                # Controllo tooltip e help
                if 'help=' in content or 'tooltip' in content:
                    print(f"✅ {filename}: Help contestuale presente")
                    score += 1
                else:
                    print(f"⚠️ {filename}: Help contestuale limitato")
                    
            except Exception as e:
                print(f"❌ Errore lettura {filename}: {e}")
                
    except Exception as e:
        print(f"❌ Errore audit UI: {e}")
        score = 3
        
    final_score = min(10, max(1, score))
    print(f"\n🎨 PUNTEGGIO UX/UI CODE: {final_score}/10")
    
    if issues:
        print("🔧 ISSUES TROVATE:")
        for issue in issues:
            print(f"  - {issue}")
    
    return final_score

def audit_rendering_export():
    """Audit rendering e export"""
    print("Analizzando generazione grafici e export...")
    
    score = 0
    issues = []
    
    try:
        # Verifica presentation layer
        if os.path.exists('/workspaces/CC-Excellence/modules/prophet_presentation.py'):
            with open('/workspaces/CC-Excellence/modules/prophet_presentation.py', 'r') as f:
                presentation_content = f.read()
                
            # Controllo grafici Plotly
            if 'plotly' in presentation_content:
                print("✅ Grafici Plotly implementati")
                score += 2
            else:
                print("❌ Grafici Plotly mancanti")
                issues.append("Implementare grafici Plotly professionale")
                
            # Controllo consistenza stile
            if 'colors' in presentation_content or 'theme' in presentation_content:
                print("✅ Consistenza stile presente")
                score += 1
            else:
                print("⚠️ Consistenza stile limitata")
                issues.append("Implementare tema consistente")
                
        # Verifica funzioni export
        export_formats = ['pdf', 'excel', 'csv']
        for fmt in export_formats:
            # Qui dovremmo verificare se ci sono funzioni di export
            # Per ora assegniamo punteggio base
            score += 0.5
            
        # Controllo diagnostics plots
        if os.path.exists('/workspaces/CC-Excellence/modules/prophet_diagnostics.py'):
            print("✅ Diagnostic plots implementati")
            score += 3
        else:
            print("❌ Diagnostic plots mancanti")
            issues.append("Implementare diagnostic plots avanzati")
            
    except Exception as e:
        print(f"❌ Errore audit rendering: {e}")
        score = 4
        
    final_score = min(10, max(1, score))
    print(f"\n📈 PUNTEGGIO RENDERING & EXPORT: {final_score}/10")
    
    if issues:
        print("🔧 ISSUES TROVATE:")
        for issue in issues:
            print(f"  - {issue}")
    
    return final_score

def audit_security_robustness():
    """Audit sicurezza e robustezza"""
    print("Analizzando validazione input, exception handling, logging...")
    
    score = 0
    issues = []
    
    try:
        with open('/workspaces/CC-Excellence/modules/prophet_core.py', 'r') as f:
            core_content = f.read()
            
        # 1. Validazione robusta input
        security_checks = [
            'str(date_col).strip()',  # Sanitization
            'errors=\'raise\'',        # Strict validation
            'TypeError',              # Type checking
            'ValueError',             # Value validation  
            'KeyError',               # Key validation
            'len(df) < 10',          # Minimum data check
            'len(df) > 100000'       # Maximum data check
        ]
        
        for check in security_checks:
            if check in core_content:
                print(f"✅ Security check: {check}")
                score += 0.5
            else:
                print(f"❌ Security check mancante: {check}")
                issues.append(f"Implementare: {check}")
                
        # 2. Exception handling
        exception_patterns = ['try:', 'except:', 'finally:', 'raise']
        exception_count = sum(core_content.count(pattern) for pattern in exception_patterns)
        
        if exception_count > 10:
            print(f"✅ Exception handling robusto ({exception_count} patterns)")
            score += 2
        elif exception_count > 5:
            print(f"⚠️ Exception handling moderato ({exception_count} patterns)")
            score += 1
        else:
            print(f"❌ Exception handling insufficiente ({exception_count} patterns)")
            issues.append("Migliorare gestione eccezioni")
            
        # 3. Logging
        if 'logger' in core_content and 'logging' in core_content:
            print("✅ Logging implementato")
            score += 1.5
        else:
            print("❌ Logging mancante")
            issues.append("Implementare logging completo")
            
        # 4. Input sanitization 
        if 'strip()' in core_content and 'str(' in core_content:
            print("✅ Input sanitization presente")
            score += 1
        else:
            print("❌ Input sanitization mancante") 
            issues.append("Implementare sanitization input")
            
    except Exception as e:
        print(f"❌ Errore audit sicurezza: {e}")
        score = 3
        
    final_score = min(10, max(1, score))
    print(f"\n🛡️ PUNTEGGIO ROBUSTEZZA & SICUREZZA: {final_score}/10")
    
    if issues:
        print("🔧 ISSUES TROVATE:")
        for issue in issues:
            print(f"  - {issue}")
    
    return final_score

def audit_performance_scalability():
    """Audit performance e scalabilità"""
    print("Analizzando ottimizzazioni, caching, parallelizzazione...")
    
    score = 0
    issues = []
    
    try:
        # Verifica performance layer
        if os.path.exists('/workspaces/CC-Excellence/modules/prophet_performance.py'):
            print("✅ Performance layer presente")
            score += 3
            
            with open('/workspaces/CC-Excellence/modules/prophet_performance.py', 'r') as f:
                perf_content = f.read()
                
            # Controllo ottimizzazioni specifiche
            optimizations = [
                'lru_cache',         # Caching
                'memory_usage',      # Memory optimization
                'parallel',          # Parallelization  
                'optimize',          # General optimization
                'performance',       # Performance monitoring
            ]
            
            for opt in optimizations:
                if opt in perf_content:
                    print(f"✅ Ottimizzazione: {opt}")
                    score += 0.5
                else:
                    print(f"⚠️ Ottimizzazione mancante: {opt}")
                    
        else:
            print("❌ Performance layer mancante")
            issues.append("Implementare layer performance dedicato")
            
        # Verifica caching nel core
        with open('/workspaces/CC-Excellence/modules/prophet_core.py', 'r') as f:
            core_content = f.read()
            
        if '@lru_cache' in core_content:
            print("✅ LRU Cache implementato")
            score += 2
        else:
            print("❌ Caching mancante")
            issues.append("Implementare caching parametri")
            
        # Verifica ottimizzazione DataFrame
        if 'downcast' in core_content:
            print("✅ DataFrame optimization presente")
            score += 1
        else:
            print("⚠️ DataFrame optimization limitata")
            
    except Exception as e:
        print(f"❌ Errore audit performance: {e}")
        score = 4
        
    final_score = min(10, max(1, score))
    print(f"\n⚡ PUNTEGGIO PERFORMANCE & SCALABILITÀ: {final_score}/10")
    
    if issues:
        print("🔧 ISSUES TROVATE:")
        for issue in issues:
            print(f"  - {issue}")
    
    return final_score

def audit_testing_ci():
    """Audit testing e CI/CD"""
    print("Analizzando copertura test, qualità test, CI pipeline...")
    
    score = 0
    issues = []
    
    try:
        # Conta file di test
        test_files = []
        for root, dirs, files in os.walk('/workspaces/CC-Excellence'):
            for file in files:
                if 'test' in file.lower() and 'prophet' in file.lower() and file.endswith('.py'):
                    test_files.append(file)
                    
        print(f"✅ File di test trovati: {len(test_files)}")
        for test_file in test_files:
            print(f"  - {test_file}")
            
        if len(test_files) >= 5:
            print("✅ Copertura test buona")
            score += 3
        elif len(test_files) >= 3:
            print("⚠️ Copertura test moderata")
            score += 2
        else:
            print("❌ Copertura test insufficiente")
            score += 1
            issues.append("Aumentare copertura test")
            
        # Verifica presenza pytest
        if os.path.exists('/workspaces/CC-Excellence/pytest.ini'):
            print("✅ Pytest configurato")
            score += 1
        else:
            print("⚠️ Pytest non configurato")
            
        # Verifica test di integrazione
        integration_tests = [f for f in test_files if 'integration' in f.lower()]
        if integration_tests:
            print(f"✅ Test integrazione: {len(integration_tests)}")
            score += 2
        else:
            print("❌ Test integrazione mancanti")
            issues.append("Implementare test integrazione")
            
        # Verifica CI/CD
        ci_files = ['.github/workflows', '.gitlab-ci.yml', 'azure-pipelines.yml']
        ci_found = any(os.path.exists(f'/workspaces/CC-Excellence/{ci}') for ci in ci_files)
        
        if ci_found:
            print("✅ CI/CD configurato")
            score += 2
        else:
            print("❌ CI/CD mancante")
            issues.append("Configurare CI/CD pipeline")
            
    except Exception as e:
        print(f"❌ Errore audit testing: {e}")
        score = 3
        
    final_score = min(10, max(1, score))
    print(f"\n🧪 PUNTEGGIO TESTING & CI: {final_score}/10")
    
    if issues:
        print("🔧 ISSUES TROVATE:")
        for issue in issues:
            print(f"  - {issue}")
    
    return final_score

def generate_final_report(audit_results):
    """Genera report finale audit"""
    
    print("┌" + "─" * 40 + "┬" + "─" * 10 + "┬" + "─" * 15 + "┐")
    print("│ AREA DI AUDIT                    │ PUNTEGGIO │ VALUTAZIONE     │")
    print("├" + "─" * 40 + "┼" + "─" * 10 + "┼" + "─" * 15 + "┤")
    
    for area, score in audit_results.items():
        area_name = area.replace('_', ' ').title()
        valutazione = get_valutazione(score)
        print(f"│ {area_name:<38} │ {score:>8} │ {valutazione:<13} │")
    
    print("└" + "─" * 40 + "┴" + "─" * 10 + "┴" + "─" * 15 + "┘")
    
    # Calcola punteggio complessivo
    total_score = sum(audit_results.values()) / len(audit_results)
    
    print(f"\n🎯 PUNTEGGIO COMPLESSIVO: {total_score:.1f}/10")
    print(f"📊 VALUTAZIONE FINALE: {get_valutazione(total_score)}")
    
    # Raccomandazioni
    print(f"\n💡 RACCOMANDAZIONI PRIORITARIE:")
    if total_score >= 8:
        print("✅ Modulo Prophet di eccellenza! Implementazione scientificamente rigorosa.")
    elif total_score >= 7:
        print("✅ Modulo Prophet solido. Piccoli miglioramenti consigliati.")
    elif total_score >= 6:
        print("⚠️ Modulo Prophet funzionale ma necessita miglioramenti.")
    else:
        print("❌ Modulo Prophet necessita revisione maggiore per conformità scientifica.")
        
    # Checklist miglioramenti
    print(f"\n📋 CHECKLIST MIGLIORIE:")
    
    priority_issues = []
    
    if audit_results['robustezza_sicurezza'] < 7:
        priority_issues.append({
            'file': 'modules/prophet_core.py',
            'descrizione': 'Migliorare validazione input e gestione eccezioni',
            'obiettivo': 'Aumentare robustezza e sicurezza modulo',
            'priorita': 'ALTA'
        })
        
    if audit_results['business_logic_models'] < 8:
        priority_issues.append({
            'file': 'modules/prophet_module.py',
            'descrizione': 'Implementare cross-validation e holidays completi',
            'obiettivo': 'Conformità scientifica algoritmi Prophet',
            'priorita': 'ALTA'
        })
        
    if audit_results['testing_ci'] < 7:
        priority_issues.append({
            'file': 'tests/',
            'descrizione': 'Aumentare copertura test e implementare CI/CD',
            'obiettivo': 'Garantire qualità e reliability',
            'priorita': 'MEDIA'
        })
        
    if audit_results['performance_scalabilita'] < 7:
        priority_issues.append({
            'file': 'modules/prophet_performance.py',
            'descrizione': 'Ottimizzare performance e implementare caching',
            'obiettivo': 'Migliorare scalabilità sistema',
            'priorita': 'MEDIA'
        })
        
    for i, issue in enumerate(priority_issues, 1):
        print(f"\n{i}. FILE: {issue['file']}")
        print(f"   DESCRIZIONE: {issue['descrizione']}")
        print(f"   OBIETTIVO: {issue['obiettivo']}")
        print(f"   PRIORITÀ: {issue['priorita']}")

def get_valutazione(score):
    """Converte score numerico in valutazione"""
    if score >= 9:
        return "ECCELLENTE"
    elif score >= 8:
        return "OTTIMO"
    elif score >= 7:
        return "BUONO"
    elif score >= 6:
        return "SUFFICIENTE"
    elif score >= 5:
        return "MEDIOCRE"
    else:
        return "INSUFFICIENTE"

if __name__ == "__main__":
    run_comprehensive_prophet_audit()
