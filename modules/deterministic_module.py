"""
Modulo per calcoli deterministici di workforce planning
Implementa modelli matematici rigorosi per dimensionamento operatori
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class DeterministicParams:
    """Parametri per calcoli deterministici"""
    call_volume: float  # Volume chiamate per periodo
    aht: float  # Average Handle Time in secondi
    shrinkage: float  # Shrinkage totale (0-1)
    period_duration: float  # Durata periodo in ore
    efficiency_factor: float  # Fattore di efficienza (0-1)
    service_time_variability: float  # Variabilità tempo servizio
    break_time: float  # Tempo pause in minuti per turno
    training_time: float  # Tempo formazione in ore per operatore
    overtime_threshold: float  # Soglia straordinari (occupazione)


@dataclass
class DeterministicResults:
    """Risultati calcoli deterministici"""
    agents_needed: int
    workload_hours: float
    occupancy: float
    productivity: float
    efficiency: float
    break_coverage: int
    training_coverage: int
    total_cost: float


class DeterministicCalculator:
    """
    Calcolatore deterministico avanzato per workforce planning
    Considera tutti i fattori operativi reali
    """
    
    def __init__(self):
        self.standard_shift_hours = 8
        self.weeks_per_month = 4.33
        self.days_per_week = 5
    
    def calculate_base_workload(self, call_volume: float, aht: float, 
                               period_duration: float = 1.0) -> float:
        """
        Calcola il carico di lavoro base in ore
        
        Args:
            call_volume: Numero di chiamate nel periodo
            aht: Average Handle Time in secondi
            period_duration: Durata del periodo in ore
            
        Returns:
            float: Ore di lavoro necessarie
        """
        if call_volume <= 0 or aht <= 0:
            return 0.0
        
        # Carico di lavoro grezzo in ore
        workload_seconds = call_volume * aht
        workload_hours = workload_seconds / 3600
        
        return workload_hours
    
    def calculate_shrinkage_components(self, base_shrinkage: float, 
                                     additional_factors: Dict[str, float] = None) -> float:
        """
        Calcola shrinkage totale considerando tutti i componenti
        
        Args:
            base_shrinkage: Shrinkage base (pause, formazione, etc.)
            additional_factors: Fattori aggiuntivi di shrinkage
            
        Returns:
            float: Shrinkage totale combinato
        """
        total_shrinkage = base_shrinkage
        
        if additional_factors:
            # Combina shrinkage usando formula: 1 - (1-s1)*(1-s2)*...
            combined_factor = 1 - base_shrinkage
            
            for factor_name, factor_value in additional_factors.items():
                combined_factor *= (1 - factor_value)
            
            total_shrinkage = 1 - combined_factor
        
        # Limita shrinkage massimo al 80%
        return min(0.80, max(0.0, total_shrinkage))
    
    def calculate_efficiency_factor(self, base_efficiency: float,
                                   service_variability: float,
                                   system_downtime: float = 0.0) -> float:
        """
        Calcola fattore di efficienza considerando variabilità
        
        Args:
            base_efficiency: Efficienza base operatori
            service_variability: Variabilità tempo di servizio
            system_downtime: Tempo inattività sistemi
            
        Returns:
            float: Fattore di efficienza corretto
        """
        # Penalità per variabilità (formula di Erlang)
        variability_penalty = service_variability * 0.1
        
        # Penalità per downtime sistemi
        downtime_penalty = system_downtime
        
        adjusted_efficiency = base_efficiency - variability_penalty - downtime_penalty
        
        return max(0.5, min(1.0, adjusted_efficiency))
    
    def calculate_break_coverage(self, total_agents: int, break_time_minutes: float,
                                shift_duration_hours: float = 8) -> int:
        """
        Calcola agenti aggiuntivi necessari per copertura pause
        
        Args:
            total_agents: Numero agenti base
            break_time_minutes: Tempo pause per turno in minuti
            shift_duration_hours: Durata turno in ore
            
        Returns:
            int: Agenti aggiuntivi per pause
        """
        if total_agents <= 0 or break_time_minutes <= 0:
            return 0
        
        # Percentuale di tempo in pausa
        break_percentage = break_time_minutes / (shift_duration_hours * 60)
        
        # Agenti aggiuntivi necessari
        additional_agents = total_agents * break_percentage
        
        return int(np.ceil(additional_agents))
    
    def calculate_training_coverage(self, total_agents: int, training_hours: float,
                                   monthly_training_requirement: float = 8) -> int:
        """
        Calcola agenti aggiuntivi per formazione
        
        Args:
            total_agents: Numero agenti base
            training_hours: Ore formazione per operatore per mese
            monthly_training_requirement: Ore formazione richieste al mese
            
        Returns:
            int: Agenti aggiuntivi per formazione
        """
        if total_agents <= 0 or training_hours <= 0:
            return 0
        
        # Percentuale di tempo in formazione
        monthly_work_hours = self.standard_shift_hours * self.days_per_week * self.weeks_per_month
        training_percentage = training_hours / monthly_work_hours
        
        # Agenti aggiuntivi necessari
        additional_agents = total_agents * training_percentage
        
        return int(np.ceil(additional_agents))
    
    def calculate_deterministic_agents(self, params: DeterministicParams) -> DeterministicResults:
        """
        Calcolo completo agenti necessari con approccio deterministico
        
        Args:
            params: Parametri del calcolo
            
        Returns:
            DeterministicResults: Risultati completi
        """
        # 1. Carico di lavoro base
        workload_hours = self.calculate_base_workload(
            params.call_volume, params.aht, params.period_duration
        )
        
        # 2. Shrinkage totale
        total_shrinkage = self.calculate_shrinkage_components(params.shrinkage)
        
        # 3. Fattore di efficienza
        efficiency = self.calculate_efficiency_factor(
            params.efficiency_factor, params.service_time_variability
        )
        
        # 4. Agenti base necessari
        if total_shrinkage >= 1.0:
            base_agents = float('inf')
        else:
            base_agents = workload_hours / ((1 - total_shrinkage) * efficiency * params.period_duration)
        
        # 5. Copertura pause
        break_coverage = self.calculate_break_coverage(
            int(np.ceil(base_agents)), params.break_time
        )
        
        # 6. Copertura formazione
        training_coverage = self.calculate_training_coverage(
            int(np.ceil(base_agents)), params.training_time
        )
        
        # 7. Totale agenti necessari
        total_agents = int(np.ceil(base_agents + break_coverage + training_coverage))
        
        # 8. Metriche finali
        actual_capacity = total_agents * (1 - total_shrinkage) * efficiency * params.period_duration
        occupancy = workload_hours / actual_capacity if actual_capacity > 0 else 1.0
        productivity = workload_hours / (total_agents * params.period_duration) if total_agents > 0 else 0.0
        
        return DeterministicResults(
            agents_needed=total_agents,
            workload_hours=workload_hours,
            occupancy=min(1.0, occupancy),
            productivity=productivity,
            efficiency=efficiency,
            break_coverage=break_coverage,
            training_coverage=training_coverage,
            total_cost=0.0  # Calcolato separatamente
        )
    
    def calculate_outbound_agents(self, target_contacts: int, contacts_per_hour: float,
                                 period_duration: float, shrinkage: float = 0.25,
                                 efficiency: float = 0.90) -> DeterministicResults:
        """
        Calcolo specifico per operazioni outbound
        
        Args:
            target_contacts: Numero contatti target nel periodo
            contacts_per_hour: Contatti per ora per operatore
            period_duration: Durata periodo in ore
            shrinkage: Shrinkage operatori
            efficiency: Efficienza operatori
            
        Returns:
            DeterministicResults: Risultati per outbound
        """
        if target_contacts <= 0 or contacts_per_hour <= 0:
            return DeterministicResults(0, 0.0, 0.0, 0.0, efficiency, 0, 0, 0.0)
        
        # Ore di lavoro necessarie
        required_hours = target_contacts / contacts_per_hour
        
        # Agenti necessari considerando shrinkage ed efficienza
        base_agents = required_hours / (period_duration * (1 - shrinkage) * efficiency)
        
        total_agents = int(np.ceil(base_agents))
        
        # Metriche
        actual_capacity = total_agents * period_duration * (1 - shrinkage) * efficiency
        occupancy = required_hours / actual_capacity if actual_capacity > 0 else 1.0
        productivity = required_hours / (total_agents * period_duration) if total_agents > 0 else 0.0
        
        return DeterministicResults(
            agents_needed=total_agents,
            workload_hours=required_hours,
            occupancy=min(1.0, occupancy),
            productivity=productivity,
            efficiency=efficiency,
            break_coverage=0,  # Incluso nel calcolo base per outbound
            training_coverage=0,
            total_cost=0.0
        )
    
    def workforce_optimization(self, base_params: DeterministicParams,
                              constraints: Dict[str, float]) -> Dict[str, DeterministicResults]:
        """
        Ottimizzazione workforce con vincoli multipli
        
        Args:
            base_params: Parametri base
            constraints: Vincoli operativi (max_occupancy, min_agents, etc.)
            
        Returns:
            Dict: Risultati per diversi scenari
        """
        results = {}
        
        # Scenario base
        results['base'] = self.calculate_deterministic_agents(base_params)
        
        # Scenario con vincolo occupazione
        if 'max_occupancy' in constraints:
            max_occ = constraints['max_occupancy']
            if results['base'].occupancy > max_occ:
                # Ricalcola con agenti aggiuntivi
                target_agents = int(np.ceil(results['base'].workload_hours / 
                                          (max_occ * base_params.period_duration)))
                
                adjusted_params = base_params
                adjusted_results = DeterministicResults(
                    agents_needed=target_agents,
                    workload_hours=results['base'].workload_hours,
                    occupancy=max_occ,
                    productivity=results['base'].workload_hours / (target_agents * base_params.period_duration),
                    efficiency=results['base'].efficiency,
                    break_coverage=results['base'].break_coverage,
                    training_coverage=results['base'].training_coverage,
                    total_cost=0.0
                )
                results['max_occupancy_constrained'] = adjusted_results
        
        # Scenario con agenti minimi
        if 'min_agents' in constraints:
            min_agents = int(constraints['min_agents'])
            if results['base'].agents_needed < min_agents:
                adjusted_occupancy = (results['base'].workload_hours / 
                                    (min_agents * base_params.period_duration))
                
                results['min_agents_constrained'] = DeterministicResults(
                    agents_needed=min_agents,
                    workload_hours=results['base'].workload_hours,
                    occupancy=min(1.0, adjusted_occupancy),
                    productivity=results['base'].workload_hours / (min_agents * base_params.period_duration),
                    efficiency=results['base'].efficiency,
                    break_coverage=results['base'].break_coverage,
                    training_coverage=results['base'].training_coverage,
                    total_cost=0.0
                )
        
        return results
    
    def sensitivity_analysis_deterministic(self, base_params: DeterministicParams,
                                         sensitivity_ranges: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Analisi di sensitività per modello deterministico
        
        Args:
            base_params: Parametri base
            sensitivity_ranges: Range valori da testare per ogni parametro
            
        Returns:
            pd.DataFrame: Risultati sensitivity analysis
        """
        results = []
        
        for param_name, param_values in sensitivity_ranges.items():
            for value in param_values:
                # Crea parametri modificati
                modified_params = DeterministicParams(
                    call_volume=base_params.call_volume,
                    aht=base_params.aht,
                    shrinkage=base_params.shrinkage,
                    period_duration=base_params.period_duration,
                    efficiency_factor=base_params.efficiency_factor,
                    service_time_variability=base_params.service_time_variability,
                    break_time=base_params.break_time,
                    training_time=base_params.training_time,
                    overtime_threshold=base_params.overtime_threshold
                )
                
                # Modifica il parametro specifico
                setattr(modified_params, param_name, value)
                
                # Calcola risultati
                result = self.calculate_deterministic_agents(modified_params)
                
                results.append({
                    'parameter': param_name,
                    'value': value,
                    'agents_needed': result.agents_needed,
                    'occupancy': result.occupancy,
                    'productivity': result.productivity,
                    'efficiency': result.efficiency
                })
        
        return pd.DataFrame(results)


# Istanza globale del calcolatore
deterministic_calculator = DeterministicCalculator()


def calculate_deterministic(call_volume, aht, shrinkage=0.25, period_duration=1.0,
                          efficiency_factor=0.90, break_time=60, training_time=8, 
                          ore_settimanali_fte=37.5):
    """
    Funzione wrapper per calcolo deterministico
    Mantiene compatibilità con codice esistente e considera tutti i parametri operativi
    
    Args:
        call_volume: Volume chiamate nel periodo
        aht: Average Handle Time in secondi
        shrinkage: Shrinkage operatori (0-1) - include pause, formazione, inefficienze
        period_duration: Durata periodo in ore
        efficiency_factor: Fattore efficienza operatori (0-1)
        break_time: Tempo pause in minuti per turno
        training_time: Ore formazione per operatore per mese
        ore_settimanali_fte: Ore settimanali lavorative per FTE
        
    Returns:
        tuple: (agenti_necessari_totali, service_level_dummy, occupazione)
    """
    params = DeterministicParams(
        call_volume=call_volume,
        aht=aht,
        shrinkage=shrinkage,
        period_duration=period_duration,
        efficiency_factor=efficiency_factor,
        service_time_variability=0.15,  # Default variability
        break_time=break_time,
        training_time=training_time,
        overtime_threshold=0.85
    )
    
    results = deterministic_calculator.calculate_deterministic_agents(params)
    
    # Mantiene compatibilità: service_level=1.0 per deterministico
    return results.agents_needed, 1.0, results.occupancy


def calculate_outbound_deterministic(target_contacts, contacts_per_hour, period_duration=1.0,
                                   shrinkage=0.25, efficiency=0.90, ore_settimanali_fte=37.5):
    """
    Funzione wrapper per calcolo outbound deterministico
    Considera shrinkage e tutti i parametri operativi per campagne outbound
    
    Args:
        target_contacts: Numero contatti target da raggiungere
        contacts_per_hour: Contatti completati per ora per operatore
        period_duration: Durata periodo in ore
        shrinkage: Shrinkage operatori (pause, formazione, inefficienze)
        efficiency: Efficienza operatori nella realizzazione contatti
        ore_settimanali_fte: Ore settimanali lavorative per FTE
        
    Returns:
        tuple: (agenti_necessari_totali, success_rate, occupazione)
    """
    results = deterministic_calculator.calculate_outbound_agents(
        target_contacts, contacts_per_hour, period_duration, shrinkage, efficiency
    )
    
    # Per outbound: success_rate al posto di service_level
    success_rate = 0.95  # Tipico per outbound
    
    return results.agents_needed, success_rate, results.occupancy
