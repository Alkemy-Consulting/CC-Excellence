"""
Modulo per simulazione Monte Carlo di sistemi di code
Utilizza SimPy per simulazioni discrete event-driven accurate
"""

import numpy as np
import pandas as pd
import simpy
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SimulationParams:
    """Parametri per la simulazione"""
    arrival_rate: float  # Chiamate per ora
    aht_mean: float  # AHT medio in secondi
    aht_std: float  # Deviazione standard AHT
    service_level_target: float  # Target SL (0-1)
    answer_time_target: float  # Target risposta in secondi
    max_occupancy: float  # Max occupazione
    patience_mean: float  # Pazienza media clienti
    patience_std: float  # Deviazione standard pazienza
    shrinkage: float  # Shrinkage operatori
    simulation_time: float  # Tempo simulazione in ore
    num_replications: int  # Numero di repliche


@dataclass
class SimulationResults:
    """Risultati della simulazione"""
    agents_needed: int
    service_level: float
    occupancy: float
    average_wait_time: float
    abandon_rate: float
    queue_length_avg: float
    queue_length_max: int
    confidence_interval_95: Tuple[float, float]


class CallCenterSimulation:
    """
    Simulatore Monte Carlo per Contact Center
    Utilizza SimPy per simulazioni discrete event-driven
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.results = defaultdict(list)
        
    def setup_simulation(self, env, num_agents):
        """Setup della simulazione con agenti e code"""
        # Risorsa agenti
        self.agents = simpy.Resource(env, capacity=num_agents)
        
        # Metriche
        self.calls_arrived = 0
        self.calls_answered = 0
        self.calls_abandoned = 0
        self.wait_times = []
        self.queue_lengths = []
        self.service_times = []
        
        # Monitoraggio code
        env.process(self.monitor_queue(env))
        
    def monitor_queue(self, env):
        """Monitora la lunghezza della coda"""
        while True:
            queue_length = len(self.agents.queue)
            self.queue_lengths.append(queue_length)
            yield env.timeout(1)  # Monitora ogni secondo
            
    def call_arrival_process(self, env):
        """Processo di arrivo delle chiamate"""
        while True:
            # Tempo tra arrivi (Poisson process)
            inter_arrival_time = np.random.exponential(3600 / self.params.arrival_rate)
            yield env.timeout(inter_arrival_time / 3600)  # Converti in ore
            
            # Genera nuova chiamata
            self.calls_arrived += 1
            env.process(self.handle_call(env, self.calls_arrived))
    
    def handle_call(self, env, call_id):
        """Gestisce una singola chiamata"""
        arrival_time = env.now
        
        # Genera pazienza del cliente
        patience = max(1, np.random.normal(
            self.params.patience_mean, self.params.patience_std
        )) / 3600  # Converti in ore
        
        # Richiedi agente con timeout per abbandono
        with self.agents.request() as request:
            try:
                # Attesa per agente disponibile
                start_wait = env.now
                yield request | env.timeout(patience)
                
                if request.triggered:
                    # Chiamata servita
                    wait_time = (env.now - start_wait) * 3600  # In secondi
                    self.wait_times.append(wait_time)
                    self.calls_answered += 1
                    
                    # Tempo di servizio
                    service_time = max(1, np.random.normal(
                        self.params.aht_mean, self.params.aht_std
                    )) / 3600  # Converti in ore
                    
                    self.service_times.append(service_time * 3600)
                    yield env.timeout(service_time)
                    
                else:
                    # Chiamata abbandonata
                    self.calls_abandoned += 1
                    
            except simpy.Interrupt:
                # Interruzione del processo
                pass
    
    def run_single_replication(self, num_agents) -> Dict:
        """Esegue una singola replica della simulazione"""
        # Setup ambiente SimPy
        env = simpy.Environment()
        self.setup_simulation(env, num_agents)
        
        # Avvia processo di arrivo chiamate
        env.process(self.call_arrival_process(env))
        
        # Esegui simulazione
        env.run(until=self.params.simulation_time)
        
        # Calcola metriche
        total_calls = self.calls_arrived
        if total_calls == 0:
            return self._empty_results()
        
        service_level = self._calculate_service_level()
        occupancy = self._calculate_occupancy(num_agents)
        abandon_rate = self.calls_abandoned / total_calls if total_calls > 0 else 0
        avg_wait_time = np.mean(self.wait_times) if self.wait_times else 0
        avg_queue_length = np.mean(self.queue_lengths) if self.queue_lengths else 0
        max_queue_length = max(self.queue_lengths) if self.queue_lengths else 0
        
        return {
            'service_level': service_level,
            'occupancy': occupancy,
            'abandon_rate': abandon_rate,
            'avg_wait_time': avg_wait_time,
            'avg_queue_length': avg_queue_length,
            'max_queue_length': max_queue_length,
            'total_calls': total_calls,
            'calls_answered': self.calls_answered,
            'calls_abandoned': self.calls_abandoned
        }
    
    def _calculate_service_level(self) -> float:
        """Calcola il service level"""
        if not self.wait_times:
            return 0.0
        
        answered_in_time = sum(1 for wt in self.wait_times 
                              if wt <= self.params.answer_time_target)
        return answered_in_time / len(self.wait_times)
    
    def _calculate_occupancy(self, num_agents) -> float:
        """Calcola l'occupazione media degli agenti"""
        if not self.service_times or num_agents == 0:
            return 0.0
        
        total_service_time = sum(self.service_times) / 3600  # In ore
        total_agent_time = num_agents * self.params.simulation_time
        
        return min(1.0, total_service_time / total_agent_time)
    
    def _empty_results(self) -> Dict:
        """Risultati vuoti per simulazioni fallite"""
        return {
            'service_level': 0.0,
            'occupancy': 0.0,
            'abandon_rate': 1.0,
            'avg_wait_time': 999.0,
            'avg_queue_length': 0.0,
            'max_queue_length': 0,
            'total_calls': 0,
            'calls_answered': 0,
            'calls_abandoned': 0
        }
    
    def find_optimal_agents(self) -> SimulationResults:
        """
        Trova il numero ottimale di agenti usando simulazione
        
        Returns:
            SimulationResults: Risultati ottimizzazione
        """
        # Stima iniziale basata su teoria delle code
        traffic_intensity = (self.params.arrival_rate * self.params.aht_mean) / 3600
        min_agents = max(1, int(np.ceil(traffic_intensity * 1.1)))
        max_agents = min_agents + 15
        
        best_agents = min_agents
        best_results = None
        
        for num_agents in range(min_agents, max_agents + 1):
            # Esegui multiple repliche
            replication_results = []
            
            for rep in range(self.params.num_replications):
                # Reset per nuova replica
                self.calls_arrived = 0
                self.calls_answered = 0
                self.calls_abandoned = 0
                self.wait_times = []
                self.queue_lengths = []
                self.service_times = []
                
                # Esegui replica
                result = self.run_single_replication(num_agents)
                replication_results.append(result)
            
            # Calcola statistiche aggregate
            avg_results = self._aggregate_results(replication_results)
            
            # Verifica se soddisfa i vincoli
            if (avg_results['service_level'] >= self.params.service_level_target and
                avg_results['occupancy'] <= self.params.max_occupancy):
                
                best_agents = num_agents
                best_results = avg_results
                break
            
            # Aggiorna il migliore risultato comunque
            if best_results is None or self._is_better_result(avg_results, best_results):
                best_agents = num_agents
                best_results = avg_results
        
        # Calcola intervallo di confidenza
        confidence_interval = self._calculate_confidence_interval(replication_results)
        
        return SimulationResults(
            agents_needed=best_agents,
            service_level=best_results['service_level'],
            occupancy=best_results['occupancy'],
            average_wait_time=best_results['avg_wait_time'],
            abandon_rate=best_results['abandon_rate'],
            queue_length_avg=best_results['avg_queue_length'],
            queue_length_max=int(best_results['max_queue_length']),
            confidence_interval_95=confidence_interval
        )
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Aggrega risultati di multiple repliche"""
        if not results:
            return self._empty_results()
        
        aggregated = {}
        for key in results[0].keys():
            values = [r[key] for r in results if key in r]
            aggregated[key] = np.mean(values) if values else 0.0
        
        return aggregated
    
    def _is_better_result(self, new_result: Dict, current_best: Dict) -> bool:
        """Determina se il nuovo risultato è migliore"""
        # Priorità: Service Level > Occupancy < Abandon Rate
        if new_result['service_level'] > current_best['service_level']:
            return True
        elif (new_result['service_level'] == current_best['service_level'] and
              new_result['abandon_rate'] < current_best['abandon_rate']):
            return True
        return False
    
    def _calculate_confidence_interval(self, results: List[Dict]) -> Tuple[float, float]:
        """Calcola intervallo di confidenza 95% per service level"""
        if len(results) < 2:
            return (0.0, 1.0)
        
        service_levels = [r['service_level'] for r in results]
        mean_sl = np.mean(service_levels)
        std_sl = np.std(service_levels, ddof=1)
        
        # t-distribution per IC 95%
        from scipy.stats import t
        t_value = t.ppf(0.975, len(results) - 1)
        margin_error = t_value * (std_sl / np.sqrt(len(results)))
        
        return (max(0, mean_sl - margin_error), min(1, mean_sl + margin_error))


class SimulationCalculator:
    """
    Calcolatore principale per simulazioni Monte Carlo
    """
    
    def __init__(self):
        self.default_simulation_time = 24  # 24 ore
        self.default_replications = 10  # Numero repliche
    
    def calculate_agents_simulation(self, arrival_rate, aht, service_level_target,
                                  answer_time_target=20, max_occupancy=0.85,
                                  patience_mean=90, shrinkage=0.25,
                                  aht_variability=0.15, patience_variability=0.3,
                                  simulation_time=None, num_replications=None):
        """
        Calcola agenti necessari usando simulazione Monte Carlo
        
        Args:
            arrival_rate: Chiamate per ora
            aht: Average Handle Time medio in secondi
            service_level_target: Target Service Level (0-1)
            answer_time_target: Tempo target risposta in secondi
            max_occupancy: Massima occupazione accettabile
            patience_mean: Pazienza media clienti in secondi
            shrinkage: Shrinkage operatori (0-1)
            aht_variability: Variabilità AHT (coefficiente di variazione)
            patience_variability: Variabilità pazienza
            simulation_time: Tempo simulazione in ore
            num_replications: Numero repliche
            
        Returns:
            tuple: (agenti_necessari, service_level, occupazione)
        """
        if arrival_rate <= 0:
            return 0, 1.0, 0.0
        
        # Parametri simulazione
        params = SimulationParams(
            arrival_rate=arrival_rate,
            aht_mean=aht,
            aht_std=aht * aht_variability,
            service_level_target=service_level_target,
            answer_time_target=answer_time_target,
            max_occupancy=max_occupancy,
            patience_mean=patience_mean,
            patience_std=patience_mean * patience_variability,
            shrinkage=shrinkage,
            simulation_time=simulation_time or self.default_simulation_time,
            num_replications=num_replications or self.default_replications
        )
        
        # Esegui simulazione
        try:
            simulator = CallCenterSimulation(params)
            results = simulator.find_optimal_agents()
            
            # Applica shrinkage
            agents_with_shrinkage = int(np.ceil(results.agents_needed / (1 - shrinkage)))
            
            return agents_with_shrinkage, results.service_level, results.occupancy
            
        except Exception as e:
            warnings.warn(f"Errore simulazione: {e}. Uso fallback deterministico.")
            return self._deterministic_fallback(arrival_rate, aht, shrinkage)
    
    def _deterministic_fallback(self, arrival_rate, aht, shrinkage):
        """Fallback deterministico in caso di errore"""
        workload_hours = arrival_rate * aht / 3600
        agents_needed = workload_hours / (1 - shrinkage)
        occupancy = workload_hours / agents_needed if agents_needed > 0 else 0
        
        return int(np.ceil(agents_needed)), 0.95, occupancy
    
    def stress_test_simulation(self, base_params, stress_scenarios):
        """
        Esegue stress test con scenari multipli
        
        Args:
            base_params: Parametri base
            stress_scenarios: Lista scenari di stress
            
        Returns:
            Dict: Risultati stress test
        """
        results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            # Modifica parametri per scenario
            modified_params = base_params.copy()
            modified_params.update(scenario_params)
            
            # Esegui simulazione
            agents, sl, occ = self.calculate_agents_simulation(**modified_params)
            
            results[scenario_name] = {
                'agents': agents,
                'service_level': sl,
                'occupancy': occ,
                'scenario_params': scenario_params
            }
        
        return results


# Istanza globale del calcolatore
simulation_calculator = SimulationCalculator()


def calculate_simulation(arrival_rate, aht, service_level_target, answer_time_target=20, 
                        max_occupancy=0.85, shrinkage=0.25, ore_settimanali_fte=37.5, 
                        patience=90, num_simulations=10):
    """
    Funzione wrapper per simulazione
    Mantiene compatibilità con codice esistente e applica tutti i parametri operativi
    
    Args:
        arrival_rate: Chiamate per ora
        aht: Average Handle Time in secondi
        service_level_target: Target Service Level (0-1)
        answer_time_target: Tempo target risposta in secondi
        max_occupancy: Massima occupazione accettabile (0-1)
        shrinkage: Shrinkage operatori (0-1)
        ore_settimanali_fte: Ore settimanali per FTE per conversione
        patience: Pazienza media clienti in secondi
        num_simulations: Numero di simulazioni da eseguire
        
    Returns:
        tuple: (agenti_necessari_con_shrinkage, service_level, occupazione)
    """
    return simulation_calculator.calculate_agents_simulation(
        arrival_rate=arrival_rate,
        aht=aht,
        service_level_target=service_level_target,
        answer_time_target=answer_time_target,
        max_occupancy=max_occupancy,
        shrinkage=shrinkage,
        patience_mean=patience,
        num_replications=num_simulations
    )
