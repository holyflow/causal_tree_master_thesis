import pandas as pd
from src.simulation import SimulationRunner
from src.scenarios import SimulationScenario, scenarios 

class SimulationOrchestrator:
    """Simple simulation orchestration class."""
    
    def __init__(self, scenarios: list[SimulationScenario]):
        self.scenarios = scenarios 

    def run_all(self) -> pd.DataFrame:
        all_results = []
        
        for scenario in self.scenarios: 
            print(f"Running scenario: {scenario.name}")

            dgp_instance = scenario.dgp() 

            estimator_instance = scenario.estimator()

            splitting_instance = scenario.splitting

            runner = SimulationRunner(dgp=dgp_instance, estimator=estimator_instance)

            results = runner.run_simulation(
                n_simulations=scenario.n_simulations, 
                n_obs=scenario.sample_size, 
                seed=scenario.first_seed,
                splitting=splitting_instance,
                **scenario.dgp_params 
            )

            results['scenario'] = scenario.name
            all_results.append(results)

        return pd.concat(all_results, ignore_index=True)