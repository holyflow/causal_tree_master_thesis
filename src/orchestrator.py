import pandas as pd
import numpy as np
from .simulation import SimulationRunner
from .scenarios import SimulationScenario

class SimulationOrchestrator:
    """Simple simulation orchestration class."""
    
    def __init__(self, scenarios: list[SimulationScenario]):
        self.scenarios = scenarios 

    def run_all(self) -> pd.DataFrame:
        all_results = []
        
        for scenario in self.scenarios: 
            print(f"Running scenario: {scenario.name}")

            estimator_instance = scenario.estimator(sample_size=scenario.sample_size)

            dgp_instance = scenario.dgp()

            splitting_instance = scenario.splitting

            runner = SimulationRunner(estimator=estimator_instance,dgp=dgp_instance)

            results = runner.run_simulation(
                n_simulations=scenario.n_simulations, 
                n_obs=scenario.sample_size, 
                seed=scenario.first_seed,
                splitting=splitting_instance,
            )

            results['scenario'] = scenario.name

            float_cols = results.select_dtypes(include=['float64']).columns
            results[float_cols] = results[float_cols].astype(np.float32)
            
            int_cols = results.select_dtypes(include=['int64']).columns
            results[int_cols] = results[int_cols].astype(np.int32)

            all_results.append(results)

        return pd.concat(all_results, ignore_index=True)