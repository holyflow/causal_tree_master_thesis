import pandas as pd
import numpy as np
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

            # Downcast to 32-bit before appending to the list
            float_cols = results.select_dtypes(include=['float64']).columns
            results[float_cols] = results[float_cols].astype(np.float32)
            
            int_cols = results.select_dtypes(include=['int64']).columns
            results[int_cols] = results[int_cols].astype(np.int32)

            all_results.append(results)

        return pd.concat(all_results, ignore_index=True)