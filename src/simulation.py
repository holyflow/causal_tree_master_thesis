import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from src.data_generation import constant_dgp, heterogeneous_dgp
from src.causalml_estimation import CausalTreeEvaluation

class SimulationRunner:
    """Class to run simulations for a given DGP and estimator in parallel."""
    
    def __init__(
        self, 
        dgp: constant_dgp | heterogeneous_dgp,
        estimator: CausalTreeEvaluation,
    ) -> None:
        self.dgp = dgp
        self.estimator = estimator

    def _run_single_simulation(self, sim_idx: int, seed: int, n_obs: int, dgp_args: dict, splitting: str) -> pd.DataFrame:

        if (sim_idx + 1) % 2000 == 0:
            print(f"Completed {sim_idx + 1} simulations...", flush=True)
        
        sim_seed = seed + sim_idx
        
        X_sim, Y_sim, T_sim, treatment_effect = self.dgp.sample(
            n_obs=n_obs, 
            seed=sim_seed, 
            **dgp_args
        )

        if splitting == "honest":
            sim_result = self.estimator.simulate_and_evaluate_causal_tree_honest(
                X_sim, Y_sim, T_sim, treatment_effect,
                random_state=sim_seed
            )
        elif splitting == "adaptive":
            sim_result = self.estimator.simulate_and_evaluate_causal_tree_adaptive(
                X_sim, Y_sim, T_sim, treatment_effect,
                random_state=sim_seed
            )
        else:
            raise ValueError(f"Unknown splitting type: {splitting}")
        
        sim_result['simulation'] = sim_idx
        sim_result['splitting_type'] = splitting 

        return sim_result

    def run_simulation(
        self, 
        n_simulations: int, 
        n_obs: int, 
        seed: int, 
        splitting: str, 
        distribution: str, 
        outcome_mechanism: str,
        treatment_mechanism: str = None,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        
        dgp_args = {
            "distribution": distribution,
            "outcome_mechanism": outcome_mechanism,
            "treatment_mechanism": treatment_mechanism
        }

        print(f"Starting {n_simulations} {splitting} simulations using {n_jobs} cores...")
        
        
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_simulation)(
                sim_idx=sim, 
                seed=seed, 
                n_obs=n_obs, 
                dgp_args=dgp_args,
                splitting=splitting
            ) 
            for sim in range(n_simulations)
        )

        results_df = pd.concat(all_results, ignore_index=True)
        return results_df