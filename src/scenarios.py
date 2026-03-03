from itertools import product
from dataclasses import dataclass
from src.data_generation import constant_dgp, heterogeneous_dgp
from src.estimation import CausalTreeEvaluation

@dataclass(frozen=True)
class SimulationScenario:
    """Data class to define the simulation scenarios."""
    
    name: str
    dgp: type[constant_dgp | heterogeneous_dgp]  
    dgp_params: dict
    estimator: type[CausalTreeEvaluation] 
    sample_size: int
    splitting: str
    n_simulations: int = 25000
    first_seed: int = 1

dgps = [
    (constant_dgp, 'uniform', 'linear'),
    # (constant_dgp, 'uniform', 'nonlinear'),
    # (heterogeneous_dgp, 'uniform', 'linear', 'first_feature'),
    # (heterogeneous_dgp, 'normal', 'linear', 'positive_features'),
    # (heterogeneous_dgp, 'uniform', 'nonlinear', 'first_feature'),
    # (heterogeneous_dgp, 'normal', 'nonlinear', 'positive_features'),
]

estimators = [CausalTreeEvaluation]

sample_sizes = [1000]

splitting_types = ["adaptive"]

scenarios = []

for dgp_tuple, estimator_class, size, split in product(dgps, estimators, sample_sizes, splitting_types):
    
    dgp_class = dgp_tuple[0]
    dist = dgp_tuple[1]
    outcome = dgp_tuple[2]
    
    dgp_params = {
        "distribution": dist, 
        "outcome_mechanism": outcome
    }
    
    treatment = ""
    if len(dgp_tuple) == 4:
        treatment = dgp_tuple[3]
        dgp_params["treatment_mechanism"] = treatment

    treatment_str = f"_{treatment}" if treatment else ""
    scenario_name = f"{dgp_class.__name__}_{outcome}{treatment_str}_{split}_{size}"

    scenario = SimulationScenario(
        name=scenario_name,
        dgp=dgp_class,
        dgp_params=dgp_params,
        estimator=estimator_class,
        sample_size=size,
        splitting=split,
    )
    
    scenarios.append(scenario)