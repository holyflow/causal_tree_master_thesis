from itertools import product
from dataclasses import dataclass
from .data_generation import constant_dgp
from .causalml_estimation import CausalTreeEvaluation
from .config import SAMPLE_SIZES,SPLITTING_TYPES

@dataclass(frozen=True)
class SimulationScenario:
    """Data class to define the simulation scenarios."""
    
    name: str
    dgp: type[constant_dgp]  
    estimator: type[CausalTreeEvaluation] 
    sample_size: int
    splitting: str
    n_simulations: int = 25000
    first_seed: int = 1


estimators = [CausalTreeEvaluation]

sample_size = SAMPLE_SIZES

scenarios = []

for estimator_class, size, split in product(estimators, sample_size, SPLITTING_TYPES):

    scenario_name = f"{split}_{size}"

    scenario = SimulationScenario(
        name=scenario_name,
        estimator=estimator_class,
        sample_size=size,
        splitting=split,
        dgp=constant_dgp
    )
    
    scenarios.append(scenario)