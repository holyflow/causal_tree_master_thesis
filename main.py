from src.orchestrator import SimulationOrchestrator
from src.scenarios import scenarios
from src.plotting import plot_coverage, plot_adj_se_treatment_dev, plot_leaf_frequency, plot_n_samples_frequency
from src.tables import coverage_table, leaf_size_table, populated_leaves_table
from src.config import SAMPLE_SIZES, RESULTS_DIR


if __name__ == "__main__":   
    for sample_size in SAMPLE_SIZES:
        
        current_scenarios = [s for s in scenarios if s.sample_size == sample_size]
        
        orchestrator = SimulationOrchestrator(current_scenarios)
        results_df = orchestrator.run_all()

        filename = f"25000_simulation_results_{sample_size}_depth_1_30_combined_treat_control_non.pkl"
        results_df.to_pickle(RESULTS_DIR / filename)

        plot_coverage(results_df, sample_size, "whole_support")
        plot_coverage(results_df, sample_size, "abs_deviation")
        plot_adj_se_treatment_dev(results_df, sample_size, "whole_support")
        plot_adj_se_treatment_dev(results_df, sample_size, "abs_deviation")
        plot_leaf_frequency(results_df, sample_size)
        plot_n_samples_frequency(results_df, sample_size)
        coverage_table(results_df, sample_size)
        leaf_size_table(results_df, sample_size)
        populated_leaves_table(results_df, sample_size)