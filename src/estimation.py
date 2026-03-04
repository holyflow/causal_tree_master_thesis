import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from econml.grf import CausalForest

class CausalTreeEvaluation:
    """Data class to store results from a Causal Tree simulation."""
    def __init__(self) -> None:
        self.n_estimators = 1
        self.subforest_size = 1
        self.inference = False
        self.criterion = 'mse'
        self.honest = False
        self.max_depth = list(range(1, 11))
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.max_samples = 1.0
        self.min_balancedness_tol = 0.5

    def simulate_and_evaluate_causal_tree_adaptive(self, X_sim:np.ndarray, Y_sim:np.ndarray, T_sim:np.ndarray, treatment_effect:np.ndarray, random_state: int) -> pd.DataFrame:
        x_sim_mean = X_sim.mean().item()

        results = []

        for depth in self.max_depth:
            causal_tree = CausalForest(
                n_estimators=self.n_estimators,     
                subforest_size=self.subforest_size,
                inference=self.inference,
                criterion=self.criterion,
                honest=self.honest,
                random_state=random_state,
                max_depth=depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_samples=self.max_samples,
                min_balancedness_tol=self.min_balancedness_tol,
            )

            causal_tree.fit(X=X_sim, T=T_sim, y=Y_sim)

            leaf_ids = causal_tree.apply(X_sim)

            unique_leaves = np.unique(leaf_ids)
            
            for leaf_id in unique_leaves:
                mask = (leaf_ids[:, 0] == leaf_id)

                x_leaf = X_sim[mask]
                y_leaf = Y_sim[mask]
                t_leaf = T_sim[mask]
                te_leaf = treatment_effect[mask]

                n_t = np.sum(t_leaf == 1)
                n_c = np.sum(t_leaf == 0)

                true_mean_te = te_leaf.mean()

                abs_dist_from_center = np.mean(np.abs(x_leaf - x_sim_mean))
                rms_dist_from_center = np.sqrt(np.mean(np.square(x_leaf - x_sim_mean)))

                if n_t > 1 and n_c > 1:
                    mu_t = y_leaf[t_leaf == 1].mean()
                    mu_c = y_leaf[t_leaf == 0].mean()
                    
                    cate_est = mu_t - mu_c

                    var_t = y_leaf[t_leaf == 1].var(ddof=1)
                    var_c = y_leaf[t_leaf == 0].var(ddof=1)
                    se_cate = np.sqrt((var_t / n_t) + (var_c / n_c))

                    ci_lower = cate_est - 1.96 * se_cate
                    ci_upper = cate_est + 1.96 * se_cate

                    coverage = int(ci_lower <= true_mean_te <= ci_upper)

                else:
                    cate_est = np.nan
                    se_cate = np.nan
                    ci_lower = np.nan
                    ci_upper = np.nan
                    coverage = np.nan
                 
                results.append({
                    'leaf_id': leaf_id,
                    'depth': depth,
                    'CATE': cate_est,
                    'standard_error': se_cate,
                    'n_samples': len(y_leaf),
                    'n_treated': n_t,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'true_treatment_effect': true_mean_te,
                    'coverage': coverage,
                    'x_leaf_mean': x_leaf.mean().item(),
                    'x_struct_mean': x_sim_mean,
                    'abs_dist_from_center': abs_dist_from_center,
                    'rms_dist_from_center': rms_dist_from_center
                })

        return pd.DataFrame(results)

    def simulate_and_evaluate_causal_tree_honest(self, X_sim:np.ndarray, Y_sim:np.ndarray, T_sim:np.ndarray, treatment_effect:np.ndarray, random_state: int) -> pd.DataFrame:
        X_struct, X_est, y_struct, y_est, t_struct, t_est, treatment_effect_struct, treatment_effect_est = train_test_split(
            X_sim, Y_sim, T_sim, treatment_effect, test_size=0.5, random_state=random_state
        )

        x_struct_mean = X_struct.mean().item()

        results = []

        for depth in self.max_depth:
            causal_tree = CausalForest(
                n_estimators=self.n_estimators,     
                subforest_size=self.subforest_size,
                inference=self.inference,
                criterion=self.criterion,
                honest=self.honest,
                random_state=random_state,
                max_depth=depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_samples=self.max_samples,
                min_balancedness_tol=self.min_balancedness_tol,
            )

            causal_tree.fit(X=X_struct, T=t_struct, y=y_struct)

            leaf_ids_est = causal_tree.apply(X_est)
            leaf_ids_struct = causal_tree.apply(X_struct)

            unique_leaves = np.unique(leaf_ids_est)
            
            for leaf_id in unique_leaves:
                mask_struct = (leaf_ids_struct[:, 0] == leaf_id)
                mask_est = (leaf_ids_est[:, 0] == leaf_id)

                x_leaf_struct = X_struct[mask_struct]
                y_leaf_est = y_est[mask_est]
                t_leaf_est = t_est[mask_est]
                te_leaf_est = treatment_effect_est[mask_est]

                n_t = np.sum(t_leaf_est == 1)
                n_c = np.sum(t_leaf_est == 0)

                true_mean_te = te_leaf_est.mean()

                abs_dist_from_center = np.mean(np.abs(x_leaf_struct - x_struct_mean))
                rms_dist_from_center = np.sqrt(np.mean(np.square(x_leaf_struct - x_struct_mean)))

                if n_t > 1 and n_c > 1:
                    mu_t = y_leaf_est[t_leaf_est == 1].mean()
                    mu_c = y_leaf_est[t_leaf_est == 0].mean()
                    
                    cate_est = mu_t - mu_c

                    var_t = y_leaf_est[t_leaf_est == 1].var(ddof=1)
                    var_c = y_leaf_est[t_leaf_est == 0].var(ddof=1)
                    se_cate = np.sqrt((var_t / n_t) + (var_c / n_c))

                    ci_lower = cate_est - 1.96 * se_cate
                    ci_upper = cate_est + 1.96 * se_cate

                    coverage = int(ci_lower <= true_mean_te <= ci_upper)

                else:
                    cate_est = np.nan
                    se_cate = np.nan
                    ci_lower = np.nan
                    ci_upper = np.nan
                    coverage = np.nan

                results.append({
                    'leaf_id': leaf_id,
                    'depth': depth,
                    'CATE': cate_est,
                    'standard_error': se_cate,
                    'n_samples': len(y_leaf_est),
                    'n_treated': n_t,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'true_treatment_effect': true_mean_te,
                    'coverage': coverage,
                    'x_leaf_mean': x_leaf_struct.mean().item(),
                    'x_struct_mean': x_struct_mean,
                    'abs_dist_from_center': abs_dist_from_center,
                    'rms_dist_from_center': rms_dist_from_center
                })

        return pd.DataFrame(results)