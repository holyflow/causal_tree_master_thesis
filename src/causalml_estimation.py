import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from causalml.inference.tree import CausalTreeRegressor # type: ignore
from .config import TREE_DEPTHS


class CausalTreeEvaluation:
    """Data class to store results from a Causal Tree simulation."""
    def __init__(self, sample_size: int) -> None:
        self.criterion = 'causal_mse'
        self.max_depth = TREE_DEPTHS[sample_size]
        self.min_samples_split = 8
        self.min_samples_leaf = 4
        self.min_group_samples = 0
        self.control_name = 0
        self.groups_penalty = 0.0
        self.min_treated_leaf = 2
        self.min_control_leaf = 2

    def simulate_and_evaluate_causal_tree_adaptive(self, X_sim:np.ndarray, Y_sim:np.ndarray, T_sim:np.ndarray, treatment_effect:np.ndarray, random_state: int) -> pd.DataFrame:

        x_sim_mean = X_sim.mean().item()
        T_sim_mean = T_sim.mean().item()

        results = []

        for depth in self.max_depth:
            causal_tree = CausalTreeRegressor(
                max_depth=depth,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_group_samples=self.min_group_samples,
                control_name=self.control_name,     
                random_state=random_state,
                groups_penalty=self.groups_penalty,
                min_treated_leaf=self.min_treated_leaf,
                min_control_leaf=self.min_control_leaf,
            )

            causal_tree.fit(X=X_sim, treatment=T_sim, y=Y_sim)

            te_pred = causal_tree.predict(X_sim)

            te_pred_round = np.round(te_pred, decimals=6)

            unique_preds = np.unique(te_pred_round)
            cate_to_leaf_id = {val: idx for idx, val in enumerate(unique_preds)}

            leaf_ids = np.array([cate_to_leaf_id.get(val, -1) for val in te_pred_round])

            unique_leaves = np.unique(leaf_ids)

            for leaf_id in unique_leaves:
                    mask = (leaf_ids == leaf_id)

                    x_leaf = X_sim[mask]
                    y_leaf = Y_sim[mask]
                    t_leaf = T_sim[mask]
                    te_leaf = treatment_effect[mask]

                    n_t = np.sum(t_leaf == 1)
                    n_c = np.sum(t_leaf == 0)

                    true_mean_te = te_leaf.mean()

                    abs_dist_from_center = np.mean(np.abs(x_leaf - 50))
                    rms_dist_from_center = np.sqrt(np.mean(np.square(x_leaf - 50)))

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

                    elif n_t >= 1 and n_c >= 1:
                        mu_t = y_leaf[t_leaf == 1].mean()
                        mu_c = y_leaf[t_leaf == 0].mean()
                        
                        cate_est = mu_t - mu_c

                        se_cate = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan
                        coverage = np.nan

                    else:
                        cate_est = np.nan
                        se_cate = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan
                        coverage = np.nan
                        
                    results.append({
                        'leaf_id': leaf_id,
                        'depth': depth,
                        'CATE_diff_in_mean': cate_est,
                        "CATE_pred": te_pred[mask].mean(),
                        'true_te': treatment_effect.mean(),
                        'abs_cate_deviation': np.abs(cate_est - treatment_effect.mean()),
                        'standard_error': se_cate,
                        'standard_error_adj': 1.96 * se_cate,
                        'coverage': coverage,
                        'n_samples': len(y_leaf),
                        'n_treated': n_t,
                        'n_control': n_c,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'true_treatment_effect': true_mean_te,
                        'x_leaf_mean': x_leaf.mean().item(),
                        'x_struct_mean': x_sim_mean,
                        'x_leaf_dist_max_min': x_leaf.max() - x_leaf.min(),
                        't_struct_mean': T_sim_mean,
                        'abs_dist_from_center': abs_dist_from_center,
                        'rms_dist_from_center': rms_dist_from_center
                    })

        return pd.DataFrame(results)


    def simulate_and_evaluate_causal_tree_honest(self, X_sim:np.ndarray, Y_sim:np.ndarray, T_sim:np.ndarray, treatment_effect:np.ndarray, random_state: int) -> pd.DataFrame:
        X_struct, X_est, y_struct, y_est, T_struct, T_est, treatment_effect_struct, treatment_effect_est = train_test_split(
            X_sim, Y_sim, T_sim, treatment_effect, test_size=0.5, random_state=random_state, stratify=T_sim
        )

        X_est_mean = X_est.mean().item()
        T_est_mean = T_est.mean().item()

        results = []

        for depth in self.max_depth:
            causal_tree = CausalTreeRegressor(
                max_depth=depth,
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_group_samples=self.min_group_samples,
                control_name=self.control_name,     
                random_state=random_state,
                groups_penalty=self.groups_penalty,
                min_treated_leaf=self.min_treated_leaf,
                min_control_leaf=self.min_control_leaf,
            )

            causal_tree.fit(X=X_struct, treatment=T_struct, y=y_struct)

            te_pred = causal_tree.predict(X_est)

            te_pred_round = np.round(te_pred, decimals=6)

            unique_preds = np.unique(te_pred_round)
            cate_to_leaf_id = {val: idx for idx, val in enumerate(unique_preds)}

            leaf_ids = np.array([cate_to_leaf_id.get(val, -1) for val in te_pred_round])

            unique_leaves = np.unique(leaf_ids)

            for leaf_id in unique_leaves:
                    mask = (leaf_ids == leaf_id)

                    x_leaf = X_est[mask]
                    y_leaf = y_est[mask]
                    t_leaf = T_est[mask]
                    te_leaf = treatment_effect_est[mask]

                    n_t = np.sum(t_leaf == 1)
                    n_c = np.sum(t_leaf == 0)

                    true_mean_te = te_leaf.mean()

                    abs_dist_from_center = np.mean(np.abs(x_leaf - 50))
                    rms_dist_from_center = np.sqrt(np.mean(np.square(x_leaf - 50)))

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

                    elif n_t >= 1 and n_c >= 1:
                        mu_t = y_leaf[t_leaf == 1].mean()
                        mu_c = y_leaf[t_leaf == 0].mean()
                        
                        cate_est = mu_t - mu_c

                        se_cate = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan
                        coverage = np.nan

                    else:
                        cate_est = np.nan
                        se_cate = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan
                        coverage = np.nan
                        
                    results.append({
                        'leaf_id': leaf_id,
                        'depth': depth,
                        'CATE_diff_in_mean': cate_est,
                        "CATE_pred": te_pred[mask].mean(),
                        'true_te': treatment_effect.mean(),
                        'abs_cate_deviation': np.abs(cate_est - treatment_effect.mean()),
                        'standard_error': se_cate,
                        'standard_error_adj': 1.96 * se_cate,
                        'n_samples': len(y_leaf),
                        'n_treated': n_t,
                        'n_control': n_c,
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'true_treatment_effect': true_mean_te,
                        'coverage': coverage,
                        'x_leaf_mean': x_leaf.mean().item(),
                        'x_leaf_dist_max_min': x_leaf.max() - x_leaf.min(),
                        'x_est_mean': X_est_mean,
                        't_est_mean': T_est_mean,
                        'abs_dist_from_center': abs_dist_from_center,
                        'rms_dist_from_center': rms_dist_from_center
                    })

        return pd.DataFrame(results)

