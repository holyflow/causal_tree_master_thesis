import numpy as np
import pandas as pd
from pathlib import Path
from .config import TREE_DEPTHS, TABLES_DIR


def coverage_table(results_df: pd.DataFrame, sample_size: int):
    """Generates a table of weighted average coverage by depth and splitting type"""
    
    results_df = results_df.copy()
    results_df['coverage'] = results_df['coverage'].fillna(0)

    def calculate_weighted_coverage(group):
        return (group['coverage'] * group['n_samples']).sum() / group['n_samples'].sum()

    sim_coverage = results_df.groupby(
        ['splitting_type', 'depth', 'simulation']
    ).apply(calculate_weighted_coverage).reset_index(name='coverage_weighted')

    final_coverage = sim_coverage.groupby(
        ['depth', 'splitting_type'], dropna=False
    )['coverage_weighted'].mean().unstack('splitting_type')

    final_coverage.index.names = ["Depth"]
    final_coverage.columns.name = None
    final_coverage = final_coverage.rename(columns={"adaptive": "Adaptive", "honest": "Honest"})

    final_coverage.style.format(precision=3, na_rep="--").to_latex(
        rf"{TABLES_DIR}\{sample_size}_coverage_table.tex",
        position="htbp",
        position_float="centering",
        caption="Weighted average coverage by depth and splitting type",
        label="tab:coverage",
        multirow_align="t",
        hrules=True,
        clines="skip-last;data"
    )


def leaf_size_table(sim_results: pd.DataFrame, sample_size: int):
    """Generates a table of mean leaf sizes by depth and splitting type"""
    
    tree_depth = TREE_DEPTHS[sample_size]
    
    bin_edges = np.linspace(47, 50, 7)

    sim_results_extreme = sim_results[sim_results['depth'].isin(tree_depth)]

    sim_results_extreme = sim_results_extreme[sim_results_extreme['abs_dist_from_center'] >= 47].copy() # Added .copy() to prevent SettingWithCopyWarning

    sim_results_extreme["abs_dist_center_int"] = pd.cut(
        sim_results_extreme["abs_dist_from_center"], 
        bins=bin_edges,
        include_lowest=True
    )
    # 
    df_overall = sim_results[sim_results["depth"] <= max(tree_depth)].groupby(
        ["depth", "coverage", "splitting_type"], dropna=False
    )["n_samples"].mean().unstack("splitting_type")

    df_overall.index.names = ["Depth", "Coverage"]
    df_overall.columns.name = None
    df_overall = df_overall.rename(columns={"adaptive": "Adaptive", "honest": "Honest"})

    df_extreme = sim_results_extreme[sim_results_extreme["depth"] <= 10].groupby(
        ["abs_dist_center_int", "depth", "coverage", "splitting_type"], dropna=False,
    )["n_samples"].mean().unstack("splitting_type").round(2)

    df_extreme.index.names = ["Abs. Deviation Interval", "Depth", "Coverage"]
    df_extreme.columns.name = None
    df_extreme = df_extreme.rename(columns={"adaptive": "Adaptive", "honest": "Honest"})

    all_intervals = df_extreme.index.get_level_values("Abs. Deviation Interval").categories

    intervals_part1 = all_intervals[0:2]
    intervals_part2 = all_intervals[2:4]
    intervals_part3 = all_intervals[4:6]

    df_extreme_part1 = df_extreme[df_extreme.index.get_level_values("Abs. Deviation Interval").isin(intervals_part1)]
    df_extreme_part2 = df_extreme[df_extreme.index.get_level_values("Abs. Deviation Interval").isin(intervals_part2)]
    df_extreme_part3 = df_extreme[df_extreme.index.get_level_values("Abs. Deviation Interval").isin(intervals_part3)]

    df_overall.style.format(precision=2, na_rep="--").to_latex(
        rf"{TABLES_DIR}\{sample_size}_overall_table.tex",
        position="htbp",
        position_float="centering",
        caption="Mean leaf sizes overall by splitting type",
        label="tab:overall",
        multirow_align="t",
        hrules=True,
        clines="skip-last;data"
    )

    df_extreme_part1.style.format(precision=2, na_rep="--").to_latex(
        rf"{TABLES_DIR}\{sample_size}_extreme_table_part1.tex",
        position="htbp",
        position_float="centering",
        caption=r"Mean leaf sizes for extreme absolute deviations of leaf means from population mean \\ (Part 1: 47.0 to 48.0)", 
        label="tab:extreme_part1",
        multirow_align="t",
        hrules=True,
        clines="skip-last;data"
    )

    df_extreme_part2.style.format(precision=2, na_rep="--").to_latex(
        rf"{TABLES_DIR}\{sample_size}_extreme_table_part2.tex",
        position="htbp",
        position_float="centering",
        caption=r"Mean leaf sizes for extreme absolute deviations of leaf means from population mean \\ (Part 2: 48.0 to 49.0)",
        label="tab:extreme_part2",
        multirow_align="t",
        hrules=True,
        clines="skip-last;data"
    )

    df_extreme_part3.style.format(precision=2, na_rep="--").to_latex(
        rf"{TABLES_DIR}\{sample_size}_extreme_table_part3.tex",
        position="htbp",
        position_float="centering",
        caption=r"Mean leaf sizes for extreme absolute deviations of leaf means from population mean \\ (Part 3: 49.0 to 50.0)",
        label="tab:extreme_part3",
        multirow_align="t",
        hrules=True,
        clines="skip-last;data"
    )



def populated_leaves_table(results_df: pd.DataFrame, sample_size: int):
    """Generates a table calculating the proportion of populated leaves for honest trees by depth."""

    results_df_honest = results_df[results_df["splitting_type"].isin(["honest"])].copy()

    results_df_honest['leaf_populated'] = 0
    results_df_honest.loc[~results_df_honest['coverage'].isna(), 'leaf_populated'] = 1 

    results_df_table = results_df_honest.groupby(['depth'])['leaf_populated'].mean()
    results_df_table.index.names = ["Depth"]
    results_df_table.name = "Proportion of Populated Leaves"

    table_df = pd.DataFrame({
        "Depth": results_df_table.index,
        "Proportion of Populated Leaves": results_df_table.values
    })

    table_df = table_df.set_index("Depth")

    table_df.style.format(precision=4, na_rep="--").to_latex(
        rf"{TABLES_DIR}\{sample_size}_populated_leaves.tex",
        position="htbp",
        position_float="centering",
        caption=rf"Percentage of Populated Leaves for $n = {sample_size}$",
        label="tab:populated_leaves",
        multirow_align="t",
        hrules=True,
        clines="skip-last;data"
    )