"""
Plotting functions for simulation results.

This module contains the following functions:
    - plot_coverage
    - plot_adj_se_treatment_dev
    - plot_leaf_frequency
    - plot_n_samples_frequency
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .config import TREE_DEPTHS, SPLITTING_TYPES, PLOTS_DIR

splitting_types_refined = ["adaptive", "honest_nan", "honest_0"]  


def plot_coverage(sim_results: pd.DataFrame, sample_size: int, x_type: str):

    tree_depth = TREE_DEPTHS[sample_size]

    for splitting_type in splitting_types_refined:
        parts = splitting_type.split('_')
        base_type = parts[0]
        end_type = parts[1] if len(parts) > 1 else ""

        if end_type in ["", "nan"]:
            results_df = sim_results[sim_results['splitting_type'] == base_type].copy()

        elif end_type == "0":
            results_df = sim_results[sim_results['splitting_type'] == base_type].copy()
            results_df['coverage'] = results_df['coverage'].fillna(0)   

        if x_type == "whole_support":
            bins = 200
            results_df["bin_interval"] = pd.cut(results_df['x_leaf_mean'], bins=bins)
            bin_width = 100 / bins

        elif x_type == "abs_deviation":
            
            if end_type == "":
                lower_bound = 30
                bins = 100
                results_df = results_df[results_df['abs_dist_from_center']>=lower_bound].copy()
                results_df["bin_interval"] = pd.cut(results_df["abs_dist_from_center"], bins=bins)
            
            elif end_type == "nan":
                lower_bound = 49
                bins = 50
                results_df = results_df[results_df['abs_dist_from_center']>=lower_bound].copy()
                results_df["bin_interval"] = pd.cut(results_df["abs_dist_from_center"], bins=bins)

            elif end_type == "0":
                lower_bound = 40
                bins = 100
                results_df = results_df[results_df['abs_dist_from_center']>=lower_bound].copy()
                results_df["bin_interval"] = pd.cut(results_df["abs_dist_from_center"], bins=bins)
            
            bin_width = (50 - lower_bound) / bins
        
        depth_means = results_df.groupby("depth")["coverage"].mean()

        results_df["bin_center"] = results_df["bin_interval"].apply(lambda x: x.mid).astype(float)

        plot_df = results_df.groupby(["depth", "bin_center"], observed=True)["coverage"].mean().reset_index()

        cover_plot = make_subplots(
            rows=5, cols=2, 
            subplot_titles=[f'Depth = {d}' for d in tree_depth],
            vertical_spacing=0.075,
            horizontal_spacing=0.05,
            shared_yaxes="rows",
        )

        for i, depth in enumerate(tree_depth):
            depth_data = plot_df[plot_df['depth'] == depth]

            row = (i // 2) + 1
            col = (i % 2) + 1

            cover_plot.add_trace(
                go.Scatter(
                    x=depth_data['bin_center'], 
                    y=depth_data['coverage'],
                    mode='lines',
                    showlegend=False 
                ),
                row=row, col=col
            )

            x_min = depth_data['bin_center'].min()
            x_max = depth_data['bin_center'].max()

            cover_plot.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[depth_means[depth], depth_means[depth]],
                    mode='lines',
                    name='mean coverage',
                    line=dict(color='green', dash='solid'),
                    legendgroup='mean_line',
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )

            cover_plot.add_annotation(
                text = f"mean coverage: {depth_means[depth]:.3f}",
                xref = "x domain",
                yref = "y domain",
                x = 0.0,
                y = 0.0,
                xanchor = "left",
                yanchor = "bottom",
                showarrow = False,
                font=dict(size=14, color="black"),
                row=row, col=col
            )

            if base_type == "adaptive":
                if depth in range(1, 5):
                    cover_plot.update_yaxes(
                        tickvals=np.linspace(0, 1, 11),
                        range=[0, 1],
                        row=row, col=col
                    )
                elif depth in range(5, 36):
                    try:
                        cover_plot.update_yaxes(
                            tickvals=np.linspace(0.4, 0.8, 5),
                            range=[0.4, 0.8],
                            row=row, col=col
                        )
                    except (ValueError, IndexError):
                        pass

        cover_plot.update_xaxes(title_text="number of samples per leaf", row=5)

        cover_plot.update_layout(
            legend=dict(
                    orientation="v",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                    )
        )

        if x_type == "whole_support":
            x_text = "leaf mean"
        elif x_type == "abs_deviation":
            x_text = "extreme absolute deviations"


        if splitting_type == "adaptive":
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Coverage for <i>n</i> = {sample_size} and adaptive estimation by {x_text} (bin width = {bin_width})",
        )
        
        elif splitting_type == "honest_nan":
            # Adapt y-axes based on x_type
            if x_type == "whole_support":
                cover_plot.update_yaxes(
                    tickvals=np.linspace(0.8, 1, 5),
                    range=[0.8, 1]
                )
            elif x_type == "abs_deviation":
                if sample_size == 5000:
                    cover_plot.update_yaxes(
                        tickvals=np.linspace(0.3, 1, 8),
                        range=[0.3, 1]
                    )
                    
                else:
                    cover_plot.update_yaxes(
                        tickvals=np.linspace(0.3, 1, 8),
                        range=[0.3, 1]
                    )

            if x_type == "whole_support":
                cover_plot.update_layout(
                    height=1500,
                    width=1000,
                    title_text=f"Coverage for <i>n</i> = {sample_size} and honest estimation by {x_text}<br>with coverage of NaN leaves as NaN (bin width = {bin_width})"
                )

            elif x_type == "abs_deviation":
                cover_plot.update_layout(
                    height=1500,
                    width=1000,
                    title_text=f"Coverage for <i>n</i> = {sample_size} and honest estimation by {x_text}<br>with coverage of NaN leaves as NaN (bin width = {bin_width})"
                )

        elif splitting_type == "honest_0":
            cover_plot.update_yaxes(
                tickvals=np.linspace(0, 1, 11),
                range=[0, 1]
            )

            if x_type == "whole_support":
                cover_plot.update_layout(
                    height=1500,
                    width=1000,
                    title_text=f"Coverage for <i>n</i> = {sample_size} and honest estimation by {x_text}<br>with coverage of NaN leaves as 0 (bin width = {bin_width})"
                )

            elif x_type == "abs_deviation":
                cover_plot.update_layout(
                    height=1500,
                    width=1000,
                    title_text=f"Coverage for <i>n</i> = {sample_size} and honest estimation by {x_text}<br>with coverage of NaN leaves as 0 (bin width = {bin_width})"
                )

        file_path = PLOTS_DIR / f"plot_support_{sample_size}_sim_25000_1_10_{splitting_type}_{x_type}.html"
        cover_plot.write_html(file_path)

        cover_plot.show()


def plot_adj_se_treatment_dev(sim_results:pd.DataFrame, sample_size: int, x_type: str):

    tree_depth = TREE_DEPTHS[sample_size]

    for splitting_type in SPLITTING_TYPES:

        results_df = sim_results[sim_results['splitting_type'] == splitting_type].copy()

        if x_type == "whole_support":
            bins = 200
            results_df["bin_interval"] = pd.cut(results_df['x_leaf_mean'], bins=bins)
            bin_width = 100 / bins

            depth_mad = results_df.groupby("depth")["abs_cate_deviation"].mean()

        elif x_type == "abs_deviation":
            
            if splitting_type == "adaptive":
                lower_bound = 30
                bins = 100
            
            elif splitting_type == "honest":
                lower_bound = 49
                bins = 50

            results_df = results_df[results_df['abs_dist_from_center']>=lower_bound].copy()
            results_df["bin_interval"] = pd.cut(results_df["abs_dist_from_center"], bins=bins)

            depth_mad = results_df.groupby("depth")["abs_cate_deviation"].mean()
            
            bin_width = (50 - lower_bound) / bins

        results_df["bin_center"] = results_df["bin_interval"].apply(lambda x: x.mid).astype(float)

        plot_df = results_df.groupby(["depth", "bin_center"], observed=True)[["abs_cate_deviation", "standard_error_adj"]].mean().reset_index()

        plot_df.loc[plot_df['standard_error_adj'].isna(), 'abs_cate_deviation'] = np.nan

        cover_plot = make_subplots(
            rows=5, cols=2, 
            subplot_titles=[f'Depth = {d}' for d in tree_depth],
            vertical_spacing=0.075,
            horizontal_spacing=0.05,
            shared_yaxes="rows",
        )

        for i, depth in enumerate(tree_depth):
            depth_data = plot_df[plot_df['depth'] == depth]

            row = (i // 2) + 1
            col = (i % 2) + 1

            show_legend = True if i == 0 else False

            cover_plot.add_trace(
                go.Scatter(
                    x=depth_data['bin_center'], 
                    y=depth_data['abs_cate_deviation'], 
                    mode='lines',
                    name="CATE estimate deviation<br>from true treatment effect",
                    legendgroup='deviation',
                    showlegend=show_legend,
                    line=dict(color='#1f77b4')
                ),
                row=row, col=col
            )
        
            cover_plot.add_trace(
                go.Scatter(
                    x=depth_data['bin_center'], 
                    y=depth_data['standard_error_adj'], 
                    mode='lines',
                    name="1.96 * standard error ",
                    legendgroup = "adj. standard error",
                    showlegend=show_legend,
                    line=dict(color='#ff7f0e')
                ),
                row=row, col=col
            )

            cover_plot.add_annotation(
                text = f"mean absolute deviation: {depth_mad[depth]:.3f}",
                xref = "x domain",
                yref = "y domain",
                x = 0.0,
                y = 1.0,
                xanchor = "left",
                yanchor = "top",
                showarrow = False,
                font=dict(size=14, color="black"),
                row=row, col=col
            )
        
        if x_type == "whole_support":
            x_text = "leaf mean"
        elif x_type == "abs_deviation":
            x_text = "extreme abs. deviations"
            
        cover_plot.update_xaxes(title_text=x_text, row=5)

        title = f"Absolute CATE estimate deviation vs. 1.96 * standard errror and<br>{splitting_type} estimation by {x_text} (bin width = {bin_width}, <i>n</i> = {sample_size})"

        cover_plot.update_layout(
            height=1500,
            width=1000,
            title_text=title,
            legend=dict(
                orientation="v",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        file_path = PLOTS_DIR / f"adj_se_cate_dev_plot_support_{sample_size}_sim_25000_1_10_{splitting_type}_{x_type}.html"
        cover_plot.write_html(file_path)

        cover_plot.show()




def plot_leaf_frequency(sim_results:pd.DataFrame, sample_size: int):

    tree_depth = TREE_DEPTHS[sample_size]

    for splitting_type in splitting_types_refined:

        parts = splitting_type.split('_')
        base_type = parts[0]
        end_type = parts[1] if len(parts) > 1 else ""
        
        results_df_plot = sim_results[sim_results['splitting_type'] == base_type].copy()

        bins = 200

        bin_width = 100 / bins

        results_df_plot["bin_interval"] = pd.cut(results_df_plot['x_leaf_mean'], bins=bins, include_lowest=True)

        results_df_plot["bin_center"] = results_df_plot["bin_interval"].apply(lambda x: x.mid)

        if end_type in ["", "0"]:
            filtered_df = results_df_plot
        elif end_type == "nan":
            filtered_df = results_df_plot[~results_df_plot['coverage'].isna()]

        leaf_counts = filtered_df.groupby(["depth", "bin_center"], observed=True).size().reset_index(name='leaf_count')

        avg_splits_per_depth = filtered_df.groupby(["depth"], observed=True).size().reset_index(name='leaves_count')

        leaf_counts['total_leaves_per_depth'] = leaf_counts['depth'].map(avg_splits_per_depth.set_index('depth')['leaves_count'])
        leaf_counts['leaf_count_share'] = leaf_counts['leaf_count'] / leaf_counts['total_leaves_per_depth']

        cover_plot = make_subplots(
            rows=5, cols=2, 
            subplot_titles=[f'Depth = {d}' for d in tree_depth],
            vertical_spacing=0.075,
            horizontal_spacing=0.05,
            shared_yaxes="rows",
        )

        for i, depth in enumerate(tree_depth):
            depth_data = leaf_counts[leaf_counts['depth'] == depth]

            row = (i // 2) + 1
            col = (i % 2) + 1

            cover_plot.add_trace(
                go.Scatter(
                    x=depth_data['bin_center'], 
                    y=depth_data['leaf_count_share'], 
                    mode='lines',
                    name="relative frequency of leaves",
                    line=dict(color='#1f77b4'),
                    showlegend=False
                ),
                row=row, col=col
            )

        cover_plot.update_xaxes(title_text="leaf mean", row=5)
           
        if end_type == "":           
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of leaf means for <i>n</i> = {sample_size} and adaptive estimation (bin width={bin_width})"
            )

        elif end_type == "nan":           
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of leaf means for <i>n</i> = {sample_size}<br>and honest estimation with nan coverages not included (bin width={bin_width})"
            )

        elif end_type == "0":           
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of leaf means for <i>n</i> = {sample_size}<br>and honest estimation with nan coverages included (bin width={bin_width})"
            )

            html_path = str(PLOTS_DIR / f"plot_{sample_size}_leaf_relative_freqency_{splitting_type}.html")

            cover_plot.write_html(html_path)

            cover_plot.show()




def plot_n_samples_frequency(sim_results:pd.DataFrame, sample_size: int):

    tree_depth = TREE_DEPTHS[sample_size]

    for splitting_type in splitting_types_refined:

        parts = splitting_type.split('_')
        base_type = parts[0]
        end_type = parts[1] if len(parts) > 1 else ""

        bin_width = 0.5
        
        results_df_plot = sim_results[sim_results['splitting_type'] == base_type].copy()

        cover_plot = make_subplots(
            rows=5, cols=2, 
            subplot_titles=[f'Depth = {d}' for d in tree_depth],
            vertical_spacing=0.075,
            horizontal_spacing=0.05,
            shared_yaxes="rows",
        )

        for i, depth in enumerate(tree_depth):

            results_df_plot_depth = results_df_plot[results_df_plot['depth'] == depth].copy()

            if end_type in ["", "0"]:
                filtered_depth_df = results_df_plot_depth.copy()
            elif end_type == "nan":
                filtered_depth_df = results_df_plot_depth[~results_df_plot_depth['coverage'].isna()].copy()
            else:
                filtered_depth_df = results_df_plot_depth.copy() # Fallback

            min_val = filtered_depth_df['n_samples'].min()
            max_val = filtered_depth_df['n_samples'].max()

            print(f"Depth {depth}: min samples = {min_val}, max samples = {max_val}")

            bin_edges = np.arange(min_val, max_val + bin_width, bin_width)

            filtered_depth_df["bin_interval"] = pd.cut(filtered_depth_df['n_samples'], bins=bin_edges, include_lowest=True)

            filtered_depth_df["bin_center"] = filtered_depth_df["bin_interval"].apply(lambda x: x.mid)

            samples_counts = filtered_depth_df.groupby(["depth", "bin_center"], observed=True).size().reset_index(name='samples_count')

            avg_splits_per_depth = filtered_depth_df.groupby(["depth"], observed=True).size().reset_index(name='total_samples')

            samples_counts['total_samples_per_depth'] = samples_counts['depth'].map(avg_splits_per_depth.set_index('depth')['total_samples'])

            samples_counts['sample_count_share'] = samples_counts['samples_count'] / samples_counts['total_samples_per_depth']

            depth_data = samples_counts[samples_counts['depth'] == depth]

            row = (i // 2) + 1
            col = (i % 2) + 1

            cover_plot.add_trace(
                go.Scatter(
                    x=depth_data['bin_center'], 
                    y=depth_data['sample_count_share'], 
                    mode='lines',
                    name="sample_count_share",
                    line=dict(color='#1f77b4'),
                    showlegend=False
                ),
                row=row, col=col
            )

            cover_plot.update_xaxes(range=[min_val, max_val],
                                    row=row, 
                                    col=col)
            
        cover_plot.update_xaxes(title_text="number of samples per leaf", row=5)

        if end_type == "":          
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of the number of samples per leaf for <i>n</i> = {sample_size}<br>and adaptive estimation (bin width={bin_width})"
            )

        elif end_type == "nan":          
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of the number of samples per leaf for <i>n</i> = {sample_size}<br>and honest estimation with nan coverages not included (bin width={bin_width})"
            )

        elif end_type == "0":          
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of the number of samples per leaf for <i>n</i> = {sample_size}<br>and honest estimation with nan coverages included (bin width={bin_width})"
            )

        html_path = str(PLOTS_DIR / f"plot_{sample_size}_n_samples_distribution_{splitting_type}.html")

        cover_plot.write_html(html_path)

        cover_plot.show()

