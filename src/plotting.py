"""
Plotting functions for simulation results.

This module contains the following functions:
    - plot_coverage_adaptive
"""

import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import SAMPLE_SIZE

pd.options.plotting.backend = "plotly"

PLOT_DIR = Path(r"C:\Users\flori\OneDrive\master_thesis\plots")

splitting_types = ["adaptive","honest"]
splitting_types_refined = ["adaptive", "honest_nan", "honest_0"]
target_depths = list(range(1,5)) + [6] + [8] + [10] + [15] + [20] + [35]


def plot_coverage(sim_results: pd.DataFrame, x_type: str):

    for splitting_type in splitting_types_refined:
        parts = splitting_type.split('_')
        base_type = parts[0]
        end_type = parts[1] if len(parts) > 1 else ""

        if end_type in ["", "nan"]:
            results_df = sim_results[sim_results['splitting_type'] == base_type]

        elif end_type == "0":
            results_df = sim_results[sim_results['splitting_type'] == base_type]
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
            subplot_titles=[f'Depth = {d}' for d in target_depths],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        for i, depth in enumerate(target_depths):
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

            cover_plot.add_hline(
                y=depth_means[depth],
                line_color="green",
                line_dash="solid",
                row=row, col=col,
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

            if x_type == "whole_support":
                x_text = "leaf mean"
            elif x_type == "abs_deviation":
                x_text = "extreme absolute deviations"

            cover_plot.update_xaxes(title_text=x_text, row=row, col=col)

        if splitting_type == "adaptive":
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Coverage for <i>n</i> = {SAMPLE_SIZE} and adaptive estimation by {x_text} (bin width = {bin_width})"
            )

        elif splitting_type == "honest_nan":
            # Adapt y-axes based on x_type
            if x_type == "whole_support":
                cover_plot.update_yaxes(
                    tickvals=np.linspace(0.8, 1, 5),
                    range=[0.8, 1]
                )
            elif x_type == "abs_deviation":
                if SAMPLE_SIZE == 5000:
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
                    title_text=f"Coverage for <i>n</i> = {SAMPLE_SIZE} and honest estimation by {x_text}<br>with coverage of NaN leaves as NaN (bin width = {bin_width})"
                )

            elif x_type == "abs_deviation":
                cover_plot.update_layout(
                    height=1500,
                    width=1000,
                    title_text=f"Coverage for <i>n</i> = {SAMPLE_SIZE} and honest estimation by {x_text}<br>with coverage of NaN leaves as NaN (bin width = {bin_width})"
                )

        elif splitting_type == "honest_0":
            cover_plot.update_yaxes(
                tickvals=np.linspace(0.2, 1, 9),
                range=[0.2, 1]
            )

            if x_type == "whole_support":
                cover_plot.update_layout(
                    height=1500,
                    width=1000,
                    title_text=f"Coverage for <i>n</i> = {SAMPLE_SIZE} and honest estimation by {x_text}<br>with coverage of NaN leaves as 0 (bin width = {bin_width})"
                )

            elif x_type == "abs_deviation":
                cover_plot.update_layout(
                    height=1500,
                    width=1000,
                    title_text=f"Coverage for <i>n</i> = {SAMPLE_SIZE} and honest estimation by {x_text}<br>with coverage of NaN leaves as 0 (bin width = {bin_width})"
                )


        file_path = PLOT_DIR / f"plot_support_{SAMPLE_SIZE}_sim_25000_1_10_{splitting_type}_{x_type}.html"
        cover_plot.write_html(file_path)
        cover_plot.show()


def plot_adj_se_treatment_dev(sim_results:pd.DataFrame, x_type: str):

    for splitting_type in splitting_types:

        results_df = sim_results[sim_results['splitting_type'] == splitting_type]

        if x_type == "whole_support":
            bins = 200
            results_df["bin_interval"] = pd.cut(results_df['x_leaf_mean'], bins=bins)
            bin_width = 100 / bins

        elif x_type == "abs_deviation":
            
            if splitting_type == "adaptive":
                lower_bound = 30
                bins = 100
                results_df = results_df[results_df['abs_dist_from_center']>=lower_bound].copy()
                results_df["bin_interval"] = pd.cut(results_df["abs_dist_from_center"], bins=bins)
            
            elif splitting_type == "honest":
                lower_bound = 49
                bins = 50
                results_df = results_df[results_df['abs_dist_from_center']>=lower_bound].copy()
                results_df["bin_interval"] = pd.cut(results_df["abs_dist_from_center"], bins=bins)
            
            bin_width = (50 - lower_bound) / bins

        results_df["bin_center"] = results_df["bin_interval"].apply(lambda x: x.mid).astype(float)

        plot_df = results_df.groupby(["depth", "bin_center"], observed=True)[["abs_cate_deviation", "standard_error_adj"]].mean().reset_index()

        plot_df.loc[plot_df['standard_error_adj'].isna(), 'abs_cate_deviation'] = np.nan

        cover_plot = make_subplots(
            rows=5, cols=2, 
            subplot_titles=[f'Depth = {d}' for d in target_depths],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        for i, depth in enumerate(target_depths):
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

            if x_type == "whole_support":

                if splitting_type == "adaptive":
                    try:
                        if 5 <= depth <= 26:
                            cover_plot.update_yaxes(
                                tickvals = np.linspace(75,150,4),
                                range=[70, 150],
                                row=row, col=col
                            )
                    except (ValueError, IndexError):
                        pass

                    else:
                        cover_plot.update_yaxes(
                            tickvals=np.linspace(0, 175, 8),
                            range=[0, 180],
                            row=row, col=col
                        )

                elif splitting_type == "honest":
                    cover_plot.update_yaxes(
                        tickvals=np.linspace(0, 150, 7),
                        range=[0, 160],
                        row=row, col=col
                    )

            if x_type == "whole_support":
                x_text = "leaf mean"
            elif x_type == "abs_deviation":
                x_text = "extreme abs. deviations"
            
            cover_plot.update_xaxes(title_text=x_text, row=row, col=col)

        # Generate the title string dynamically
        title = f"Absolute CATE estimate deviation vs. 1.96 * standard errror<br>for {splitting_type} estimation by {x_text} (bin width = {bin_width})"

        # Apply the layout update once for all conditions
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

        for size in SAMPLE_SIZES:

            file_path = PLOT_DIR / f"adj_se_cate_dev_plot_support_{size}_sim_25000_1_10_{splitting_type}_{x_type}.html"
            cover_plot.write_html(file_path)
            cover_plot.show()




def plot_leaf_frequency(sim_results:pd.DataFrame):

    for splitting_type in splitting_types_refined:

        parts = splitting_type.split('_')
        base_type = parts[0]
        end_type = parts[1] if len(parts) > 1 else ""
        

        results_df_plot = sim_results[sim_results['splitting_type'] == base_type].copy()

        bins = 200

        bin_width = 100 / bins

        results_df_plot["bin_interval"] = pd.cut(results_df_plot['x_leaf_mean'], bins=bins, include_lowest=True)

        results_df_plot["bin_center"] = results_df_plot["bin_interval"].apply(lambda x: x.mid)

        leaf_counts = results_df_plot.groupby(["depth", "bin_center"], observed=True).size().reset_index(name='leaf_count')

        if end_type in ["", "0"]:
            avg_splits_per_depth = results_df_plot.groupby(["depth"], observed=True).size().reset_index(name='leaves_count')

        elif end_type == "nan":
            avg_splits_per_depth = results_df_plot[~results_df_plot['coverage'].isna()].groupby(["depth"], observed=True).size().reset_index(name='leaves_count')

        leaf_counts['total_leaves_per_depth'] = leaf_counts['depth'].map(avg_splits_per_depth.set_index('depth')['leaves_count'])

        leaf_counts['leaf_count_share'] = leaf_counts['leaf_count'] / leaf_counts['total_leaves_per_depth']

        cover_plot = make_subplots(
            rows=5, cols=2, 
            subplot_titles=[f'Depth = {d}' for d in target_depths],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )

        for i, depth in enumerate(target_depths):
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

            if end_type == "":
                try:
                    if 7 <= depth <= 25:
                        cover_plot.update_yaxes(
                            tickvals = np.linspace(0,0.03,4),
                            range=[0,0.03],
                            row=row, col=col
                        )
                except (ValueError, IndexError):
                    pass

                else:
                    cover_plot.update_yaxes(
                        tickvals=np.linspace(0,0.05,6),
                        range=[0,0.05],
                        row=row, col=col
                    )

            elif end_type == "nan":
                try:
                    if 7 <= depth <= 25:
                        cover_plot.update_yaxes(
                            tickvals = np.linspace(0,0.03,4),
                            range=[0,0.03],
                            row=row, col=col
                        )
                except(ValueError, IndexError):
                    pass

                else:
                    cover_plot.update_yaxes(
                        tickvals=np.linspace(0,0.05,6),
                        range=[0,0.05],
                        row=row, col=col
                    )

            elif end_type == "0": 
                try:
                    if 5 <= depth <= 26:
                        cover_plot.update_yaxes(
                            tickvals = np.linspace(0,0.03,4),
                            range=[0,0.03],
                            row=row, col=col
                        )
                except(ValueError, IndexError):
                    pass
                else:
                    cover_plot.update_yaxes(
                        tickvals=np.linspace(0,0.04,5),
                        range=[0,0.04],
                        row=row, col=col
                    )
            cover_plot.update_xaxes(title_text="leaf mean", row=row, col=col)
                
        # cover_plot.update_yaxes(
        #         tickvals=np.linspace(0,1,11), 
        #         range=[0, 1],
        # )
                        
        if end_type == "":           
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of leaf means for adaptive estimation (bin width={bin_width})"
            )

        elif end_type == "nan":           
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of leaf means for honest estimation<br>with nan coverages not included (bin width={bin_width})"
            )

        elif end_type == "0":           
            cover_plot.update_layout(
                height=1500,
                width=1000,
                title_text=f"Relative frequency distribution of leaf means for honest estimation<br>with nan coverages included (bin width={bin_width})"
            )

        for size in SAMPLE_SIZES:

            file_path = PLOT_DIR / f"plot_{size}_leaf_relative_freqency_{splitting_type}.html"
            cover_plot.write_html(file_path)
            cover_plot.show()

