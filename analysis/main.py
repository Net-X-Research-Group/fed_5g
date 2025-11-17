import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats
import itertools
from data_loading import *
from helpers import *
import pingouin as pg

# Configure matplotlib for LaTeX fonts
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}'
})

column_name_map = {
    'train_loss': 'Training Loss',
    'train_time': 'Training Time (s)',
    'eval_loss': 'Evaluation Loss',
    'eval_acc': 'Evaluation Accuracy',
    'eval_time': 'Evaluation Time (s)',
    'round_duration': 'Round Duration (s)',
}

TIME_METRICS = ['train_time', 'eval_time', 'round_duration']

# Global list to track all filtering operations
FILTER_LOG = []

def record_filtering(metric_name, filter_description, original_count, filtered_count, additional_info="",
                    filter_min=None, filter_max=None, median_replacement=None):
    """Record filtering operations for later CSV export - percentages calculated at the end"""
    removed_count = original_count - filtered_count

    # Extract sweep_param and experiment from additional_info
    sweep_param = ""
    experiment = ""
    if "sweep_param=" in additional_info:
        parts = additional_info.split(", ")
        for part in parts:
            if part.startswith("sweep_param="):
                sweep_param = part.replace("sweep_param=", "")
            elif part.startswith("experiment="):
                experiment = part.replace("experiment=", "")

    # Use "TOTAL" for null/empty experiment values
    if not experiment or experiment.strip() == "":
        experiment = "TOTAL"

    FILTER_LOG.append({
        'sweep_param': sweep_param,
        'experiment': experiment,
        'total_count': original_count,
        'filtered_count': filtered_count,
        'removed_count': removed_count,
        'metric_name': metric_name,
        'filter_threshold_min': filter_min,
        'filter_threshold_max': filter_max,
        'median_value_replacement': median_replacement,
        'filter_description': filter_description
    })

def save_filter_log(output_dir):
    """Save the filter log to a CSV file with final percentage calculations"""
    if FILTER_LOG:
        # Calculate percentages only when saving
        for entry in FILTER_LOG:
            if entry['total_count'] > 0:
                entry['percent_removed'] = (entry['removed_count'] / entry['total_count'] * 100)
            else:
                entry['percent_removed'] = 0

        filter_df = pd.DataFrame(FILTER_LOG)
        # Reorder columns as requested
        column_order = ['sweep_param', 'experiment', 'percent_removed', 'total_count',
                       'filtered_count', 'removed_count', 'metric_name',
                       'filter_threshold_min', 'filter_threshold_max', 'median_value_replacement']
        filter_df = filter_df[column_order]

        csv_path = output_dir / 'filtering_operations_log.csv'
        filter_df.to_csv(csv_path, index=False)
        print(f"Saved filtering operations log to {csv_path}")
        return csv_path
    return None



def plot_agg_metric(agg_metrics, output_dir, filter_str, sweep_param):
    """Create one figure per metric, with a line for each experiment"""
    sns.set_style("whitegrid")
    agg_metrics = sort_experiments_by_sweep(agg_metrics, sweep_param)

    metric_cols = [col for col in agg_metrics[0]['metrics'].columns
                   if col not in ['server_round', 'timestamp']]

    metric_cols = ['eval_time', 'train_time', 'eval_acc', 'eval_loss', 'train_loss', 'round_duration']


    figs = {}
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        display_name = column_name_map.get(metric, metric.replace('_', ' ').title())

        if metric in TIME_METRICS:
            # Frequency plot for time metrics
            for exp in agg_metrics:
                if 'round_duration' not in exp['metrics'].columns:
                    df = exp['metrics'].copy()
                    if 'round_duration' not in df.columns:
                        df['round_duration'] = df['timestamp'].diff()

                    # Replace outliers with median for time calculation
                    original_count = len(df['round_duration'].dropna())
                    outliers_mask = (df['round_duration'] > 200) | (pd.isna(df['round_duration']))
                    outliers_count = outliers_mask.sum()

                    median_duration = df['round_duration'].median()
                    df['round_duration_cleaned'] = df['round_duration'].apply(
                        lambda x: median_duration if (pd.isna(x) or x > 200) else x
                    )

                    # Record filtering operation
                    if outliers_count.any():
                        record_filtering(
                            'round_duration_time_calc',
                            'Replace outliers >200s or NaN with median',
                            original_count,
                            original_count - outliers_count.sum(),
                            f"sweep_param={sweep_param}, experiment={format_sweep_label(sweep_param, exp[sweep_param])}",
                            filter_min=0,
                            filter_max=200,
                            median_replacement=median_duration
                        )
                    exp['metrics'] = df
                label = format_sweep_label(sweep_param, exp[sweep_param])
                sns.histplot(exp['metrics'][metric], label=label, ax=ax)
            ax.set_xlabel(display_name, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
        else:
            # Line plot for other metrics
            for exp in agg_metrics:
                df = exp['metrics']
                label = format_sweep_label(sweep_param, exp[sweep_param])
                sns.lineplot(x=df['server_round'], y=df[metric],
                             label=label, linewidth=2, ax=ax)
            ax.set_xlabel('Server Round', fontsize=12)
            ax.set_ylabel(display_name, fontsize=12)

        ax.legend()
        save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str)
        figs[metric] = fig

    return figs


def plot_agg_metric_vs_time(agg_metrics, output_dir, filter_str, sweep_param):
    sns.set_style("whitegrid")
    agg_metrics = sort_experiments_by_sweep(agg_metrics, sweep_param)

    metric_cols = [col for col in agg_metrics[0]['metrics'].columns
                   if col not in ['server_round', 'timestamp']]

    figs = {}
    for metric in metric_cols:
        if metric in TIME_METRICS:
            continue

        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_dist = fig.add_subplot(gs[1])

        display_name = column_name_map.get(metric, metric.replace('_', ' ').title())

        for exp in agg_metrics:
            df = exp['metrics'].copy()
            if 'round_duration' not in df.columns:
                df['round_duration'] = df['timestamp'].diff()

            # Replace outliers with median for time calculation
            original_count = len(df['round_duration'].dropna())
            outliers_mask = (df['round_duration'] > 200) | (pd.isna(df['round_duration']))
            outliers_count = outliers_mask.sum()

            median_duration = df['round_duration'].median()
            df['round_duration_cleaned'] = df['round_duration'].apply(
                lambda x: median_duration if (pd.isna(x) or x > 200) else x
            )

            # Record filtering operation
            if outliers_count.any():
                record_filtering(
                    'round_duration_time_calc',
                    'Replace outliers >200s or NaN with median',
                    original_count,
                    original_count - outliers_count.sum(),
                    f"sweep_param={sweep_param}, experiment={format_sweep_label(sweep_param, exp[sweep_param])}",
                    filter_min=0,
                    filter_max=200,
                    median_replacement=median_duration
                )

            # Compute elapsed_time as cumulative sum of cleaned durations
            df['elapsed_time'] = df['round_duration_cleaned'].cumsum()

            label = format_sweep_label(sweep_param, exp[sweep_param])
            sns.lineplot(x=df['elapsed_time'], y=df[metric],
                         label=label, linewidth=2, ax=ax_main)

        # Distribution subplot: histogram of round durations
        for exp in agg_metrics:
            df = exp['metrics'].copy()
            if 'round_duration' not in df.columns:
                df['round_duration'] = df['timestamp'].diff()

            # Filter based on impossible values
            original_count = len(df['round_duration'].dropna())
            outliers_mask = (df['round_duration'] > 200) & pd.notna(df['round_duration'])
            outliers_count = outliers_mask.sum()

            median_duration = df['round_duration'].median()
            df['round_duration'] = df['round_duration'].apply(
                lambda x: median_duration if (pd.notna(x) and x > 200) else x
            )

            # Record filtering operation
            if outliers_count.any():
                record_filtering(
                    'round_duration_distribution',
                    'Replace outliers >200s with median for distribution plot',
                    original_count,
                    original_count - outliers_count.sum(),
                    f"sweep_param={sweep_param}, experiment={format_sweep_label(sweep_param, exp[sweep_param])}",
                    filter_min=0,
                    filter_max=200,
                    median_replacement=median_duration
                )

            label = format_sweep_label(sweep_param, exp[sweep_param])
            sns.kdeplot(df['round_duration'].dropna(),
                        label=label, linewidth=2, ax=ax_dist)

        ax_main.set_xlabel('')
        ax_main.set_ylabel(display_name, fontsize=12)
        ax_main.legend(title='Bandwidth' if sweep_param == 'bandwidth' else sweep_param.title(), loc='best')
        ax_main.grid(True, alpha=0.3)

        ax_dist.set_xlabel('Round Duration (s)', fontsize=10)
        ax_dist.set_ylabel('Density', fontsize=10)
        ax_dist.tick_params(labelsize=9)
        ax_dist.grid(True, alpha=0.3, axis='y')

        save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str, suffix="_vs_time")
        figs[metric] = fig


    return figs

def plot_individual(metrics: list, output_dir: Path, filter_str: str, sweep_param: str):
    combined_df = prepare_individual_metrics(metrics, sweep_param)

    sweep_values = combined_df[sweep_param].unique()
    if sweep_param == 'bandwidth':
        sweep_values = sorted(sweep_values, key=lambda x: int(x.replace(' MHz', '')))
    elif sweep_param == 'tdd':
        sweep_values = sorted(sweep_values, key=lambda x: tuple(map(int, x.split('-'))))
    else:
        sweep_values = sorted(sweep_values)

    rd = combined_df['round_duration'].dropna()
    print('Round Duration Statistics:')
    print(f"Min: {rd.min():.2f}, Max: {rd.max():.2f}, Median: {rd.median():.2f}")
    print(f"Mean: {rd.mean():.2f}, Std: {rd.std():.2f}")
    print(f"Negative values: {(rd < 0).sum()}")
    print(f"99th percentile: {rd.quantile(0.99):.2f}")
    print(combined_df.groupby('cid')['round_duration'].describe())

    # Box plot for each CID for time metrics
    for metric in TIME_METRICS:
        fig, ax = plt.subplots(figsize=(12, 6))

        n_sweeps = len(sweep_values)
        cids = sorted(combined_df['cid'].unique())
        colors, markers = get_sweep_colors_markers(sweep_values)

        width = 0.8 / n_sweeps

        for i, sweep_val in enumerate(sweep_values):
            subset = combined_df[combined_df[sweep_param] == sweep_val]
            positions = np.arange(len(cids)) + (i - n_sweeps / 2 + 0.5) * width
            data_to_plot = [subset[subset['cid'] == cid][metric].dropna().values
                            for cid in cids]

            bp = ax.boxplot(
                data_to_plot,
                positions=positions,
                widths=width * 0.8,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(
                    marker=markers[i],
                    markersize=6,
                    markerfacecolor=colors[i],
                    markeredgecolor=colors[i],
                    alpha=0.6
                ),
                boxprops=dict(facecolor=colors[i], alpha=0.7),
                medianprops=dict(color='black', linewidth=1.5),
                whiskerprops=dict(color=colors[i]),
                capprops=dict(color=colors[i]),
            )

        lower = combined_df[metric].quantile(0.05)
        upper = combined_df[metric].quantile(0.95)
        margin = (upper - lower) * 0.1
        ax.set_ylim(lower - margin, upper + margin)

        ax.set_xticks(np.arange(len(cids)))
        ax.set_xticklabels(cids)
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel(column_name_map[metric], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker=markers[i], color='w',
                   markerfacecolor=colors[i], markersize=8,
                   label=format_sweep_label(sweep_param, val))
            for i, val in enumerate(sweep_values)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str, suffix="_by_client")

    # Round duration distribution
    for metric in ['round_duration']:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Filter for positive values and record the filtering
        original_count = len(combined_df[metric])
        plot_df = combined_df[combined_df[metric] > 0].copy()
        filtered_count = len(plot_df)

        if original_count != filtered_count:
            record_filtering(
                metric,
                'Filter for positive values only',
                original_count,
                filtered_count,
                f"sweep_param={sweep_param}",
                filter_min=0,
                filter_max=None
            )

        colors, _ = get_sweep_colors_markers(sweep_values)

        for i, sweep_val in enumerate(sweep_values):
            subset = plot_df[plot_df[sweep_param] == sweep_val][metric]
            label = format_sweep_label(sweep_param, sweep_val)

            sns.kdeplot(
                data=subset,
                ax=ax,
                label=label,
                color=colors[i],
                linewidth=2.5,
                fill=False,
                common_norm=False
            )

        ax.set_xlabel(column_name_map[metric], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Add vertical lines for medians
        for i, sweep_val in enumerate(sweep_values):
            subset = plot_df[plot_df[sweep_param] == sweep_val][metric]
            median_val = subset.median()
            ax.axvline(median_val, color=colors[i], linestyle='--',
                       linewidth=1.5, alpha=0.8)

        save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str, suffix="_distribution")

def plot_latency_metrics(latency_metrics: list, output_dir: Path, filter_str: str, sweep_param: str):
    """Plot latency metrics with CSV stats and figures for UL/DL latencies"""

    # Combine all latency data
    combined_latency = []
    for exp in latency_metrics:
        df = exp['metrics'].copy()
        df[sweep_param] = exp[sweep_param]
        combined_latency.append(df)

    if not combined_latency:
        print("No latency data found")
        return

    combined_df = pd.concat(combined_latency, ignore_index=True)

    def filter_outliers(data, column_name):
        original_count = len(data)

        # Filter data
        filtered_data = data[
            (data[column_name] >= 0) &
            (data[column_name] <= 50)
        ]

        outliers_removed = len(data) - len(filtered_data)
        if outliers_removed > 0:
            print(f"Filtered {outliers_removed} outliers from {column_name} "
                  f"(bounds: {0:.3f}-{50:.1f}s)")

        return filtered_data

    # Apply filtering to each metric separately
    print("Applying outlier filtering...")
    original_size = len(combined_df)

    # Filter each latency type separately to preserve data
    dl_filtered = filter_outliers(combined_df, 'downlink_latency')
    ul_filtered = filter_outliers(combined_df, 'uplink_latency')

    # For statistics, use the intersection of both filtered datasets
    combined_filtered = combined_df[
        combined_df.index.isin(dl_filtered.index) &
        combined_df.index.isin(ul_filtered.index)
    ]

    print(f"Original data points: {original_size}, After filtering: {len(combined_filtered)}")

    # Record the overall filtering operation after all filtering is complete
    if len(combined_filtered) != original_size:
        record_filtering(
            'latency_combined',
            'Combined downlink and uplink latency filtering (0-50s range, intersection)',
            original_size,
            len(combined_filtered),
            f"sweep_param={sweep_param}",
            filter_min=0,
            filter_max=50
        )

    combined_df = combined_filtered

    # Sort sweep values
    sweep_values = combined_df[sweep_param].unique()
    if sweep_param == 'bandwidth':
        sweep_values = sorted(sweep_values, key=lambda x: int(x.replace(' MHz', '')))
    elif sweep_param == 'tdd':
        sweep_values = sorted(sweep_values, key=lambda x: tuple(map(int, x.split('-'))))
    else:
        sweep_values = sorted(sweep_values)

    # Generate CSV with statistics
    stats_data = []
    for sweep_val in sweep_values:
        subset = combined_df[combined_df[sweep_param] == sweep_val]
        for metric in ['downlink_latency', 'uplink_latency']:
            if len(subset) > 0:
                stats = subset[metric].describe()
                stats_data.append({
                    sweep_param: sweep_val,
                    'metric': metric,
                    'count': stats['count'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    '25%': stats['25%'],
                    '50%': stats['50%'],
                    '75%': stats['75%'],
                    'max': stats['max']
                })

    stats_df = pd.DataFrame(stats_data)
    stats_csv_path = output_dir / 'latency_statistics_filtered.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved filtered latency statistics to {stats_csv_path}")

    colors, markers = get_sweep_colors_markers(sweep_values)

    # Figure 1: Individual figures for each sweep value showing UL/DL for each CID
    for i, sweep_val in enumerate(sweep_values):
        subset = combined_df[combined_df[sweep_param] == sweep_val]
        cids = sorted(subset['cid'].unique())

        if len(subset) == 0:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Downlink latency by CID
        dl_data = [subset[subset['cid'] == cid]['downlink_latency'].values for cid in cids]
        bp1 = ax1.boxplot(dl_data, tick_labels=cids, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
        for patch in bp1['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Downlink Latency (s)')
        #ax1.set_title(f'Downlink Latency - {format_sweep_label(sweep_param, sweep_val)}')
        ax1.grid(True, alpha=0.3)

        # Uplink latency by CID
        ul_data = [subset[subset['cid'] == cid]['uplink_latency'].values for cid in cids]
        bp2 = ax2.boxplot(ul_data, tick_labels=cids, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
        for patch in bp2['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
        ax2.set_xlabel('Client ID')
        ax2.set_ylabel('Uplink Latency (s)')
        #ax2.set_title(f'Uplink Latency - {format_sweep_label(sweep_param, sweep_val)}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_sweep_val = str(sweep_val).replace('/', '_').replace(' ', '_')
        filename = f"latency_by_cid_{sweep_param}_{safe_sweep_val}.svg"
        fig.savefig(output_dir / filename, format='svg', bbox_inches='tight')
        plt.close(fig)

    # Figure 2: Box plots for all sweep values showing UL/DL distributions side by side
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for box plots
    all_data = []
    all_labels = []
    positions = []
    colors_list = []

    pos = 1
    for i, sweep_val in enumerate(sweep_values):
        subset = combined_df[combined_df[sweep_param] == sweep_val]
        if len(subset) > 0:
            # Add downlink data
            all_data.append(subset['downlink_latency'].values)
            all_labels.append(f"{format_sweep_label(sweep_param, sweep_val)}\nDL")
            positions.append(pos)
            colors_list.append(colors[i])

            # Add uplink data
            all_data.append(subset['uplink_latency'].values)
            all_labels.append(f"{format_sweep_label(sweep_param, sweep_val)}\nUL")
            positions.append(pos + 0.5)
            colors_list.append(colors[i])

            pos += 1.5  # Space between different sweep values

    # Create box plot
    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.4, medianprops=dict(color='black', linewidth=1.5), showfliers=False)

    # Color the boxes
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_list[i])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_ylabel('Latency (s)')
    #ax.set_title(f'Latency Distribution by {sweep_param.title()}')
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7,
                            label=format_sweep_label(sweep_param, val))
                      for i, val in enumerate(sweep_values)]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    filename = f"latency_boxplot_{sweep_param}_sweep.svg"
    fig.savefig(output_dir / filename, format='svg', bbox_inches='tight')
    plt.close(fig)

def plot_cellular_sweep(experiment_paths: list, filters: dict, sweep: str, output_dir: Path):
    # Clear filter log for this sweep
    global FILTER_LOG
    FILTER_LOG = []

    filtered = [exp for exp in experiment_paths
                if all(exp.get(k) in (v, None) for k, v in filters.items())]

    filter_parts = [f"{k}_{str(v).replace('/', '_').replace(' ', '_')}" for k, v in filters.items()]
    filter_dir = "_".join(filter_parts)
    sweep_output_dir = output_dir / filter_dir / sweep
    sweep_output_dir.mkdir(exist_ok=True, parents=True)

    agg_metrics = []
    individual_metrics = []
    latency_metrics = []

    for exp in filtered:
        metrics = load(exp)
        common_data = {**exp, 'execution_time': metrics.get('execution_time', None),
                       'start_time': metrics.get('start_time', None), 'sweep': sweep}
        agg_metrics.append({**common_data, 'metrics': metrics['server_agg_metric']})
        if sweep != 'network':
            individual_metrics.append({**common_data, 'metrics': metrics['individual_metrics']})
        latency_metrics.append({**common_data, 'metrics': metrics['latency']})

    if sweep != 'network':
        plot_agg_metric(agg_metrics, sweep_output_dir, "", sweep)
        plot_agg_metric_vs_time(agg_metrics, sweep_output_dir, "", sweep)
        plot_individual(individual_metrics, sweep_output_dir, "", sweep)
        plot_latency_metrics(latency_metrics, sweep_output_dir, "", sweep)
        tost_test(individual_metrics, sweep, sweep_output_dir)
    else:
        convergence_analysis(agg_metrics, sweep_output_dir, "", sweep)
        plot_latency_metrics(latency_metrics, sweep_output_dir, "", sweep)



    # Calculate individual metric statistics and save to CSV
    #sweep_anova(individual_metrics, sweep, sweep_output_dir)


def sweep_anova(metrics, sweep_param, output_dir):
    """Perform ANOVA on individual metrics across different sweep values"""

    df = prepare_individual_metrics(metrics, sweep_param)

    anova_results = []

    for metric in TIME_METRICS:
        for cid in sorted(df['cid'].unique()):
            cid_data = df[df['cid'] == cid]

            groups = {name: group[metric].dropna().values
                      for name, group in cid_data.groupby(sweep_param)}

            if len(groups) < 2:
                continue

            group_items = list(groups.values())
            group_names = list(groups.keys())

            f_stat, p_val = stats.f_oneway(*group_items)

            anova_results.append({'cid': cid,
                                  'metric': metric,
                                  'sweep_param': sweep_param,
                                  'f_statistic': f_stat,
                                  'p_value': p_val,
                                  'num_groups': len(groups)})

            """# Plot each distribution for visual inspection
            plt.figure(figsize=(8, 4))
            for name, data in groups.items():
                sns.kdeplot(data, label=f"{sweep_param}={name}", fill=True)
            plt.title(f'Distribution of {metric} for CID {cid}\nANOVA p={p_val:.9f}')
            plt.xlabel(column_name_map.get(metric, metric))
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.show()"""


            # Pairwise t-tests if p < 0.005
            if p_val < 0.005:
                combs = list(set(itertools.combinations(group_names, 2)))
                for i, j in combs:
                    g1 = groups.get(i)
                    g2 = groups.get(j)

                    t_stat, pair_p_val = stats.ttest_ind(g1, g2)

                    """# Plot each distribution for visual inspection
                    plt.figure(figsize=(8, 4))
                    sns.kdeplot(g1, label=f"{sweep_param}={i}", fill=True)
                    sns.kdeplot(g2, label=f"{sweep_param}={j}", fill=True)
                    plt.title(f'Distribution of {metric} for CID {cid}\n{sweep_param}={i} vs {sweep_param}={j} | p={pair_p_val:.9f}')
                    plt.xlabel(column_name_map.get(metric, metric))
                    plt.ylabel('Density')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()"""

                    print('Pairwise t-test:', cid, metric, i, j, pair_p_val)


    pd.DataFrame(anova_results).to_csv(
        output_dir / f'anova_{sweep_param}.csv', index=False)


    return anova_results

def tost_test(metrics, sweep_param, output_dir):
    df = prepare_individual_metrics(metrics, sweep_param)
    results = []
    for metric in TIME_METRICS:
        client_means = df.groupby('cid')[metric].mean()
        cv_between_clients = client_means.std() / client_means.mean()
        print(f'CV between clients for {metric}: {cv_between_clients}')
        print(f'Metric {metric} range across clients: {client_means.min()} - {client_means.max()}')

        for cid in df['cid'].unique():
            cid_data = df[df['cid'] == cid]
            cid_mean = cid_data[metric].mean()



            delta = 0.05 * cid_mean

            tdd_groups = {name: group[metric].dropna().values
                      for name, group in cid_data.groupby(sweep_param)}

            n_pairs = 0
            n_equiv = 0

            for tdd1, tdd2 in itertools.combinations(tdd_groups.keys(), 2):
                tost_result = pg.tost(
                    tdd_groups[tdd1], tdd_groups[tdd2], delta
                )
                p_val = tost_result['pval'].values[0]
                equiv = p_val < 0.05


                n_pairs += 1
                if equiv:
                    n_equiv += 1

            results.append({
                'metric': metric,
                'CID': cid,
                'mean_metric': cid_mean,
                'bound_5pct': delta,
                'equiv_pairs': n_equiv,
                'pct_equiv': 100 * n_equiv / n_pairs if n_pairs > 0 else 0,

            })

    equiv_df = pd.DataFrame(results)
    equiv_df.to_csv(
        output_dir / f'tost_equivalence_{sweep_param}.csv', index=False)

def convergence_analysis(agg_metrics, output_dir, filter_str, sweep_param):
    """Accuracy and loss convergence analysis across experiments
        When loss has no improvement if eval_loss does not decrease for 10 rounds, return number of rounds
        Plot eval_acc vs time, eval_loss vs time using this round number.
        Only perform analysis if it is a network sweep
    """
    print('stop')

    # Calcualte convergence time on wwan experiment (inplace)
    wwan_exp = next((exp for exp in agg_metrics if exp['network'] == 'wwan'), None)
    if wwan_exp is None:
        print("No wwan experiment found for convergence analysis")
        return

    wwan_df = wwan_exp['metrics']

    # Determine convergence round based on eval_loss
    patience = 20
    tolerance = 0.01
    best_loss = float('inf')
    count = 0
    early_stopping_round = None
    for i, loss in enumerate(wwan_df['eval_loss']):
        if loss < best_loss - tolerance:
            best_loss = loss
            count = 0
        else:
            count += 1
            if count >= patience:
                early_stopping_round = i + 1
                break
    print(f'Stopping triggered at round: {early_stopping_round}')

    # Truncate the df
    if early_stopping_round is not None:
        wwan_df = wwan_df.iloc[:early_stopping_round]

    # Plot eval_acc vs time and eval_loss vs time for all experiments
    wwan_exp['metrics'] = wwan_df

    plot_agg_metric(agg_metrics, output_dir, filter_str, sweep_param)
    plot_agg_metric_vs_time(agg_metrics, output_dir, filter_str, sweep_param)

if __name__ == '__main__':
    directory = Path("/Users/roberthayek/Documents/git_repos/fed_5g/IMC")
    output_dir = Path.cwd()
    print(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    all_experiments = []
    for exp in directory.iterdir():
        if not exp.is_dir():
            continue
        params = parse_experiment_name(exp.name)
        all_experiments.append({'path': exp, **params})

    """plot_cellular_sweep(all_experiments, {'bandwidth': '100 MHz', 'rank': '2x2',
                                          'distribution': 'dirichlet', 'congestion': False, 'tdd': '7-2', 'network': 'wwan'},
                        sweep='nodes', output_dir=output_dir)
    """

    # NETWORK COMPARISON: 7-2 TDD, 40 MHz, 2x2, iid, no congestion
    plot_cellular_sweep(all_experiments, {'bandwidth': '40 MHz', 'rank': '2x2',
                                          'distribution': 'iid', 'congestion': False, 'tdd': '7-2', 'nodes': '6N'},
                        sweep='network', output_dir=output_dir)



