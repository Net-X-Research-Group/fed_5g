import pandas as pd
import re
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

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

network_map = {
    'wwan': '5G',
    'wlan': 'Wifi',
    'lan': 'Ethernet'
}

TIME_METRICS = ['train_time', 'eval_time']


def parse_experiment_name(name):
    params = {
        'bandwidth': None,
        'tdd': None,
        'nodes': None,
        'rank': '1x1',
        'distribution': 'dirichlet',
        'congestion': False,
        'network': 'wwan'
    }

    parts = name.split('_')

    for part in parts:
        if re.search(r"\d+MHz", part, re.IGNORECASE):
            match = re.match(r"(\d+)(MHz)", part, re.IGNORECASE)
            if match:
                params['bandwidth'] = match.group(1) + ' MHz'
        elif re.match(r"\d+N$", part):
            params['nodes'] = part
        elif re.match(r"\d+-\d+$", part):
            params['tdd'] = part
        elif re.search(r"MIMO", part, re.IGNORECASE):
            params['rank'] = '2x2'
        elif re.search(r"SISO", part, re.IGNORECASE):
            params['rank'] = '1x1'
        elif re.search(r"Dirichlet", part, re.IGNORECASE):
            params['distribution'] = 'dirichlet'
        elif re.search(r"IID", part, re.IGNORECASE):
            params['distribution'] = 'iid'
        elif re.search('Congestion', part, re.IGNORECASE):
            params['congestion'] = True
        elif re.search(r"wlan", part, re.IGNORECASE):
            params['network'] = 'wlan'
        elif re.search(r"wwan", part, re.IGNORECASE):
            params['network'] = 'wwan'
        elif re.search(r"lan", part, re.IGNORECASE):
            params['network'] = 'lan'

    return params


def load(experiment_path: dict) -> dict:
    """Load a single experiment return metrics"""
    metrics = {}

    # Get latencies from a single experiment
    latencies = list(experiment_path['path'].glob('latency_*_CID*.csv'))
    if latencies:
        latency_dfs = []
        for latency in latencies:
            cid_match = re.search(r"CID(\d+)", latency.name)
            if cid_match:
                cid = int(cid_match.group(1))
                df = pd.read_csv(latency)
                df['cid'] = cid
                df['server_round'] = range(1, len(df) + 1)
                latency_dfs.append(df)
        if latency_dfs:
            metrics['latency'] = pd.concat(latency_dfs, ignore_index=True)

    # Get aggregated metrics from csv
    agg_file = experiment_path['path'] / 'train_agg_metrics.csv'
    if agg_file.exists():
        metrics['server_agg_metric'] = pd.read_csv(agg_file)

    # Load the .json individual metrics
    individual_file = experiment_path['path'] / 'individual_metrics.json'
    if individual_file.exists():
        with open(individual_file, 'r') as f:
            individual_metrics = json.load(f)
            individual_metrics = {k: v['train'] for k, v in individual_metrics.items()}
            metrics['individual_metrics'] = individual_metrics

    # Load the exec time
    exec_time_file = experiment_path['path'] / 'execution_time.txt'
    if exec_time_file.exists():
        with open(exec_time_file, 'r') as f:
            time_str = f.read().strip().rstrip('s')
            metrics['execution_time'] = float(time_str)

    # Load start time
    start_time_file = experiment_path['path'] / 'start_time.txt'
    if start_time_file.exists():
        with open(start_time_file, 'r') as f:
            time_str = f.read().strip().rstrip('s')
            metrics['start_time'] = float(time_str)

    return metrics


def sort_experiments_by_sweep(experiments, sweep_param):
    """Sort experiments by sweep parameter value"""
    if sweep_param == 'bandwidth':
        return sorted(experiments, key=lambda x: int(x['bandwidth'].replace(' MHz', '')))
    elif sweep_param == 'tdd':
        return sorted(experiments, key=lambda x: tuple(map(int, x['tdd'].split('-'))))
    else:
        return sorted(experiments, key=lambda x: x[sweep_param])


def format_sweep_label(sweep_param, sweep_value):
    """Generate display label for sweep parameter value"""
    if sweep_param == 'network':
        return network_map.get(sweep_value, sweep_value)
    elif sweep_param == 'tdd':
        return sweep_value.replace('-', ':')
    elif sweep_param == 'bandwidth':
        return sweep_value
    else:
        return f"{sweep_param.title()} {sweep_value}"


def save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str, suffix=""):
    """Save figure with consistent naming and close it"""
    if filter_str == "":
        filename = f"{metric}{suffix}_{sweep_param}_sweep.png"
    else:
        filename = f"{metric}{suffix}_{filter_str}.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_agg_metric(agg_metrics, output_dir, filter_str, sweep_param):
    """Create one figure per metric, with a line for each experiment"""
    sns.set_style("whitegrid")
    agg_metrics = sort_experiments_by_sweep(agg_metrics, sweep_param)

    metric_cols = [col for col in agg_metrics[0]['metrics'].columns
                   if col not in ['server_round', 'timestamp']]

    figs = {}
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        display_name = column_name_map.get(metric, metric.replace('_', ' ').title())

        if metric in TIME_METRICS:
            # Frequency plot for time metrics
            for exp in agg_metrics:
                label = format_sweep_label(sweep_param, exp[sweep_param])
                sns.kdeplot(exp['metrics'][metric], label=label, alpha=0.5, ax=ax)
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

        # Main plot: Line plot vs elapsed time
        for exp in agg_metrics:
            df = exp['metrics'].copy()
            df['elapsed_time'] = df['timestamp'] - df['timestamp'].iloc[0]
            label = format_sweep_label(sweep_param, exp[sweep_param])
            sns.lineplot(x=df['elapsed_time'], y=df[metric],
                         label=label, linewidth=2, ax=ax_main)

        # Distribution subplot: histogram of round durations
        for exp in agg_metrics:
            df = exp['metrics'].copy()
            df['round_duration'] = df['timestamp'].diff()
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


def prepare_individual_metrics(metrics, sweep_param):
    """Flatten individual metrics and add sweep parameter"""
    processed = []
    for exp in metrics:
        rows = []
        for server_round, clients in exp['metrics'].items():
            for client in clients:
                row = {'server_round': int(server_round), **client}
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(['cid', 'server_round'])
        df['round_duration'] = df.groupby('cid')['timestamp'].diff()
        df[sweep_param] = exp[sweep_param]
        processed.append(df)

    return pd.concat(processed, ignore_index=True)


def get_sweep_colors_markers(sweep_values):
    """Get consistent colors and markers for sweep configurations"""
    n_sweeps = len(sweep_values)
    colors = sns.color_palette("Set2", n_colors=n_sweeps)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:n_sweeps]
    return colors, markers


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
        plot_df = combined_df[combined_df[metric] > 0].copy()
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
                alpha=1,
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

    # Apply outlier filtering using IQR method + reasonable upper bounds
    def filter_outliers(data, column_name):
        """Filter extreme outliers using IQR method with additional sanity checks"""
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1

        # IQR-based bounds (1.5 * IQR is standard, but we'll be more conservative)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Additional sanity bounds for latency (in seconds)
        # Normal cellular latency should be < 10 seconds for most cases
        sanity_upper_bound = 10.0  # 10 seconds max - anything higher is likely an error
        sanity_lower_bound = 0.0001  # 0.1ms min

        # Use the more restrictive of IQR or sanity bounds
        final_upper_bound = min(upper_bound, sanity_upper_bound)
        final_lower_bound = max(lower_bound, sanity_lower_bound)

        # Filter data
        filtered_data = data[
            (data[column_name] >= final_lower_bound) &
            (data[column_name] <= final_upper_bound)
        ]

        outliers_removed = len(data) - len(filtered_data)
        if outliers_removed > 0:
            print(f"Filtered {outliers_removed} outliers from {column_name} "
                  f"(bounds: {final_lower_bound:.3f}-{final_upper_bound:.1f}s)")

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
        bp1 = ax1.boxplot(dl_data, tick_labels=cids, patch_artist=True)
        for patch in bp1['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
        ax1.set_xlabel('Client ID')
        ax1.set_ylabel('Downlink Latency (s)')
        ax1.set_title(f'Downlink Latency - {format_sweep_label(sweep_param, sweep_val)}')
        ax1.grid(True, alpha=0.3)

        # Uplink latency by CID
        ul_data = [subset[subset['cid'] == cid]['uplink_latency'].values for cid in cids]
        bp2 = ax2.boxplot(ul_data, tick_labels=cids, patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
        ax2.set_xlabel('Client ID')
        ax2.set_ylabel('Uplink Latency (s)')
        ax2.set_title(f'Uplink Latency - {format_sweep_label(sweep_param, sweep_val)}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_sweep_val = str(sweep_val).replace('/', '_').replace(' ', '_')
        filename = f"latency_by_cid_{sweep_param}_{safe_sweep_val}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
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
    bp = ax.boxplot(all_data, positions=positions, patch_artist=True, widths=0.4)

    # Color the boxes
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_list[i])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_ylabel('Latency (s)')
    ax.set_title(f'Latency Distribution by {sweep_param.title()} (Outliers Filtered)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7,
                            label=format_sweep_label(sweep_param, val))
                      for i, val in enumerate(sweep_values)]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    filename = f"latency_boxplot_{sweep_param}_sweep.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_cellular_sweep(experiment_paths: list, filters: dict, sweep: str, output_dir: Path):
    filtered = [exp for exp in experiment_paths
                if all(exp.get(k) == v for k, v in filters.items())]

    filter_parts = [f"{k}_{str(v).replace('/', '_').replace(' ', '_')}" for k, v in filters.items()]
    filter_dir = "_".join(filter_parts)
    sweep_output_dir = output_dir / filter_dir / sweep
    sweep_output_dir.mkdir(exist_ok=True, parents=True)

    agg_metrics = []
    individual_metrics = []
    latency_metrics = []

    for exp in filtered:
        metrics = load(exp)
        common_data = {**exp, 'execution_time': metrics['execution_time'],
                       'start_time': metrics['start_time'], 'sweep': sweep}
        agg_metrics.append({**common_data, 'metrics': metrics['server_agg_metric']})
        individual_metrics.append({**common_data, 'metrics': metrics['individual_metrics']})
        latency_metrics.append({**common_data, 'metrics': metrics['latency']})

    plot_agg_metric(agg_metrics, sweep_output_dir, "", sweep)
    plot_agg_metric_vs_time(agg_metrics, sweep_output_dir, "", sweep)
    plot_individual(individual_metrics, sweep_output_dir, "", sweep)
    plot_latency_metrics(latency_metrics, sweep_output_dir, "", sweep)


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

    plot_cellular_sweep(all_experiments, {'bandwidth': '40 MHz', 'rank': '2x2',
                                          'distribution': 'dirichlet', 'congestion': False, 'nodes': '6N'},
                        sweep='tdd', output_dir=output_dir)