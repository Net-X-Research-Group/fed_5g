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


def parse_experiment_name(name):
    params = {
        'bandwidth': None,
        'tdd': None,
        'nodes': None,
        'rank': '1x1', # Default
        'distribution': 'dirichlet',  # default
        'congestion': False,
        'network': 'wwan' # Default
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


def load(experiment_path: Path) -> dict:
    """
    Load a single experiment return metrics
    Args:
        experiment_path:

    Returns:

    """

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
            individual_metrics = {k: v['train'] for k, v in individual_metrics.items()} # Flatten ['train']
            metrics['individual_metrics']  = individual_metrics

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


def plot_agg_metric(agg_metrics, output_dir, filter_str, sweep_param):
    """
    Create one figure per metric, with a line for each experiment.
    Time metrics get frequency plots instead.

    Args:
        agg_metrics: List of dicts with params and 'metrics' (DataFrame)
        output_dir: Path to save figures
        filter_str: String representation of filters for filename
        sweep_param: Parameter being swept

    Returns:
        dict: {metric_name: figure} for each metric plotted
    """
    # Default mapping
    column_name_map = {
        'train_loss': 'Training Loss',
        'train_time': 'Training Time (s)',
        'eval_loss': 'Evaluation Loss',
        'eval_acc': 'Evaluation Accuracy',
        'eval_time': 'Evaluation Time (s)'
    }

    # Network type mapping
    network_map = {
        'wwan': '5G',
        'wlan': 'Wifi',
        'lan': 'Ethernet'
    }

    sns.set_style("whitegrid")

    # Sort experiments by sweep parameter
    if sweep_param == 'bandwidth':
        agg_metrics = sorted(agg_metrics, key=lambda x: int(x['bandwidth'].replace(' MHz', '')))
    elif sweep_param == 'tdd':
        agg_metrics = sorted(agg_metrics, key=lambda x: tuple(map(int, x['tdd'].split('-'))))
    else:
        agg_metrics = sorted(agg_metrics, key=lambda x: x[sweep_param])

    # Get metric columns (exclude server_round, timestamp)
    metric_cols = [col for col in agg_metrics[0]['metrics'].columns
                   if col not in ['server_round', 'timestamp']]

    time_metrics = ['train_time', 'eval_time']

    figs = {}
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))

        display_name = column_name_map.get(metric, metric.replace('_', ' ').title())

        if metric in time_metrics:
            # Frequency plot for time metrics
            for exp in agg_metrics:
                df = exp['metrics']
                sweep_value = exp[sweep_param]
                if sweep_param == 'network':
                    label = network_map.get(sweep_value, sweep_value)
                elif sweep_param == 'tdd':
                    label = sweep_value.replace('-', ':')
                elif sweep_param == 'bandwidth':
                    label = sweep_value  # Already has space
                else:
                    label = f"{sweep_param.title()} {sweep_value}"
                sns.kdeplot(df[metric], label=label, alpha=0.5, ax=ax)

            ax.set_xlabel(display_name, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
        else:
            # Line plot for other metrics
            for exp in agg_metrics:
                df = exp['metrics']
                sweep_value = exp[sweep_param]
                if sweep_param == 'network':
                    label = network_map.get(sweep_value, sweep_value)
                elif sweep_param == 'tdd':
                    label = sweep_value.replace('-', ':')
                elif sweep_param == 'bandwidth':
                    label = sweep_value  # Already has space
                else:
                    label = f"{sweep_param.title()} {sweep_value}"
                sns.lineplot(x=df['server_round'], y=df[metric],
                             label=label, linewidth=2, ax=ax)

            ax.set_xlabel('Server Round', fontsize=12)
            ax.set_ylabel(display_name, fontsize=12)

        ax.legend()

        # Save figure
        filename = f"{metric}_{sweep_param}_sweep.png" if filter_str == "" else f"{metric}_{filter_str}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        figs[metric] = fig

    return figs

def plot_agg_metric_vs_time(agg_metrics, output_dir, filter_str, sweep_param):
    # Default mapping
    column_name_map = {
        'train_loss': 'Training Loss',
        'train_time': 'Training Time (s)',
        'eval_loss': 'Evaluation Loss',
        'eval_acc': 'Evaluation Accuracy',
        'eval_time': 'Evaluation Time (s)'
    }

    # Network type mapping
    network_map = {
        'wwan': '5G',
        'wlan': 'Wifi',
        'lan': 'Ethernet'
    }

    sns.set_style("whitegrid")

    # Sort experiments by sweep parameter
    if sweep_param == 'bandwidth':
        agg_metrics = sorted(agg_metrics, key=lambda x: int(x['bandwidth'].replace(' MHz', '')))
    elif sweep_param == 'tdd':
        agg_metrics = sorted(agg_metrics, key=lambda x: tuple(map(int, x['tdd'].split('-'))))
    else:
        agg_metrics = sorted(agg_metrics, key=lambda x: x[sweep_param])

    # Get metric columns (exclude server_round, timestamp)
    metric_cols = [col for col in agg_metrics[0]['metrics'].columns
                   if col not in ['server_round', 'timestamp']]

    time_metrics = ['train_time', 'eval_time']

    figs = {}
    for metric in metric_cols:
        if metric in time_metrics:
            continue

        # Create figure with GridSpec for main plot + small subplot
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)

        ax_main = fig.add_subplot(gs[0])
        ax_dist = fig.add_subplot(gs[1])

        display_name = column_name_map.get(metric, metric.replace('_', ' ').title())

        # Main plot: Line plot vs elapsed time
        for exp in agg_metrics:
            df = exp['metrics'].copy()
            # Normalize timestamp to elapsed time
            df['elapsed_time'] = df['timestamp'] - df['timestamp'].iloc[0]

            sweep_value = exp[sweep_param]
            if sweep_param == 'network':
                label = network_map.get(sweep_value, sweep_value)
            elif sweep_param == 'tdd':
                label = sweep_value.replace('-', ':')
            elif sweep_param == 'bandwidth':
                label = sweep_value  # Already has space
            else:
                label = f"{sweep_param.title()} {sweep_value}"
            sns.lineplot(x=df['elapsed_time'], y=df[metric],
                         label=label, linewidth=2, ax=ax_main)

        # Distribution subplot: histogram of round durations
        for exp in agg_metrics:
            df = exp['metrics'].copy()
            df['round_duration'] = df['timestamp'].diff()

            sweep_value = exp[sweep_param]
            if sweep_param == 'network':
                label = network_map.get(sweep_value, sweep_value)
            elif sweep_param == 'tdd':
                label = sweep_value.replace('-', ':')
            elif sweep_param == 'bandwidth':
                label = sweep_value  # Already has space
            else:
                label = f"{sweep_param.title()} {sweep_value}"

            sns.kdeplot(df['round_duration'].dropna(),
                        label=label, linewidth=2, ax=ax_dist)

        ax_main.set_xlabel('')  # Remove x-label from main plot
        ax_main.set_ylabel(display_name, fontsize=12)
        ax_main.legend(title='Bandwidth' if sweep_param == 'bandwidth' else sweep_param.title(), loc='best')
        ax_main.grid(True, alpha=0.3)

        ax_dist.set_xlabel('Round Duration (s)', fontsize=10)
        ax_dist.set_ylabel('Density', fontsize=10)
        ax_dist.tick_params(labelsize=9)
        ax_dist.grid(True, alpha=0.3, axis='y')

        # Save figure
        filename = f"{metric}_vs_time_{sweep_param}_sweep.png" if filter_str == "" else f"{metric}_vs_time_{filter_str}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

        figs[metric] = fig

    return figs


def plot_individual(metrics: list, output_dir: Path, filter_str: str, sweep_param: str):
    column_name_map = {
        'train_loss': 'Training Loss',
        'train_time': 'Training Time (s)',
        'eval_loss': 'Evaluation Loss',
        'eval_acc': 'Evaluation Accuracy',
        'eval_time': 'Evaluation Time (s)',
        'round_duration': 'Round Duration (s)',
    }

    # Network type mapping
    network_map = {
        'wwan': '5G',
        'wlan': 'Wifi',
        'lan': 'Ethernet'
    }

    # Clean and flatten
    for idx, metric in enumerate(metrics):
        metrics_dict = metrics[idx]['metrics']
        rows = []
        for server_round, clients in metrics_dict.items():
            for client in clients:
                row = {'server_round': int(server_round), **client}
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(['server_round', 'cid'])
        metrics[idx]['metrics'] = df

    all_metrics = []
    for exp in metrics:
        df = exp['metrics'].copy()
        df = df.sort_values(['cid', 'server_round'])
        df['round_duration'] = df.groupby('cid')['timestamp'].diff()
        df[sweep_param] = exp[sweep_param]
        all_metrics.append(df)

    combined_df = pd.concat(all_metrics, ignore_index=True)

    # Sort sweep values properly
    sweep_values = combined_df[sweep_param].unique()
    if sweep_param == 'bandwidth':
        sweep_values = sorted(sweep_values, key=lambda x: int(x.replace(' MHz', '')))
    elif sweep_param == 'tdd':
        sweep_values = sorted(sweep_values, key=lambda x: tuple(map(int, x.split('-'))))
    else:
        sweep_values = sorted(sweep_values)

    rd = combined_df['round_duration'].dropna()
    print(f"Min: {rd.min():.2f}, Max: {rd.max():.2f}, Median: {rd.median():.2f}")
    print(f"Mean: {rd.mean():.2f}, Std: {rd.std():.2f}")
    print(f"Negative values: {(rd < 0).sum()}")
    print(f"99th percentile: {rd.quantile(0.99):.2f}")

    # Check per-client
    print(combined_df.groupby('cid')['round_duration'].describe())


    # Box plot for each CID for round duration
    time_metrics = ['train_time', 'eval_time']

    for metric in time_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))

        n_sweeps = len(sweep_values)
        cids = sorted(combined_df['cid'].unique())

        # Define colors and markers
        colors = sns.color_palette("Set2", n_colors=n_sweeps)
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:n_sweeps]

        # Manual boxplot with custom styling
        width = 0.8 / n_sweeps  # Box width per sweep config

        for i, sweep_val in enumerate(sweep_values):
            subset = combined_df[combined_df[sweep_param] == sweep_val]

            # Positions offset per sweep config
            positions = np.arange(len(cids)) + (i - n_sweeps / 2 + 0.5) * width

            # Data grouped by CID
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

        # Y-limits (per metric)
        lower = combined_df[metric].quantile(0.05)
        upper = combined_df[metric].quantile(0.95)
        margin = (upper - lower) * 0.1
        ax.set_ylim(lower - margin, upper + margin)

        # Formatting
        ax.set_xticks(np.arange(len(cids)))
        ax.set_xticklabels(cids)
        ax.set_xlabel('Client ID', fontsize=12)
        ax.set_ylabel(column_name_map[metric], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker=markers[i], color='w',
                   markerfacecolor=colors[i], markersize=8,
                   label=network_map.get(val, f'{sweep_param.title()} = {val}') if sweep_param == 'network'
                   else val.replace('-', ':') if sweep_param == 'tdd'
                   else val if sweep_param == 'bandwidth'
                   else f'{sweep_param.title()} = {val}')
            for i, val in enumerate(sweep_values)
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        # Save figure
        filename = f"{metric}_by_client_{sweep_param}_sweep.png" if filter_str == "" else f"{metric}_by_client_{filter_str}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    for metric in ['round_duration']:
        fig, ax = plt.subplots(figsize=(12, 6))

        plot_df = combined_df[combined_df[metric] > 0].copy()

        colors = sns.color_palette("Set2", n_colors=len(sweep_values))

        for i, sweep_val in enumerate(sweep_values):
            subset = plot_df[plot_df[sweep_param] == sweep_val][metric]

            if sweep_param == 'network':
                label = network_map.get(sweep_val, sweep_val)
            elif sweep_param == 'tdd':
                label = sweep_val.replace('-', ':')
            elif sweep_param == 'bandwidth':
                label = sweep_val  # Already has space
            else:
                label = f'{sweep_param}={sweep_val}'

            sns.kdeplot(
                data=subset,
                ax=ax,
                label=label,
                color=colors[i],
                linewidth=2.5,
                alpha=1,
                fill=False,
                common_norm=False  # Each curve normalized independently
            )

        ax.set_xlabel(column_name_map[metric], fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

        # Optional: Add vertical lines for medians
        for i, sweep_val in enumerate(sweep_values):
            subset = plot_df[plot_df[sweep_param] == sweep_val][metric]
            median_val = subset.median()
            ax.axvline(median_val, color=colors[i], linestyle='--',
                       linewidth=1.5, alpha=0.8)

        # Save figure
        filename = f"{metric}_distribution_{sweep_param}_sweep.png" if filter_str == "" else f"{metric}_distribution_{filter_str}.png"
        fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_cellular_sweep(experiment_paths: list, filters: dict, sweep: str, output_dir: Path):
    filtered = [exp for exp in experiment_paths
                if all(exp.get(k) == v for k, v in filters.items())]

    # Create subdirectory path based on filters
    filter_parts = [f"{k}_{str(v).replace('/', '_').replace(' ', '_')}" for k, v in filters.items()]
    filter_dir = "_".join(filter_parts)

    # Create nested subdirectory structure: output_dir / filter_dir / sweep
    sweep_output_dir = output_dir / filter_dir / sweep
    sweep_output_dir.mkdir(exist_ok=True, parents=True)

    agg_metrics = []
    individual_metrics = []
    latency_metrics = []

    for exp in filtered:
        metrics = load(exp)
        agg_metrics.append({**exp, 'metrics': metrics['server_agg_metric'], 'execution_time': metrics['execution_time'],
                            'start_time': metrics['start_time'], 'sweep': sweep})
        individual_metrics.append(
            {**exp, 'metrics': metrics['individual_metrics'], 'execution_time': metrics['execution_time'],
             'start_time': metrics['start_time'], 'sweep': sweep})
        latency_metrics.append(
            {**exp, 'metrics': metrics['latency'], 'execution_time': metrics['execution_time'],
             'start_time': metrics['start_time'], 'sweep': sweep})


    # Plot agg metrics (original)
    plot_agg_metric(agg_metrics, sweep_output_dir, "", sweep)
    plot_agg_metric_vs_time(agg_metrics, sweep_output_dir, "", sweep)


    # Plot individual metrics
    plot_individual(individual_metrics, sweep_output_dir, "", sweep)

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


    plot_cellular_sweep(all_experiments, {'tdd': '7-2', 'bandwidth': '100 MHz', 'rank': '2x2', 'distribution': 'dirichlet', 'congestion': False}, sweep='nodes', output_dir=output_dir)
