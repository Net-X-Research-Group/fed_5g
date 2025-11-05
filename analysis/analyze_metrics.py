import numpy as np
import pandas as pd
import seaborn as sns
import re
from pathlib import Path
import json
import torch
from matplotlib import pyplot as plt


def parse_experiment_name(name):
    params = {
        'bandwidth': None,
        'tdd': None,
        'nodes': None,
        'rank': '1x1',
    }

    parts = name.split('_')

    for part in parts:
        if re.search(r"\d+MHz", part, re.IGNORECASE):
            match = re.match(r"(\d+)(MHz)", part, re.IGNORECASE)
            if match:
                params['bandwidth'] = match.group(1) + 'MHz'
        elif re.match(r"\d+N$", part):
            params['nodes'] = part
        elif re.match(r"\d+-\d+$", part):
            params['tdd'] = part
        elif re.search(r"MIMO", part, re.IGNORECASE):
            params['rank'] = '2x2'

    return params


def load(experiment_path):
    """
    Load a single experiment and return metrics.

    Args:
        experiment_path: Dict with 'path' key and experiment params

    Returns:
        Dict with all metrics from the experiment
    """
    metrics = {}

    # Get latencies from a single experiment
    latencies = list(experiment_path['path'].glob('latency_*_CID*.csv'))
    if latencies:
        latency_dfs = []
        for latency in sorted(latencies):
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
            metrics['individual_metrics'] = json.load(f)

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


def aggregate_metrics(metrics):
    """
    Aggregate raw metrics into summary statistics.

    Args:
        metrics: Dict returned by load()

    Returns:
        Dict with aggregated statistics
    """
    agg = {}

    # Latency aggregates
    if 'latency' in metrics:
        lat_df = metrics['latency']
        agg['downlink_mean'] = lat_df['downlink_latency'].mean()
        agg['downlink_median'] = lat_df['downlink_latency'].median()
        agg['downlink_p95'] = lat_df['downlink_latency'].quantile(0.95)
        agg['downlink_std'] = lat_df['downlink_latency'].std()
        agg['downlink_min'] = lat_df['downlink_latency'].min()
        agg['downlink_max'] = lat_df['downlink_latency'].max()

        agg['uplink_mean'] = lat_df['uplink_latency'].mean()
        agg['uplink_median'] = lat_df['uplink_latency'].median()
        agg['uplink_p95'] = lat_df['uplink_latency'].quantile(0.95)
        agg['uplink_std'] = lat_df['uplink_latency'].std()
        agg['uplink_min'] = lat_df['uplink_latency'].min()
        agg['uplink_max'] = lat_df['uplink_latency'].max()

    # Training aggregates
    if 'server_agg_metric' in metrics:
        train_df = metrics['server_agg_metric']
        final_round = train_df.iloc[-1]

        agg['final_train_loss'] = final_round['train_loss']
        agg['final_eval_loss'] = final_round['eval_loss']
        agg['final_eval_acc'] = final_round['eval_acc']
        agg['mean_train_time'] = train_df['train_time'].mean()
        agg['mean_eval_time'] = train_df['eval_time'].mean()
        agg['std_train_time'] = train_df['train_time'].std()
        agg['total_rounds'] = len(train_df)

        # Additional training metrics
        agg['best_eval_acc'] = train_df['eval_acc'].max()
        agg['best_eval_loss'] = train_df['eval_loss'].min()
        agg['convergence_round'] = train_df['eval_acc'].idxmax() + 1

    # Model aggregates
    if 'model_metric' in metrics and metrics['model_metric'] is not None:
        agg['total_params'] = metrics['model_metric']['total_params']
        agg['model_size_mb'] = metrics['model_metric']['model_size_mb']
        agg['layer_count'] = metrics['model_metric']['layer_count']

    # Execution time
    if 'execution_time' in metrics:
        agg['execution_time'] = metrics['execution_time']

    if 'start_time' in metrics:
        agg['start_time'] = metrics['start_time']

    return agg


def compare(experiments, filters, sweep_param):
    """
    Compare experiments by filtering and sweeping one parameter.

    Args:
        experiments: List of all experiment dicts
        filters: Dict of fixed parameters (e.g., {'bandwidth': '100MHz', 'nodes': '6N'})
        sweep_param: Parameter to vary (e.g., 'tdd' or 'rank')

    Returns:
        DataFrame with one row per experiment, aggregated metrics as columns
    """
    # Filter to experiments matching the fixed parameters
    filtered = [
        exp for exp in experiments
        if all(exp.get(k) == v for k, v in filters.items())
    ]

    print(f"Found {len(filtered)} experiments matching filters: {filters}")

    results = []
    for exp in filtered:
        print(f"  Loading {exp['path'].name}...")

        # Load raw metrics
        metrics = load(exp)

        # Aggregate into summary statistics
        agg = aggregate_metrics(metrics)

        # Build result row
        row = {
            sweep_param: exp[sweep_param],
            'exp_name': exp['path'].name,
            **filters,  # Include fixed params for reference
            **agg  # Include all aggregated metrics
        }

        results.append(row)

    df = pd.DataFrame(results)

    # Sort by sweep parameter
    if not df.empty:
        df = df.sort_values(sweep_param)

    return df


def plot_round_comparison(all_experiments, filters, sweep_param,
                               metrics_to_plot=None, save_path=None):
    """
    Plot metric evolution over server rounds with multiple experiments overlaid.

    Args:
        metrics_to_plot: List of column names from train_agg_metrics.csv
                        (e.g., ['train_loss', 'eval_acc', 'eval_loss'])
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['train_loss', 'eval_loss', 'eval_acc']

    # Filter experiments
    filtered = [
        exp for exp in all_experiments
        if all(exp.get(k) == v for k, v in filters.items())
    ]

    if not filtered:
        print(f"No experiments match filters: {filters}")
        return None

    # Load time-series data
    timeseries_data = []
    for exp in filtered:
        metrics = load(exp)
        if 'server_agg_metric' not in metrics:
            continue
        df = metrics['server_agg_metric'].copy()
        df[sweep_param] = exp[sweep_param]  # Add sweep value as column
        timeseries_data.append(df)

    if not timeseries_data:
        print("No time-series data found")
        return None

    combined_df = pd.concat(timeseries_data, ignore_index=True)

    # Plot
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        for sweep_val in sorted(combined_df[sweep_param].unique()):
            subset = combined_df[combined_df[sweep_param] == sweep_val]
            ax.plot(subset['server_round'], subset[metric], label=f"{sweep_param}={sweep_val}", linewidth=2)

        ax.set_xlabel('Server Round', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f"{metric.replace('_', ' ').title()} - {filters}", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_timeseries_comparison(all_experiments, filters, sweep_param,
                               metrics_to_plot=None, save_path=None):
    """
    Plot metric evolution over normalized timestamp with multiple experiments overlaid.

    Normalizes each experiment's timestamp to start at 0 for direct comparison
    of round durations across different configurations.

    Args:
        all_experiments: List of experiment dicts from parse_experiment_name
        filters: Dict of fixed parameters (e.g., {'bandwidth': '100MHz', 'nodes': '6N'})
        sweep_param: Parameter to vary (e.g., 'tdd', 'rank')
        metrics_to_plot: List of column names from train_agg_metrics.csv
                        (default: ['train_loss', 'eval_loss', 'eval_acc'])
        save_path: Path object or None
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['train_loss', 'eval_loss', 'eval_acc']

    # Filter experiments matching fixed parameters
    filtered = [
        exp for exp in all_experiments
        if all(exp.get(k) == v for k, v in filters.items())
    ]

    if not filtered:
        print(f"No experiments match filters: {filters}")
        return None

    print(f"Found {len(filtered)} experiments matching filters")

    # Load time-series data for each experiment
    timeseries_data = []
    for exp in filtered:
        metrics = load(exp)
        if 'server_agg_metric' not in metrics:
            print(f"  Skipping {exp['path'].name} - no training metrics")
            continue

        df = metrics['server_agg_metric'].copy()

        # Check for timestamp column
        if 'timestamp' not in df.columns:
            print(f"  ERROR: {exp['path'].name} missing 'timestamp' column")
            print(f"  Available columns: {df.columns.tolist()}")
            continue

        # Normalize timestamp to start at 0
        df['elapsed_time'] = df['timestamp'] - df['timestamp'].iloc[0]
        df[sweep_param] = exp[sweep_param]

        timeseries_data.append(df)
        print(f"  Loaded {exp['path'].name}: {len(df)} rounds, "
              f"duration={df['elapsed_time'].iloc[-1]:.1f}s")

    if not timeseries_data:
        print("No valid time-series data found")
        return None

    combined_df = pd.concat(timeseries_data, ignore_index=True)

    # Create subplots
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for ax, metric in zip(axes, metrics_to_plot):
        if metric not in combined_df.columns:
            print(f"  Warning: {metric} not found in data")
            ax.text(0.5, 0.5, f"Metric '{metric}' not found",
                    ha='center', va='center', transform=ax.transAxes)
            continue

        for sweep_val in sorted(combined_df[sweep_param].unique()):
            subset = combined_df[combined_df[sweep_param] == sweep_val]
            ax.plot(subset['elapsed_time'], subset[metric], label=f"{sweep_param}={sweep_val}",
                    linewidth=2, markersize=6, alpha=0.8)

        ax.set_xlabel('Elapsed Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f"{metric.replace('_', ' ').title()} - {', '.join(f'{k}={v}' for k, v in filters.items())}",
                     fontsize=12)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig

def plot_comparison(comparison_df, sweep_param, metrics_to_plot=None,
                    title_prefix="", save_path=None):
    """
    Plot multiple metrics across sweep parameter values.

    Args:
        comparison_df: DataFrame from compare() function
        sweep_param: Name of parameter being swept (e.g., 'tdd', 'rank')
        metrics_to_plot: List of metric names to plot. If None, plots common metrics
        title_prefix: Prefix for plot titles (e.g., "6N, 100MHz, MIMO")
        save_path: If provided, save figure to this path

    Returns:
        Figure object
    """
    # Default metrics if not specified
    if metrics_to_plot is None:
        metrics_to_plot = [
            'downlink_mean', 'uplink_mean',
            'final_eval_acc', 'final_train_loss',
            'mean_train_time', 'execution_time'
        ]

    # Filter to metrics that actually exist in the dataframe
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

    if not available_metrics:
        print("No metrics found in comparison DataFrame")
        return None

    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    # Plot each metric
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]

        # Plot with markers
        ax.plot(comparison_df[sweep_param], comparison_df[metric],
                marker='o', linewidth=2, markersize=8, label=metric)

        # Formatting
        ax.set_xlabel(sweep_param.upper(), fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f"{title_prefix}\n{metric.replace('_', ' ').title()}",
                     fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Rotate x-labels if needed
        if comparison_df[sweep_param].dtype == 'object':
            ax.tick_params(axis='x', rotation=45)

    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig

def plot_detailed_comparison(all_experiments, filters, sweep_param,
                             metrics_to_plot=None, save_path=None):
    """
    Create detailed comparison plots with multiple curves.

    This version creates one plot per metric, with sweep parameter on x-axis
    and potentially multiple curves if there are repeated experiments.

    Args:
        all_experiments: List of all experiment dicts
        filters: Dict of fixed parameters
        sweep_param: Parameter to sweep
        metrics_to_plot: List of metrics to plot
        save_path: Path to save figure
    """
    # Get comparison data
    comparison_df = compare(all_experiments, filters, sweep_param)

    if comparison_df.empty:
        print("No data to plot")
        return None

    # Create title from filters
    title_prefix = ", ".join([f"{k}={v}" for k, v in filters.items()])

    # Plot
    fig = plot_comparison(comparison_df, sweep_param, metrics_to_plot,
                          title_prefix, save_path)

    return fig

if __name__ == '__main__':
    directory = Path("/Users/roberthayek/hayekr@ieee.org - Google Drive/My Drive/5G Experiment Data/IMC_Analysis")
    all_experiments = []
    for exp in directory.iterdir():
        if not exp.is_dir():
            continue
        params = parse_experiment_name(exp.name)
        all_experiments.append({'path': exp, **params})

    print(f"Loaded {len(all_experiments)} experiments\n")

    # Test the comparison
    plot_detailed_comparison(
        all_experiments,
        filters={'nodes': '6N', 'rank': '1x1', 'tdd': '2-7'},
        sweep_param='bandwidth',
        save_path=Path.cwd()
    )