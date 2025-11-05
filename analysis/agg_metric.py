import csv
import pandas as pd
import re
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

def analyze_model(model_path):
    model = torch.load(model_path)

def load(experiment_path):
    """
    Load a single experiment return metrics
    Args:
        experiment_path:

    Returns:

    """

    metrics = {}


    # Get latencies from a single experiment
    latencies =  list(experiment_path['path'].glob('latency_*_CID*.csv'))
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


    """# Load the model
    model_file = experiment_path['path'] / 'final_model.pt'
    if model_file.exists():
        metrics['model_metric'] = analyze_model(model_file)"""

    return metrics




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


def plot_latency_over_rounds(experiment, save_path=None):
    """
    Plot how latency evolves over training rounds for a single experiment.

    Args:
        experiment: Single experiment dict
        save_path: Path to save figure
    """
    metrics = load(experiment)

    if 'latency' not in metrics or 'server_agg_metric' not in metrics:
        print("Missing required metrics")
        return None

    # Aggregate latency per round
    latency_per_round = metrics['latency'].groupby('server_round').agg({
        'downlink_latency': ['mean', 'std'],
        'uplink_latency': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    latency_per_round.columns = ['_'.join(col).strip('_') for col in latency_per_round.columns]

    # Merge with training metrics
    train_df = metrics['server_agg_metric']
    combined = train_df.merge(latency_per_round,
                              left_on='server_round',
                              right_on='server_round',
                              how='left')

    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy over rounds
    ax = axes[0, 0]
    ax.plot(combined['server_round'], combined['eval_acc'],
            'b-', linewidth=2, label='Eval Accuracy')
    ax.set_xlabel('Server Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Evaluation Accuracy Over Rounds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Downlink latency over rounds
    ax = axes[0, 1]
    ax.plot(combined['server_round'], combined['downlink_latency_mean'],
            'r-', linewidth=2, label='Mean')
    ax.fill_between(combined['server_round'],
                    combined['downlink_latency_mean'] - combined['downlink_latency_std'],
                    combined['downlink_latency_mean'] + combined['downlink_latency_std'],
                    alpha=0.3, color='r', label='±1 std')
    ax.set_xlabel('Server Round', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Downlink Latency Over Rounds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 3: Uplink latency over rounds
    ax = axes[1, 0]
    ax.plot(combined['server_round'], combined['uplink_latency_mean'],
            'g-', linewidth=2, label='Mean')
    ax.fill_between(combined['server_round'],
                    combined['uplink_latency_mean'] - combined['uplink_latency_std'],
                    combined['uplink_latency_mean'] + combined['uplink_latency_std'],
                    alpha=0.3, color='g', label='±1 std')
    ax.set_xlabel('Server Round', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Uplink Latency Over Rounds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 4: Loss over rounds
    ax = axes[1, 1]
    ax.plot(combined['server_round'], combined['train_loss'],
            'orange', linewidth=2, label='Train Loss')
    ax.plot(combined['server_round'], combined['eval_loss'],
            'purple', linewidth=2, label='Eval Loss')
    ax.set_xlabel('Server Round', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Eval Loss Over Rounds', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add experiment info as suptitle
    exp_info = f"{experiment['nodes']}, {experiment['bandwidth']}, {experiment['tdd']}, {experiment['rank']}"
    fig.suptitle(f"Experiment: {exp_info}", fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_multi_experiment_comparison(all_experiments, filters, sweep_param,
                                     metric='final_eval_acc', save_path=None):
    """
    Create a focused plot for comparing one metric across sweep values.
    Useful for presentations/papers.

    Args:
        all_experiments: List of all experiment dicts
        filters: Dict of fixed parameters
        sweep_param: Parameter to sweep
        metric: Single metric to plot
        save_path: Path to save figure
    """
    comparison_df = compare(all_experiments, filters, sweep_param)

    if comparison_df.empty:
        print("No data to plot")
        return None

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot with error bars if we have std
    if f"{metric}_std" in comparison_df.columns:
        ax.errorbar(comparison_df[sweep_param], comparison_df[metric],
                    yerr=comparison_df[f"{metric}_std"],
                    marker='o', markersize=10, linewidth=2.5,
                    capsize=5, capthick=2, label=metric)
    else:
        ax.plot(comparison_df[sweep_param], comparison_df[metric],
                marker='o', markersize=10, linewidth=2.5, label=metric)

    # Formatting
    title_parts = [f"{k}={v}" for k, v in filters.items()]
    ax.set_title(", ".join(title_parts), fontsize=14, fontweight='bold')
    ax.set_xlabel(sweep_param.upper(), fontsize=13, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    # Rotate x-labels if categorical
    if comparison_df[sweep_param].dtype == 'object':
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_heatmap_comparison(all_experiments, param1, param2, metric='final_eval_acc',
                            filters=None, save_path=None):
    """
    Create a heatmap comparing a metric across two swept parameters.

    Args:
        all_experiments: List of all experiment dicts
        param1: First parameter for x-axis
        param2: Second parameter for y-axis
        metric: Metric to show in heatmap
        filters: Additional fixed parameters
        save_path: Path to save figure
    """
    if filters is None:
        filters = {}

    # Filter experiments
    filtered = [
        exp for exp in all_experiments
        if all(exp.get(k) == v for k, v in filters.items())
    ]

    # Load all and build matrix
    data = []
    for exp in filtered:
        metrics = load(exp)
        agg = aggregate_metrics(metrics)

        if metric in agg:
            data.append({
                param1: exp[param1],
                param2: exp[param2],
                metric: agg[metric]
            })

    if not data:
        print("No data for heatmap")
        return None

    df = pd.DataFrame(data)
    pivot = df.pivot(index=param2, columns=param1, values=metric)

    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                ax=ax, cbar_kws={'label': metric.replace('_', ' ').title()})

    title_parts = [f"{k}={v}" for k, v in filters.items()]
    ax.set_title(f"{metric.replace('_', ' ').title()}\n" + ", ".join(title_parts),
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(param1.upper(), fontsize=13, fontweight='bold')
    ax.set_ylabel(param2.upper(), fontsize=13, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig






if __name__ == '__main__':
    directory = Path("/Users/roberthayek/hayekr@ieee.org - Google Drive/My Drive/5G Experiment Data/IMC_Analysis")
    all_experiments = []
    for exp in directory.iterdir():
        if not exp.is_dir():
            continue
        params = parse_experiment_name(exp.name)
        all_experiments.append({'path': exp, **params})

   """ # 6N, 40Mhz, MIMO, sweep TDD
    target_experiments = [
        exp for exp in all_experiments
        if exp['rank'] == '2x2' and exp['bandwidth'] == '100MHz' and exp['nodes'] == '6N'
    ]

    compare(all_experiments, filters={'bandwidth': '100MHz', 'nodes': '6N', 'rank': '2x2'}, sweep_param='tdd')"""

# ========================================
    # PLOT 1: TDD sweep for 6N, 100MHz, MIMO
    # ========================================
    print("Creating TDD sweep plots...")
    fig1 = plot_detailed_comparison(
        all_experiments,
        filters={'bandwidth': '100MHz', 'nodes': '6N', 'rank': '2x2'},
        sweep_param='tdd',
        metrics_to_plot=['downlink_mean', 'uplink_mean', 'final_eval_acc',
                        'final_train_loss', 'mean_train_time', 'execution_time'],
        save_path='plots/tdd_sweep_6N_100MHz_MIMO.png'
    )




