import csv
import pandas as pd
import re
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec


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


def plot_agg_metric(agg_metrics):
    """
    Create one figure per metric, with a line for each experiment.
    Time metrics get frequency plots instead.

    Args:
        agg_metrics: List of dicts with params and 'metrics' (DataFrame)
        column_name_map: Dict mapping column names to display names

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

    sns.set_style("whitegrid")

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
                label = f"TDD {exp['tdd']}"
                sns.histplot(df[metric], kde=False, label=label, alpha=0.5, ax=ax)

            ax.set_xlabel(display_name, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{display_name} Distribution', fontsize=14)
        else:
            # Line plot for other metrics
            for exp in agg_metrics:
                df = exp['metrics']
                label = f"TDD {exp['tdd']}"
                sns.lineplot(x=df['server_round'], y=df[metric],
                             label=label, linewidth=2, ax=ax)

            ax.set_xlabel('Server Round', fontsize=12)
            ax.set_ylabel(display_name, fontsize=12)
            ax.set_title(f'{display_name} vs Server Round', fontsize=14)

        ax.legend(title='Configuration')
        figs[metric] = fig

    plt.tight_layout()
    return figs

def plot_agg_metric_vs_time(agg_metrics):
    # Default mapping
    column_name_map = {
        'train_loss': 'Training Loss',
        'train_time': 'Training Time (s)',
        'eval_loss': 'Evaluation Loss',
        'eval_acc': 'Evaluation Accuracy',
        'eval_time': 'Evaluation Time (s)'
    }

    sns.set_style("whitegrid")

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

            label = f"TDD {exp['tdd']}"
            sns.lineplot(x=df['elapsed_time'], y=df[metric],
                         label=label, linewidth=2, ax=ax_main)

        # Distribution subplot: histogram of round durations
        for exp in agg_metrics:
            df = exp['metrics'].copy()
            df['round_duration'] = df['timestamp'].diff()

            label = f"TDD {exp['tdd']}"
            # Skip first round (NaN duration)
            #sns.histplot(df['round_duration'].dropna(), kde=True,
            #             label=label, alpha=0.6, bins=50, ax=ax_dist,
            #             stat='frequency', element='step')

            sns.kdeplot(df['round_duration'].dropna(),
                        label=label, linewidth=2, ax=ax_dist)

        ax_main.set_xlabel('')  # Remove x-label from main plot
        ax_main.set_ylabel(display_name, fontsize=12)
        ax_main.set_title(f'{display_name} vs Elapsed Time', fontsize=14)
        ax_main.legend(title='Configuration', loc='best')
        ax_main.grid(True, alpha=0.3)

        ax_dist.set_xlabel('Round Duration (s)', fontsize=10)
        ax_dist.set_ylabel('Frequency', fontsize=10)
        ax_dist.tick_params(labelsize=9)
        ax_dist.grid(True, alpha=0.3, axis='y')

        figs[metric] = fig

    plt.tight_layout()
    plt.show()
    return figs

def plot_individual(metrics):
    print(metrics)






def filter_metrics(experiment_paths: list, filters: dict):
    filtered = [exp for exp in experiment_paths
                if all(exp.get(k) == v for k, v in filters.items())]

    agg_metrics = []
    individual_metrics = []
    latency_metrics = []

    for exp in filtered:
        metrics = load(exp)
        agg_metrics.append({**exp, 'metrics': metrics['server_agg_metric'], 'execution_time': metrics['execution_time'],
                            'start_time': metrics['start_time']})
        individual_metrics.append(
            {**exp, 'metrics': metrics['individual_metrics'], 'execution_time': metrics['execution_time'],
             'start_time': metrics['start_time']})
        latency_metrics.append(
            {**exp, 'metrics': metrics['latency'], 'execution_time': metrics['execution_time'],
             'start_time': metrics['start_time']})

    # Plot agg metrics
    #plot_agg_metric(agg_metrics)
    #plot_agg_metric_vs_time(agg_metrics)

    # Plot individual metrics
    plot_individual(individual_metrics)

    # Plot latency metrics


if __name__ == '__main__':
    directory = Path("/Users/roberthayek/Documents/git_repos/fed_5g/IMC")
    all_experiments = []
    for exp in directory.iterdir():
        if not exp.is_dir():
            continue
        params = parse_experiment_name(exp.name)
        all_experiments.append({'path': exp, **params})

    filter_metrics(all_experiments, {'bandwidth': '100MHz', 'nodes': '6N', 'rank': '2x2'})
