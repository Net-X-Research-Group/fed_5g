import json
import re
from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })


def _read_json(file):
    with open(file, 'r') as f:
        raw = json.load(f)
    return raw


def _aggregate_metrics(metrics_dict: dict, output_path: str, name: str) -> pd.DataFrame:
    """Aggregate metrics across trials by computing the mean for each round."""
    df = pd.concat(metrics_dict.values(), axis=1).T.groupby(level=0).mean().T.drop('Round', axis=1)
    df.to_csv(join(output_path, f'{name}_aggregated.csv'))
    return df


def _aggregate_ml_metrics(metrics_dict: dict, output_path: str, name: str) -> list:
    """Aggregate metrics across trials by computing the mean for each round."""
    dfs = list(metrics_dict.values())
    agg_dfs = []
    for df in dfs:
        try:
            df = df.drop('Round', axis=1)
        except KeyError:
            pass
        df['avg'] = df.mean(axis=1)
        # Drop all but 'avg'
        agg_dfs.append(df[['avg']])
    pd.concat(agg_dfs, axis=1).to_csv(join(output_path, f'avg_{name}.csv'))  # Concat and save as csv
    return agg_dfs


def process_trial_latency(path: str) -> dict:
    """
    Process latency data captured by patched flower version.
    """
    files = [f for f in listdir(path) if f.startswith('latency')]
    results = {}
    for file in files:
        cid = re.search(r'latency_\d+_CID(\d+)', file)
        if not cid:
            continue
        cid = cid.group(1)
        results[f'CID_{cid}'] = pd.read_csv(join(path, file), names=['Round', 'Downlink', 'Uplink'])
    return results


def export_metrics_to_csv(data, output_dir):
    """
    Export each metric to a separate CSV file where columns are CIDs and rows are rounds.

    Args:
        data (dict): Nested dictionary with structure {Round#: {CID: {metrics}}}
        output_dir (str): Directory to save the CSV files
    """
    # Get all metrics from the first round and first CID
    first_round = list(data.keys())[0]
    first_cid = list(data[first_round].keys())[0]
    metrics = list(data[first_round][first_cid].keys())

    # Get all unique CIDs
    cids = set()
    for round_data in data.values():
        cids.update(round_data.keys())
    cids = sorted(list(cids))

    # Get all rounds
    rounds = sorted(list(data.keys()), key=int)

    # For each metric, create a DataFrame and save to CSV
    for metric in metrics:
        # Create empty DataFrame with rounds as index
        df = pd.DataFrame(index=rounds)

        # Fill in data for each CID
        for cid in cids:
            cid_values = []
            for round_num in rounds:
                if cid in data[round_num]:
                    cid_values.append(data[round_num][cid][metric])
                else:
                    cid_values.append(None)  # Handle missing data
            df[f'CID_{cid}'] = cid_values
        df.index.name = 'Round'
        # Save to CSV
        filename = f"{metric}.csv"
        filepath = join(output_dir, filename)
        df.to_csv(filepath)
        print(f"Saved {filename}")


def plot_latency_statistics(agg_latencies, output_dir):
    """
    Create violin plots for the latencies captured by flower.
    """
    plot_data = []
    for cid, df in agg_latencies.items():
        plot_data.extend([{
            'Client': cid,
            'Direction': 'Downlink',
            'Latency (ms)': value
        } for value in df['Downlink']])
        plot_data.extend([{
            'Client': cid,
            'Direction': 'Uplink',
            'Latency (ms)': value
        } for value in df['Uplink']])

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=plot_df, x='Client', y='Latency (ms)',
                   hue='Direction', split=True, inner='quartile')

    plt.title('Latency Distribution by Client')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_distribution'))
    # plt.show()
    plt.close()

    # Split
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    sns.violinplot(data=plot_df[plot_df['Direction'] == 'Downlink'],
                   x='Client', y='Latency (ms)', inner='quartile', ax=ax1)
    ax1.set_title('Downlink Latency Distribution')
    ax1.grid(True, alpha=0.3)

    sns.violinplot(data=plot_df[plot_df['Direction'] == 'Uplink'],
                   x='Client', y='Latency (ms)', inner='quartile', ax=ax2)
    ax2.set_title('Uplink Latency Distribution')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.suptitle('Latency Distribution by Client')
    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_distribution_split'))
    # plt.show()
    plt.close()


def plot_latency_histograms(agg_latencies, output_dir):
    """Create histogram subplots for each CID's latencies"""
    num_cids = len(agg_latencies)
    fig, axs = plt.subplots(num_cids, 2, figsize=(12, 4 * num_cids))

    for idx, (cid, df) in enumerate(agg_latencies.items()):
        # Downlink histogram
        axs[idx, 0].hist(df['Downlink'], alpha=0.75, color='blue')
        axs[idx, 0].set_title(f'{cid.replace("_", " ")} - Downlink Latency')
        axs[idx, 0].set_xlabel('Latency (ms)')
        axs[idx, 0].set_ylabel('Frequency')
        axs[idx, 0].grid(True, alpha=0.3)

        # Uplink histogram
        axs[idx, 1].hist(df['Uplink'], alpha=0.75, color='green')
        axs[idx, 1].set_title(f'{cid.replace("_", " ")} - Uplink Latency')
        axs[idx, 1].set_xlabel('Latency (ms)')
        axs[idx, 1].set_ylabel('Frequency')
        axs[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_histograms'))
    # plt.show()
    plt.close()


def plot_time_histograms(data: pd.DataFrame, output_dir: str, name: str = 'Time'):
    """Create histogram subplots for each CID's various time measurements"""
    num_cids = len(data.columns)
    fig, axs = plt.subplots(num_cids, 1, figsize=(10, 4 * num_cids))

    # Handle single column case
    if num_cids == 1:
        axs = [axs]

    # Main Histogram
    for idx, col in enumerate(data.columns):
        axs[idx].hist(data[col], alpha=0.75, color='blue')
        axs[idx].set_title(f'{col.replace("_", " ")} Distribution')
        axs[idx].set_xlabel('Time (s)')
        axs[idx].set_ylabel('Frequency')
        axs[idx].grid(True, alpha=0.3)
    plt.suptitle(f'{name} Distribution by Client')
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_histograms'))
    # plt.show()
    plt.close()

    # Violin plots
    plt.figure(figsize=(10, 6))
    plot_data = pd.melt(data, var_name='Client', value_name='Time (s)')
    sns.violinplot(data=plot_data, x='Client', y='Time (s)', inner='quartile')
    plt.title(f'{name} Distribution by Client')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_violin'))
    # plt.show()
    plt.close()

    # Overlay Histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in data.columns:
        ax.hist(data[col], alpha=0.75, label=col.replace("_", " "))
    ax.set_title(f'{name} Distribution')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_overlay_histogram'))
    # plt.show()
    plt.close()


def plot_ml_metric(data: pd.DataFrame, output_dir: str, name: str) -> None:
    num_cids = len(data.columns)

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in data.columns:
        ax.plot(data[col], alpha=0.75, label=col.replace("_", " "))
    ax.set_title(f'{name}')
    ax.set_xlabel('Round')
    # ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_plot'))
    # plt.show()
    plt.close()


def plot_ml_metrics(train_data: list, validation_data: list, output_dir: str, name: str) -> None:
    """Plot training and validation metrics with confidence bands."""
    # Stack all trials into DataFrames
    all_train_trials = pd.concat([df['avg'] for df in train_data], axis=1)
    all_val_trials = pd.concat([df['avg'] for df in validation_data], axis=1)

    # Calculate statistics
    train_mean = all_train_trials.mean(axis=1)
    train_std = all_train_trials.std(axis=1)
    val_mean = all_val_trials.mean(axis=1)
    val_std = all_val_trials.std(axis=1)

    # Create confidence bands
    train_lower = train_mean - train_std
    train_upper = train_mean + train_std
    val_lower = val_mean - val_std
    val_upper = val_mean + val_std

    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=train_mean, label='Train', color='blue')
    plt.fill_between(train_mean.index, train_lower, train_upper, alpha=0.3, color='blue', label='Train ±1 std')

    sns.lineplot(data=val_mean, label='Validation', color='red')
    plt.fill_between(val_mean.index, val_lower, val_upper, alpha=0.3, color='red', label='Validation ±1 std')

    plt.title(name)
    plt.xlabel('Round')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}'))
    # plt.show()
    plt.close()


def main(input_path: str) -> None:
    trial_dirs = [d for d in listdir(input_path) if isdir(join(input_path, d))]

    # Latency
    latencies = {}
    agg_latencies = {}

    # Times
    training_times = {}
    train_test_times = {}
    val_test_times = {}

    # ML Metrics
    val_accuracies = {}
    val_losses = {}
    train_accuracies = {}
    train_losses = {}

    for trial_dir in trial_dirs:
        trial_path = join(input_path, trial_dir)

        # Convert individual metrics to csv
        individual_metrics = _read_json(join(trial_path, 'individual_metrics.json'))
        export_metrics_to_csv(individual_metrics, trial_path)

        # Process latency data captured by Flower
        try:
            trial_results = process_trial_latency(trial_path)
            for cid, df in trial_results.items():
                if cid not in latencies:
                    latencies[cid] = []
                latencies[cid].append(df)
        except Exception as e:
            print(f"Error processing {trial_dir}: {e}")

        # Process training time metrics, measured by flower
        try:
            training_times[trial_dir] = pd.read_csv(join(trial_path, 'training_time.csv'))
            train_test_times[trial_dir] = pd.read_csv(join(trial_path, 'train_test_time.csv'))
            val_test_times[trial_dir] = pd.read_csv(join(trial_path, 'val_test_time.csv'))
        except Exception as e:
            print(f"Error processing {trial_dir}: {e}")

        # Process ML Metrics
        try:
            val_accuracies[trial_dir] = pd.read_csv(join(trial_path, 'val_acc.csv'))
            val_losses[trial_dir] = pd.read_csv(join(trial_path, 'val_loss.csv'))
            train_losses[trial_dir] = pd.read_csv(join(trial_path, 'train_loss.csv'))
            train_accuracies[trial_dir] = pd.read_csv(join(trial_path, 'train_acc.csv'))
        except Exception as e:
            print(f"Error processing {trial_dir}: {e}")

    for cid, dfs in latencies.items():
        agg_df = pd.concat(dfs, axis=1).T.groupby(level=0).mean().T.drop('Round', axis=1)
        agg_latencies[cid] = agg_df
    plot_latency_histograms(agg_latencies, input_path)
    plot_latency_statistics(agg_latencies, input_path)

    # Training Time Metrics
    agg_training_times = _aggregate_metrics(training_times, input_path, name='training_time')
    agg_train_test_times = _aggregate_metrics(train_test_times, input_path, name='train_test_time')
    agg_val_test_times = _aggregate_metrics(val_test_times, input_path, name='val_test_time')

    plot_time_histograms(agg_training_times, input_path, name='Training Time')
    plot_time_histograms(agg_train_test_times, input_path, name='Train Test Time')
    plot_time_histograms(agg_val_test_times, input_path, name='Validation Test Time')

    # ML Metrics
    avg_val_accuracies = _aggregate_ml_metrics(val_accuracies, input_path, name='val_acc')
    avg_train_accuracies = _aggregate_ml_metrics(train_accuracies, input_path, name='train_acc')
    avg_val_losses = _aggregate_ml_metrics(val_losses, input_path, name='val_loss')
    avg_train_losses = _aggregate_ml_metrics(train_losses, input_path, name='train_loss')
    plot_ml_metrics(avg_train_accuracies, avg_val_accuracies, input_path, 'Accuracy')
    plot_ml_metrics(avg_train_losses, avg_val_losses, input_path, 'Loss')


if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)
