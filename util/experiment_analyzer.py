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
        df.to_csv(filepath, index=False)
        print(f"Saved {filename}")


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


def _aggregate_metrics(metrics_dict: dict, output_path: str, name: str) -> pd.DataFrame:
    """Aggregate metrics across trials by computing the mean for each round."""
    df = pd.concat(metrics_dict.values(), axis=1).T.groupby(level=0).mean().T
    df.to_csv(join(output_path, f'{name}_aggregated.csv'), index=False)
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
    pd.concat(agg_dfs, axis=1).to_csv(join(output_path, f'avg_{name}.csv'), index=False)  # Concat and save as csv
    return agg_dfs


def plot_ml_metric(data: pd.DataFrame, fig, label: str, color) -> None:
    plt.figure(fig)
    ax = plt.gca()

    """Plot metrics with confidence bands."""
    # Calculate statistics
    mean = data.mean(axis=1)
    std = data.std(axis=1)

    # Create confidence bands
    lower = mean - std
    upper = mean + std

    # Create the plot
    sns.lineplot(data=mean, label=label, color=color, ax=ax)
    if ML_CONFIDENCE_BANDS:
        plt.fill_between(mean.index, lower, upper, alpha=0.3, color=color, label='±1 std')

    plt.legend()


def format_save_ml_plots(output_dir: str, name: str, fig) -> None:
    plt.figure(fig)
    ax = plt.gca()
    
    plt.title(name)
    plt.xlabel('Round')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}'))
    # plt.show()
    plt.close()


def save_latency_histograms(output_dir, fig):
    plt.figure(fig)
    # ax = plt.gca()

    plt.savefig(join(output_dir, 'latency_histograms'))
    # plt.show()
    plt.close()


def plot_latency_statistics(uplink, downlink, output_dir):
    """
    Create violin plots for the latencies captured by flower.
    """
    plot_data = []

    for configuration in uplink:
        plot_data.extend([{
            'Configuration': configuration,
            'Direction': 'Downlink',
            'Latency (ms)': value
        } for value in downlink[configuration]])
        plot_data.extend([{
            'Configuration': configuration,
            'Direction': 'Uplink',
            'Latency (ms)': value
        } for value in uplink[configuration]])

    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 6))
    sns.violinplot(data=plot_df, x='Configuration', y='Latency (ms)',
                   hue='Direction', split=True, inner='quartile')

    plt.title('Latency Distribution by Configuration')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_distribution'))
    # plt.show()
    plt.close()

    # Split
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    sns.violinplot(data=plot_df[plot_df['Direction'] == 'Downlink'],
                   x='Configuration', y='Latency (ms)', inner='quartile', ax=ax1)
    ax1.set_title('Downlink Latency Distribution')
    ax1.grid(True, alpha=0.3)

    sns.violinplot(data=plot_df[plot_df['Direction'] == 'Uplink'],
                   x='Configuration', y='Latency (ms)', inner='quartile', ax=ax2)
    ax2.set_title('Uplink Latency Distribution')
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.suptitle('Latency Distribution by Configuration')
    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_distribution_split'))
    # plt.show()
    plt.close()


def plot_latency_histograms(uplink, downlink, fig, row_idx, trial_info):
    plt.figure(fig)
    allaxes = fig.get_axes()

    # Downlink histogram
    ax = allaxes[2*row_idx]
    ax.hist(downlink, alpha=0.75, color='blue')
    ax.set_title(f'{trial_info} - Downlink Latency')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Uplink histogram
    ax = allaxes[2*row_idx+1]
    ax.hist(uplink, alpha=0.75, color='green')
    ax.set_title(f'{trial_info} - Uplink Latency')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_time_histograms(data: pd.DataFrame, output_dir: str, name: str = 'Time'):
    """Create histogram subplots for each CID's various time measurements"""
    num_configurations = len(data.columns)
    fig, axs = plt.subplots(num_configurations, 1, figsize=(10, 4 * num_configurations))

    # Handle single column case
    if num_configurations == 1:
        axs = [axs]

    # Main Histogram
    for idx, col in enumerate(data.columns):
        axs[idx].hist(data[col], alpha=0.75, color='blue')
        axs[idx].set_title(f'{col.replace("_", " ")} Distribution')
        axs[idx].set_xlabel('Time (s)')
        axs[idx].set_ylabel('Frequency')
        axs[idx].grid(True, alpha=0.3)
    plt.suptitle(f'{name} Distribution by Configuration')
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_histograms'))
    # plt.show()
    plt.close()

    # Violin plots
    plt.figure(figsize=(10, 6))
    plot_data = pd.melt(data, var_name='Configuration', value_name='Time (s)')
    sns.violinplot(data=plot_data, x='Configuration', y='Time (s)', inner='quartile')
    plt.title(f'{name} Distribution by Configuration')
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


def plot_latency_statistics_by_cid(agg_latencies, output_dir):
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


def plot_latency_histograms_by_cid(agg_latencies, output_dir):
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


def retrieveTrialMetrics(trial_path, trial_dir, configuration_path):
    # Convert individual metrics to csv
    if IMPORT_ALL_TRIAL_DATA:
        individual_metrics = _read_json(join(trial_path, 'individual_metrics.json'))
        export_metrics_to_csv(individual_metrics, trial_path)

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

    agg_latencies = dict(sorted(agg_latencies.items(), key=lambda x: int(x[0].split('_')[1]))) # Sort the dict by CID
    pd.concat(agg_latencies, axis=1).to_csv(join(configuration_path, f'latencies_aggregated.csv'), index=False)  # Concat and save as csv

    # Training Time Metrics
    agg_training_times = _aggregate_metrics(training_times, configuration_path, name='training_time')
    agg_train_test_times = _aggregate_metrics(train_test_times, configuration_path, name='train_test_time')
    agg_val_test_times = _aggregate_metrics(val_test_times, configuration_path, name='val_test_time')

    # ML Metrics
    avg_val_accuracies = _aggregate_ml_metrics(val_accuracies, configuration_path, name='val_acc')
    avg_train_accuracies = _aggregate_ml_metrics(train_accuracies, configuration_path, name='train_acc')
    avg_val_losses = _aggregate_ml_metrics(val_losses, configuration_path, name='val_loss')
    avg_train_losses = _aggregate_ml_metrics(train_losses, configuration_path, name='train_loss')

    if PLOT_TRIAL_DATA:
        plot_latency_histograms_by_cid(agg_latencies, configuration_path)
        plot_latency_statistics_by_cid(agg_latencies, configuration_path)
        plot_time_histograms(agg_training_times, configuration_path, name='Training Time')
        plot_time_histograms(agg_train_test_times, configuration_path, name='Train Test Time')
        plot_time_histograms(agg_val_test_times, configuration_path, name='Validation Test Time')
        
        accuracy_fig = plt.figure(figsize=(10, 6))
        loss_fig = plt.figure(figsize=(10, 6))

        colors = sns.color_palette()

        plot_ml_metric(avg_train_accuracies, accuracy_fig, 'Train', colors[0])
        plot_ml_metric(avg_val_accuracies, accuracy_fig, 'Validation', colors[1])
        plot_ml_metric(avg_train_losses, loss_fig, 'Train', colors[0])
        plot_ml_metric(avg_val_losses, loss_fig, 'Validation', colors[1])

        format_save_ml_plots(configuration_path, 'Accuracy', accuracy_fig)
        format_save_ml_plots(configuration_path, 'Loss', loss_fig)


def retrieveConfigurationMetrics(input_path, experiment_dir):
    for configuration_dir in experiment_dir:
        configuration_path = join(input_path, configuration_dir)
        trial_dirs = [d for d in listdir(configuration_path) if isdir(join(configuration_path, d))]
        for trial_dir in trial_dirs:
            trial_path = join(configuration_path, trial_dir)
            try:
                retrieveTrialMetrics(trial_path, trial_dir, configuration_path)
            except Exception as e:
                print(f"Error processing {trial_dir}: {e}")


def plot_configurations(input_path, experiment_dir):
    time_metrics = ['Training Time', 'Train Test Time', 'Val Test Time']
    data = {}

    for result in experiment_dir:
        result_path = join(input_path, result)
        result = result.replace("_", " ")

        try:
            data[result] = {}
            
            latency = pd.read_csv(join(result_path, 'latencies_aggregated.csv'), skiprows=1)
            data[result]['Uplink'] = pd.concat([latency[df] for df in latency if 'Uplink' in df], axis=1).mean(axis=1)
            data[result]['Uplink'].name = result
            data[result]['Downlink'] = pd.concat([latency[df] for df in latency if 'Downlink' in df], axis=1).mean(axis=1)
            data[result]['Downlink'].name = result

            for metric in time_metrics:
                filename = f'{metric.lower().replace(" ", "_")}_aggregated.csv'
                data[result][metric] = pd.read_csv(join(result_path, filename)).mean(axis=1)
                data[result][metric].name = result
            
            data[result]['Validation Accuracy'] = pd.read_csv(join(result_path, 'avg_val_acc.csv'))
            data[result]['Validation Loss'] = pd.read_csv(join(result_path, 'avg_val_loss.csv'))
        
        except Exception as e:
            data.popitem()
            print(f"Error processing {result}: {e}")


    # assume all data should be plotted together for now (no separation of IID/Dirichlet datasets)
    plots = {}
    num_configurations = len(data)

    plots['Latency Hist'], axs = plt.subplots(num_configurations, 2, figsize=(12, 4 * num_configurations))
    val_acc_fig = plt.figure(figsize=(10, 6))
    val_loss_fig = plt.figure(figsize=(10, 6))

    idx = 0
    colors = sns.color_palette()
    for configuration in data:
        plot_latency_histograms(data[configuration]['Uplink'], data[configuration]['Downlink'], plots['Latency Hist'], idx, configuration)

        plot_ml_metric(data[configuration]['Validation Accuracy'], val_acc_fig, configuration, colors[idx])
        plot_ml_metric(data[configuration]['Validation Loss'], val_loss_fig, configuration, colors[idx])
        idx += 1
    
    save_latency_histograms(input_path, plots['Latency Hist'])

    uplink = pd.concat([data[configuration]['Uplink'] for configuration in data], axis=1)
    downlink = pd.concat([data[configuration]['Downlink'] for configuration in data], axis=1)
    plot_latency_statistics(uplink, downlink, input_path)
    
    format_save_ml_plots(input_dir, 'Validation Accuracy', val_acc_fig)
    format_save_ml_plots(input_dir, 'Validation Loss', val_loss_fig)
    configuration = None
    for metric in time_metrics:
        try:
            data_concat = pd.concat([data[configuration][metric] for configuration in data], axis=1)
            plot_time_histograms(data_concat, input_path, metric)
        except Exception as e:
            print(f"Error processing {metric} for {configuration}: {e}")


def main(input_path: str) -> None:
    experiment_dir = [d for d in listdir(input_path) if isdir(join(input_path, d))]

    if AGGREGATE_TRIAL_DATA:
        retrieveConfigurationMetrics(input_path, experiment_dir)

    if PLOT_EXPERIMENT_DATA:
        plot_configurations(input_path, experiment_dir)


if __name__ == '__main__':
    ML_CONFIDENCE_BANDS = False
    
    PLOT_EXPERIMENT_DATA = True # whether we should plot overview figures of all configurations
    AGGREGATE_TRIAL_DATA = True # whether we should go into each configuration (ex. 3Node_Ethernet_IID) and aggregate data from all trials (directories named with run IDs)
    PLOT_TRIAL_DATA = False # whether we should go into each configuration (ex. 3Node_Ethernet_IID) and plot time, latency data by CID; ML training and validation data. sub-condition of AGGREGATE_ALL_TRIAL_DATA
    IMPORT_ALL_TRIAL_DATA = True # whether we should go into each trial (within in a configuration) and export the json data to CSV files. sub-condition of AGGREGATE_ALL_TRIAL_DATA
    
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)