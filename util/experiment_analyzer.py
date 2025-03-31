import json
import re
from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 18,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.format': 'svg',
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

    ''' Just repeat the code from the following for loop (for metric in metrics) to inelegantly calculate round time'''
    round_time_df = pd.DataFrame(index=rounds)
    for cid in cids:
        cid_values = []
        for round_num in rounds:
            if cid in data[round_num]:
                value = 0
                for metric in list(data[first_round][first_cid].keys()):
                    if 'time' in metric and 'start' not in metric and 'end' not in metric:
                        # cid_values.append(data[round_num][cid][metric])
                        value += data[round_num][cid][metric]
                cid_values.append(value)
            else:
                cid_values.append(None)  # Handle missing data
        round_time_df[f'CID_{cid}'] = cid_values
    round_time_df.index.name = 'Round'
    # Save to CSV
    filename = f"round_time.csv"
    filepath = join(output_dir, filename)
    round_time_df.to_csv(filepath, index=False)
    if VERBOSITY == 2:
        print(f"Saved {str(filepath)}")
    ''' End of repeated code section (repeats following code for one new metric)'''

    comm_round_time = round_time_df.max(axis=1, numeric_only=True)
    comm_round_time.to_csv(join(output_dir, 'comm_' + filename), index=False)
    # df = pd.concat(metrics_dict.values(), axis=1).T.groupby(level=0).mean().T #here
    # data[display_name][metric] = pd.read_csv(join(str(result_path), filename)).mean(axis=1)
    

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
        if VERBOSITY == 2:
            print(f"Saved {str(filepath)}")


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
    if VERBOSITY >= 1:
        print(f'Saved {output_path}/{name}_aggregated.csv')
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
    if VERBOSITY >= 1:
        print(f'Saved {output_path}/avg_{name}.csv')
    return agg_dfs


def _rename_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    changed_df = df.copy()
    changed_df.rename(columns={'Latency (s)': 'Time (s)'}, inplace=True)
    changed_df.loc[changed_df['Direction'] == 'Downlink', 'Direction'] = '$t_d$'
    changed_df.loc[changed_df['Direction'] == 'Uplink', 'Direction'] = '$t_u$'

    return changed_df


def plot_ml_metric(data, fig, label: str, color, round_time) -> None:
    plt.figure(fig)
    ax = plt.gca()

    """Plot metrics with confidence bands."""
    if isinstance(data, list):
        data = pd.concat(data, axis=1)

    # Calculate statistics
    mean = data.mean(axis=1)
    std = data.std(axis=1)

    # Create confidence bands
    lower = mean - std
    upper = mean + std

    # Create the plot
    if PLOT_ROUNDS_INSTEAD_OF_TIME:
        # Plot by round number
        x_values = list(range(len(mean)))
        y_values = mean.values.tolist()

        sns.lineplot(x=x_values, y=y_values, label=label, color=color, ax=ax)

        if ML_CONFIDENCE_BANDS:
            plt.fill_between(x_values,
                            lower.values.tolist(),
                            upper.values.tolist(),
                            alpha=0.3, color=color, label='±1 std')
    else:
        try:
            # First ensure we're working with pandas Series to use pandas methods
            if not isinstance(mean, pd.Series):
                mean = pd.Series(mean)
            if not isinstance(lower, pd.Series):
                lower = pd.Series(lower)
            if not isinstance(upper, pd.Series):
                upper = pd.Series(upper)

            # Ensure round_time is numeric
            if isinstance(round_time, pd.DataFrame):
                # If it's a DataFrame, take the mean of all columns
                round_time_numeric = round_time.mean(axis=1)
            else:
                # Otherwise, convert to Series if it's not already
                round_time_numeric = pd.Series(round_time) if not isinstance(round_time, pd.Series) else round_time

            # Calculate cumulative time manually
            round_time_values = round_time_numeric.values
            cumulative_time = np.zeros(len(round_time_values) + 1)
            for i in range(1, len(round_time_values) + 1):
                if np.isnan(round_time_values[i-1]):
                    cumulative_time[i] = cumulative_time[i-1]
                else:
                    cumulative_time[i] = cumulative_time[i-1] + round_time_values[i-1]

            # Ensure arrays are the right length
            time_x = cumulative_time[:len(mean)]

            # Convert everything to lists for plotting
            x_values = time_x.tolist()
            y_values = mean.values.tolist()
            lower_values = lower.values.tolist()
            upper_values = upper.values.tolist()

            # Plot with the time array
            sns.lineplot(x=x_values, y=y_values, label=label, color=color, ax=ax)
            if ML_CONFIDENCE_BANDS:
                plt.fill_between(x_values, lower_values, upper_values,
                                alpha=0.3, color=color, label='±1 std')

        except Exception as e:
            print(f"Error in plotting with time: {e}")
            # Fall back to basic plot by round number
            x_values = list(range(len(mean)))
            y_values = mean.values.tolist()
            sns.lineplot(x=x_values, y=y_values, label=label, color=color, ax=ax)

    plt.legend()


def format_save_ml_plots(output_dir: str, name: str, fig) -> None:
    plt.figure(fig)
    ax = plt.gca()

    #plt.title(name)
    if PLOT_ROUNDS_INSTEAD_OF_TIME:
        plt.xlabel('Round')
    else:
        plt.xlabel('Time (s)')
    plt.ylabel('')
    if 'accuracy' in name.lower():
        plt.ylabel('Accuracy')
    elif 'loss' in name.lower():
        plt.ylabel('Loss')
    else:
        plt.ylabel('')
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
            'Latency (s)': value
        } for value in downlink[configuration]])
        plot_data.extend([{
            'Configuration': configuration,
            'Direction': 'Uplink',
            'Latency (s)': value
        } for value in uplink[configuration]])

    plot_df = pd.DataFrame(plot_data)
    # Store the filtered values to examine them

    filtered_values = plot_df[plot_df['Latency (s)'] >= 30]

    # Count how many values are filtered out
    filtered_count = len(filtered_values)

    # Print the filtered values for inspection
    print("Filtered values (latencies ≥ 30s):")
    print(filtered_values)
    print(f"Total number of latencies filtered out: {filtered_count}")

    # Create the filtered dataframe
    filtered_plot_df = plot_df[plot_df['Latency (s)'] < 30].copy()
    filtered_plot_df['Latency (s)'] = filtered_plot_df['Latency (s)'] - 0.0573

    filtered_plot_df = _rename_for_latex(filtered_plot_df)  # Rename axis and legend

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=filtered_plot_df, x='Configuration', y='Time (s)',
                   hue='Direction')

    #plt.title('Latency Distribution by Configuration')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_distribution'))
    # plt.show()
    plt.close()

    # Split
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    sns.violinplot(data=filtered_plot_df[filtered_plot_df['Direction'] == '$t_d$'],
                   x='Configuration', y='Time (s)', inner='quartile', ax=ax1, density_norm="area", common_norm=True)
    #ax1.set_title('Downlink Latency Distribution')
    ax1.grid(True, alpha=0.3)
    sns.violinplot(data=filtered_plot_df[filtered_plot_df['Direction'] == '$t_u$'],
                   x='Configuration', y='Time (s)', inner='quartile', ax=ax2, density_norm="area", common_norm=True)
    #ax2.set_title('Uplink Latency Distribution')
    ax2.grid(True, alpha=0.3)
    #plt.suptitle('Latency Distribution by Configuration')
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
    #ax.set_title(f'{trial_info} - Downlink Latency')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Uplink histogram
    ax = allaxes[2*row_idx+1]
    ax.hist(uplink, alpha=0.75, color='green')
    #ax.set_title(f'{trial_info} - Uplink Latency')
    ax.set_xlabel('Time (ss)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


def plot_time_histograms(data: pd.DataFrame, output_dir: str, name: str = 'Time'):
    """Create histogram subplots for each CID's various time measurements"""
    num_configurations = len(data.columns)

    # Filter out extreme values (>=100s)
    data_filtered = data.copy()
    for col in data.columns:
        extreme_values = data[col][data[col] >= 100]
        if not extreme_values.empty:
            print(f"Filtered out {len(extreme_values)} extreme values (>=100s) from {col}")
            data_filtered[col] = data[col][data[col] < 100]
    data = data_filtered

    fig, axs = plt.subplots(num_configurations, 1, figsize=(10, 4 * num_configurations))

    # Handle single column case
    if num_configurations == 1:
        axs = [axs]

    # Main Histogram
    for idx, col in enumerate(data.columns):
        axs[idx].hist(data[col], alpha=0.75, color='blue')
        #axs[idx].set_title(f'{col.replace("_", " ")} Distribution')
        axs[idx].set_xlabel('Time (s)')
        axs[idx].set_ylabel('Frequency')
        axs[idx].grid(True, alpha=0.3)
    #plt.suptitle(f'{name} Distribution by Configuration')
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_histograms'))
    # plt.show()
    plt.close()

    # Violin plots
    plt.figure(figsize=(10, 6))
    plot_data = pd.melt(data, var_name='Configuration', value_name='Time (s)')
    sns.violinplot(data=plot_data, x='Configuration', y='Time (s)', inner='quartile', density_norm="area", common_norm=True)
    #plt.title(f'{name} Distribution by Configuration')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_violin'))
    # plt.show()
    plt.close()

    # Overlay Histograms
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in data.columns:
        ax.hist(data[col], alpha=0.75, label=col.replace("_", " "))
    #ax.set_title(f'{name} Distribution')
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
            'Latency (s)': value
        } for value in df['Downlink']])
        plot_data.extend([{
            'Client': cid,
            'Direction': 'Uplink',
            'Latency (s)': value
        } for value in df['Uplink']])

    plot_df = pd.DataFrame(plot_data)

    # Store the filtered values to examine them
    filtered_values = plot_df[plot_df['Latency (s)'] >= 30]

    # Count how many values are filtered out
    filtered_count = len(filtered_values)

    # Print the filtered values for inspection
    print("Filtered values (latencies ≥ 30s):")
    print(filtered_values)
    print(f"Total number of latencies filtered out: {filtered_count}")

    # Create the filtered dataframe
    plot_df = plot_df[plot_df['Latency (s)'] < 30]
    plot_df = _rename_for_latex(plot_df) # Rename axis and legend
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=plot_df, x='Client', y='Time (s)',
                   hue='Direction', split=True, inner='quartile', density_norm="area", common_norm=True)

    #plt.title('Latency Distribution by Client')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_distribution'))
    # plt.show()
    plt.close()

    # Split
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    sns.violinplot(data=plot_df[plot_df['Direction'] == 'Downlink'],
                   x='Client', y='Time (s)', inner='quartile', ax=ax1, density_norm="area", common_norm=True)
    #ax1.set_title('Downlink Latency Distribution')
    ax1.grid(True, alpha=0.3)

    sns.violinplot(data=plot_df[plot_df['Direction'] == 'Uplink'],
                   x='Client', y='Time (s)', inner='quartile', ax=ax2, density_norm="area", common_norm=True)
    #ax2.set_title('Uplink Latency Distribution')
    ax2.grid(True, alpha=0.3)
    #plt.suptitle('Latency Distribution by Client')
    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_distribution_split'))
    # plt.show()
    plt.close()


def plot_latency_histograms_by_cid(agg_latencies, output_dir):
    """Create histogram subplots for each CID's latencies"""
    num_cids = len(agg_latencies)
    fig, axs = plt.subplots(num_cids, 2, figsize=(12, 4 * num_cids))
    try:
        for idx, (cid, df) in enumerate(agg_latencies.items()):
            # Downlink histogram
            axs[idx, 0].hist(df['Downlink'], alpha=0.75, color='blue')
            #axs[idx, 0].set_title(f'{cid.replace("_", " ")} - Downlink Latency')
            axs[idx, 0].set_xlabel('Time (s)')
            axs[idx, 0].set_ylabel('Frequency')
            axs[idx, 0].grid(True, alpha=0.3)

            # Uplink histogram
            axs[idx, 1].hist(df['Uplink'], alpha=0.75, color='green')
            #axs[idx, 1].set_title(f'{cid.replace("_", " ")} - Uplink Latency')
            axs[idx, 1].set_xlabel('Time (ss)')
            axs[idx, 1].set_ylabel('Frequency')
            axs[idx, 1].grid(True, alpha=0.3)
    except Exception as e:
        print(f"Error plotting latency histograms by CID: {e}")
    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_histograms'))
    # plt.show()
    plt.close()


def retrieve_trial_metrics(configuration):
    trial_dirs = [d for d in listdir(configuration) if isdir(join(configuration, d))]

    # Latency
    latencies = {}
    agg_latencies = {}

    # Times
    training_times = {}
    train_test_times = {}
    val_test_times = {}
    round_times = {}
    comm_round_times = {}

    # ML Metrics
    val_accuracies = {}
    val_losses = {}
    train_accuracies = {}
    train_losses = {}

    for trial_dir in trial_dirs:
        trial_path = join(configuration, trial_dir)

        # Convert individual metrics to csv
        try:
            individual_metrics = _read_json(join(trial_path, 'individual_metrics.json'))
            export_metrics_to_csv(individual_metrics, trial_path)
        except Exception as e:
            print(f"Error processing {str(trial_path)}: {e}")

        try:
            # Process latency data captured by Flower
            trial_results = process_trial_latency(trial_path)
            for cid, df in trial_results.items():
                if cid not in latencies:
                    latencies[cid] = []
                latencies[cid].append(df)
            
            # Process training time metrics, measured by flower
            training_times[trial_dir] = pd.read_csv(join(trial_path, 'training_time.csv'))
            train_test_times[trial_dir] = pd.read_csv(join(trial_path, 'train_test_time.csv'))
            val_test_times[trial_dir] = pd.read_csv(join(trial_path, 'val_test_time.csv'))

            # Process round time metric
            round_times[trial_dir] = pd.read_csv(join(trial_path, 'round_time.csv'))
            comm_round_times[trial_dir] = pd.read_csv(join(trial_path, 'comm_round_time.csv'))

        except Exception as e:
            print(f"Error processing {trial_path}: {e}")

        # Process ML Metrics
        try:
            val_accuracies[trial_dir] = pd.read_csv(join(trial_path, 'val_acc.csv'))
            val_losses[trial_dir] = pd.read_csv(join(trial_path, 'val_loss.csv'))
            train_losses[trial_dir] = pd.read_csv(join(trial_path, 'train_loss.csv'))
            train_accuracies[trial_dir] = pd.read_csv(join(trial_path, 'train_acc.csv'))
        except Exception as e:
            print(f"Error processing {trial_path}: {e}")

    for cid, dfs in latencies.items():
        agg_df = pd.concat(dfs, axis=1).T.groupby(level=0).mean().T.drop('Round', axis=1)
        agg_latencies[cid] = agg_df
    
    configuration_path = str(configuration)

    try:
        agg_latencies = dict(sorted(agg_latencies.items(), key=lambda x: int(x[0].split('_')[1]))) # Sort the dict by CID
        pd.concat(agg_latencies, axis=1).to_csv(join(configuration, f'latencies_aggregated.csv'), index=False)  # Concat and save as csv
        if VERBOSITY >= 1:
            print(f'Saved {configuration_path}/latencies_aggregated.csv')
    except Exception as e:
        print(f"Error processing agg_latencies for {configuration_path}: {e}")

    agg_training_times = pd.DataFrame()
    agg_train_test_times = pd.DataFrame()
    agg_val_test_times = pd.DataFrame()
    agg_round_times = pd.DataFrame()
    agg_comm_round_times = pd.DataFrame()
    try:
        # Training Time Metrics
        agg_training_times = _aggregate_metrics(training_times, configuration_path, name='training_time')
        agg_train_test_times = _aggregate_metrics(train_test_times, configuration_path, name='train_test_time')
        agg_val_test_times = _aggregate_metrics(val_test_times, configuration_path, name='val_test_time')

        agg_round_times = _aggregate_metrics(round_times, configuration_path, name='round_time')
        agg_comm_round_times = _aggregate_metrics(comm_round_times, configuration_path, name='comm_round_time')

    except Exception as e:
        print(f"Error processing training time metrics for {configuration_path}: {e}")

    avg_val_accuracies = []
    avg_train_accuracies = []
    avg_val_accuracies = []
    avg_train_losses = []
    try:
        avg_val_accuracies = _aggregate_ml_metrics(val_accuracies, configuration_path, name='val_acc')
        avg_train_accuracies = _aggregate_ml_metrics(train_accuracies, configuration_path, name='train_acc')
        avg_val_losses = _aggregate_ml_metrics(val_losses, configuration_path, name='val_loss')
        avg_train_losses = _aggregate_ml_metrics(train_losses, configuration_path, name='train_loss')
    except Exception as e:
            assert f"Error processing ML metrics for {configuration_path}: {e}"
    client_times = pd.DataFrame()
    try:
        # Client-side round time
        uplink = pd.DataFrame({df: agg_latencies[df]['Uplink'] for df in agg_latencies})

        # Convert all DataFrames to numeric types before addition
        client_times = agg_training_times.apply(pd.to_numeric, errors='coerce')
        client_times = client_times.add(agg_train_test_times.apply(pd.to_numeric, errors='coerce'), fill_value=0)
        client_times = client_times.add(agg_val_test_times.apply(pd.to_numeric, errors='coerce'), fill_value=0)

        # Use pandas add method instead of += operator
        client_times = client_times.add(uplink.apply(pd.to_numeric, errors='coerce'), fill_value=0)

        client_times.to_csv(join(configuration_path, f'client_time_aggregated.csv'), index=False)
        if VERBOSITY >= 1:
            print(f'Saved {configuration_path}/client_time_aggregated.csv')
    except Exception as e:
        assert f"Error processing client time metrics for {configuration_path}: {e}"

    if PLOT_TRIAL_DATA:
        try:
            plot_latency_histograms_by_cid(agg_latencies, configuration_path)
            plot_latency_statistics_by_cid(agg_latencies, configuration_path)
            plot_time_histograms(agg_training_times, configuration_path, name='Training Time')
            plot_time_histograms(agg_train_test_times, configuration_path, name='Train Test Time')
            plot_time_histograms(agg_val_test_times, configuration_path, name='Validation Test Time')
            plot_time_histograms(agg_round_times, configuration_path, name='Round Time')
            plot_time_histograms(client_times, configuration_path, name='Client-Side Time')
            plot_time_histograms(agg_comm_round_times, configuration_path, name='Communication Round Time')
            accuracy_fig = plt.figure(figsize=(10, 6))
            loss_fig = plt.figure(figsize=(10, 6))

            colors = sns.color_palette()

            plot_ml_metric(avg_train_accuracies, accuracy_fig, 'Train', colors[0], agg_round_times)
            plot_ml_metric(avg_val_accuracies, accuracy_fig, 'Validation', colors[1], agg_round_times)
            plot_ml_metric(avg_train_losses, loss_fig, 'Train', colors[0], agg_round_times)
            plot_ml_metric(avg_val_losses, loss_fig, 'Validation', colors[1], agg_round_times)

            format_save_ml_plots(configuration_path, 'Accuracy', accuracy_fig)
            format_save_ml_plots(configuration_path, 'Loss', loss_fig)
        
        except Exception as e:
            print(f"Error plotting trial data on {configuration_path}: {e}")
    

    return trial_dirs


def retrieve_configuration_metrics(input_path, experiment_dir):
    for configuration_dir in experiment_dir:
        retrieve_trial_metrics(join(input_path, configuration_dir))


def plot_configurations(input_path, experiment_dir):
    time_metrics = ['Training Time', 'Train Test Time', 'Val Test Time', 'Round Time', 'Client Time', 'Comm Round Time']
    data = {}

    for result in experiment_dir:
        result_path = join(input_path, result)
        # Clean up the result name by replacing underscores with spaces and removing "IID"
        display_name = result.replace("_", " ").replace(" IID", "")

        try:
            data[display_name] = {}

            latency = pd.read_csv(join(str(result_path), 'latencies_aggregated.csv'), skiprows=1)
            data[display_name]['Uplink'] = pd.concat([latency[df] for df in latency if 'Uplink' in df], axis=1).mean(axis=1)
            data[display_name]['Uplink'].name = display_name
            data[display_name]['Downlink'] = pd.concat([latency[df] for df in latency if 'Downlink' in df], axis=1).mean(axis=1)
            data[display_name]['Downlink'].name = display_name

            for metric in time_metrics:
                filename = f'{metric.lower().replace(" ", "_")}_aggregated.csv'
                data[display_name][metric] = pd.read_csv(join(str(result_path), filename)).mean(axis=1)
                data[display_name][metric].name = display_name

            data[display_name]['Validation Accuracy'] = pd.read_csv(join(str(result_path), 'avg_val_acc.csv'))
            data[display_name]['Validation Loss'] = pd.read_csv(join(str(result_path), 'avg_val_loss.csv'))

        except Exception as e:
            if display_name in data:
                data.pop(display_name)
            print(f"Error processing {result}: {e}")

    # Sort configurations for consistent ordering in plots
    sorted_configs = sorted(data.keys())
    sorted_data = {config: data[config] for config in sorted_configs}
    data = sorted_data

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

        plot_ml_metric(data[configuration]['Validation Accuracy'], val_acc_fig, configuration, colors[idx], data[configuration]['Round Time'])
        plot_ml_metric(data[configuration]['Validation Loss'], val_loss_fig, configuration, colors[idx], data[configuration]['Round Time'])
        idx += 1

    save_latency_histograms(input_path, plots['Latency Hist'])

    uplink = pd.concat([data[configuration]['Uplink'] for configuration in data], axis=1)
    downlink = pd.concat([data[configuration]['Downlink'] for configuration in data], axis=1)
    plot_latency_statistics(uplink, downlink, input_path)

    format_save_ml_plots(input_path, 'Validation Accuracy', val_acc_fig)
    format_save_ml_plots(input_path, 'Validation Loss', val_loss_fig)

    for metric in time_metrics:
        try:
            # Create DataFrame with sorted column names
            data_concat = pd.concat([data[configuration][metric] for configuration in data], axis=1)
            data_concat.columns = data.keys()  # Set column names to configuration names
            plot_time_histograms(data_concat, input_path, metric)
        except Exception as e:
            print(f"Error processing {metric}: {e}")

def main(input_path: str) -> None:
    experiment_dirs = [d for d in listdir(input_path) if isdir(join(input_path, d))]

    if AGGREGATE_TRIAL_DATA:
        retrieve_configuration_metrics(input_path, experiment_dirs)

    if PLOT_EXPERIMENT_DATA:
        plot_configurations(input_path, experiment_dirs)


if __name__ == '__main__':
    ML_CONFIDENCE_BANDS = True
    PLOT_ROUNDS_INSTEAD_OF_TIME = False

    ''' level of print statements (higher level -> more print statements)
    - when csv files are created and saved from trial jsons (level 2)
    - csv files created for aggregated metrics (level 1)
    - errors always printed
    '''
    VERBOSITY = 1
    
    PLOT_EXPERIMENT_DATA = True # whether we should plot overview figures of all configurations
    AGGREGATE_TRIAL_DATA = True # whether we should go into each configuration (ex. 3Node_Ethernet_IID) and aggregate data from all trials (directories named with run IDs)
    PLOT_TRIAL_DATA = True # whether we should go into each configuration (ex. 3Node_Ethernet_IID) and plot time, latency data by CID; ML training and validation data. sub-condition of AGGREGATE_ALL_TRIAL_DATA
    IMPORT_ALL_TRIAL_DATA = True # whether we should go into each trial (within in a configuration) and export the json data to CSV files. sub-condition of AGGREGATE_ALL_TRIAL_DATA
    
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)