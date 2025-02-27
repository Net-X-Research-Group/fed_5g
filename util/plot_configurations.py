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
        plt.fill_between(mean.index, lower, upper, alpha=0.3, color=color)

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


def main(input_path: str) -> None:
    experiment_dir = [d for d in listdir(input_path) if isdir(join(input_path, d))]
    
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

    for metric in time_metrics:
        try:
            data = pd.concat([data[configuration][metric] for configuration in data], axis=1)
            plot_time_histograms(data, input_path, metric)
        except Exception as e:
            print(f"Error processing {metric} for {configuration}: {e}")


if __name__ == '__main__':
    ML_CONFIDENCE_BANDS = False
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)