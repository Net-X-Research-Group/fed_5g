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


def plot_ml_metric(data: pd.DataFrame, fig, label: str, color: str) -> None:
    plt.figure(fig)
    ax = plt.gca()

    """Plot training and validation metrics with confidence bands."""
    # Calculate statistics
    mean = data.mean(axis=1)
    std = data.std(axis=1)

    # Create confidence bands
    lower = mean - std
    upper = mean + std

    # Create the plot
    sns.lineplot(data=mean, label=label, color=color, ax=ax)
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
    ax = plt.gca()

    plt.tight_layout()
    plt.savefig(join(output_dir, 'latency_histograms'))
    # plt.show()
    plt.close()


# def plot_latency_statistics(agg_latencies, fig, num_rows, row_idx, num_nodes):
#     """
#     Create violin plots for the latencies captured by flower.
#     """
#     plot_data = []
#     for cid, df in agg_latencies.items():
#         plot_data.extend([{
#             'Client': cid,
#             'Direction': 'Downlink',
#             'Latency (ms)': value
#         } for value in df['Downlink']])
#         plot_data.extend([{
#             'Client': cid,
#             'Direction': 'Uplink',
#             'Latency (ms)': value
#         } for value in df['Uplink']])

#     plot_df = pd.DataFrame(plot_data)

#     plt.figure(figsize=(12, 6))
#     sns.violinplot(data=plot_df, x='Client', y='Latency (ms)',
#                    hue='Direction', split=True, inner='quartile')

#     plt.title('Latency Distribution by Client')
#     plt.xticks(rotation=45)
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(join(output_dir, 'latency_distribution'))
#     # plt.show()
#     plt.close()

#     # Split
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
#     sns.violinplot(data=plot_df[plot_df['Direction'] == 'Downlink'],
#                    x='Client', y='Latency (ms)', inner='quartile', ax=ax1)
#     ax1.set_title('Downlink Latency Distribution')
#     ax1.grid(True, alpha=0.3)

#     sns.violinplot(data=plot_df[plot_df['Direction'] == 'Uplink'],
#                    x='Client', y='Latency (ms)', inner='quartile', ax=ax2)
#     ax2.set_title('Uplink Latency Distribution')
#     ax2.grid(True, alpha=0.3)
#     plt.xticks(rotation=45)
#     plt.suptitle('Latency Distribution by Client')
#     plt.tight_layout()
#     plt.savefig(join(output_dir, 'latency_distribution_split'))
#     # plt.show()
#     plt.close()


def plot_latency_histograms(agg_latencies, fig, num_rows, row_idx, num_nodes):
    plt.figure(fig)

    uplink = pd.concat([agg_latencies[df] for df in agg_latencies if 'Uplink' in df], axis=1).mean(axis=1)
    downlink = pd.concat([agg_latencies[df] for df in agg_latencies if 'Downlink' in df], axis=1).mean(axis=1)

    # Downlink histogram
    ax = fig.add_subplot(num_rows, 2, 2*row_idx+1)
    ax.hist(downlink, alpha=0.75, color='blue')
    ax.set_title(f'{num_nodes} Nodes - Downlink Latency')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

    # Uplink histogram
    ax = fig.add_subplot(num_rows, 2, 2*row_idx+2)
    ax.hist(uplink, alpha=0.75, color='green')
    ax.set_title(f'{num_nodes} Nodes - Uplink Latency')
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)


def plot_time_histograms(data: pd.DataFrame, figs, num_rows, row_idx, num_nodes):
    plt.figure(figs[0])
    ax = plt.gca()

    """Create histogram subplots for each CID's various time measurements"""
    # num_cids = len(data.columns)
    # fig, axs = plt.subplots(num_cids, 1, figsize=(10, 4 * num_cids))

    # Handle single column case
    # if num_cids == 1:
    #     axs = [axs]

    # Main Histogram
    ax.hist(data, alpha=0.75, color='blue')
    # ax.set_title(f'{col.replace("_", " ")} Distribution')
    
    

    # Violin plots
    plt.figure(figs[1])
    ax = plt.gca()
    
    # plot_data = pd.melt(data, var_name='Client', value_name='Time (s)')
    # sns.violinplot(data=data, x='Client', y='Time (s)', inner='quartile')

    # # Overlay Histograms
    # plt.figure(figs[2])
    # ax = plt.gca()
    # # fig, ax = plt.subplots(figsize=(10, 6))
    # for col in data.columns:
    #     ax.hist(data[col], alpha=0.75, label=col.replace("_", " "))


def format_save_time_figs(name, output_dir, figs):
    # normal hist
    plt.figure(figs[0])
    ax = plt.gca()

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    plt.suptitle(f'{name} Distribution by Client')
    plt.tight_layout()
    plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_histograms'))
    # plt.show()
    plt.close()

    # # violin plots
    # plt.figure(figs[1])
    # ax = plt.gca()

    # plt.title(f'{name} Distribution by Client')
    # plt.xticks(rotation=45)
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_violin'))
    # # plt.show()
    # plt.close()

    # # overlay hist
    # plt.figure(figs[2])
    # ax = plt.gca()

    # ax.set_title(f'{name} Distribution')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel('Frequency')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig(join(output_dir, f'{name.lower().replace(" ", "_")}_overlay_histogram'))
    # # plt.show()
    # plt.close()


def main(input_path: str) -> None:
    experiment_dir = [d for d in listdir(input_path) if isdir(join(input_path, d))]

    # Latency
    latencies = {}

    # Times
    training_times = {}
    train_test_times = {}
    val_test_times = {}
    time_metrics = ['training_times', 'train_test_times', 'val_test_times']

    # # ML Metrics
    # val_accuracies = {}
    # val_losses = {}
    # train_accuracies = {}
    # train_losses = {}

    # latency = ['latencies_aggregated']
    # time_metrics = ['training_time_aggregated', 'train_test_time_aggregated', 'val_test_time_aggregated']
    # ml_metrics = ['avg_val_acc', 'avg_val_loss', 'avg_train_loss', 'avg_train_acc']
    # metrics = latency + time_metrics + ml_metrics

    # time_metrics = ['Training Time', 'Train Test Time', 'Validation Test Time']
    ml_metrics = ['Accuracy', 'Loss']
    plots = {}
    data = {}

    for metric in ml_metrics:
        plots[metric] = plt.figure(figsize=(10, 6))

    # assume only have Ethernet, IID for now
    colors = [sns.color_palette(), sns.color_palette("pastel")]
    # networks = {}
    nodes = {}
    # distributions = {}
    for result in experiment_dir:
        result_path = join(input_path, result)

        meta = result.split('_')
        if len(meta) != 3: # TODO write in a better way -- maybe don't need because of exception handling
            break
        # try:
        network = meta[0]
        num_nodes = meta[1][0]
        distribution = meta[2]

        # Directory validation
        # if network not in networks:
        #     networks[network] = 'blue'
        if num_nodes not in nodes:
            nodes[num_nodes] = [colors[0].pop(0), colors[1].pop(0)]
        # if distribution not in distributions:
        #     distributions[distribution] = 'blue'
        # for metric in metrics:
        #     data[metric] = pd.read_csv(join(result_path, metric + '.csv'))
        
        # Load aggregated training time and averaged ML metrics
        latencies[num_nodes] = pd.read_csv(join(result_path, 'latencies_aggregated.csv'), skiprows=1)

        training_times[num_nodes] = pd.read_csv(join(result_path, 'training_time_aggregated.csv')).mean(axis=1)
        train_test_times[num_nodes] = pd.read_csv(join(result_path, 'train_test_time_aggregated.csv')).mean(axis=1)
        val_test_times[num_nodes] = pd.read_csv(join(result_path, 'val_test_time_aggregated.csv')).mean(axis=1)

        # for metric in time_metrics:
        #     filename = f'{metric.lower().replace(" ", "_")}'

        # Import and plot ML Metrics
        val_accuracies = pd.read_csv(join(result_path, 'avg_val_acc.csv'))
        val_losses = pd.read_csv(join(result_path, 'avg_val_loss.csv'))
        train_losses = pd.read_csv(join(result_path, 'avg_train_loss.csv'))
        train_accuracies = pd.read_csv(join(result_path, 'avg_train_acc.csv'))

        plot_ml_metric(val_accuracies, plots['Accuracy'], num_nodes + ' nodes - validation', nodes[num_nodes][0])
        plot_ml_metric(train_accuracies, plots['Accuracy'], num_nodes + ' nodes - train', nodes[num_nodes][1])
        
        plot_ml_metric(val_losses, plots['Loss'], num_nodes + ' nodes - validation', nodes[num_nodes][0])
        plot_ml_metric(train_losses, plots['Loss'], num_nodes + ' nodes - train', nodes[num_nodes][1])
        
        # except Exception as e:
        #     print(f"Error processing {result}: {e}")
    for metric in ml_metrics:
        format_save_ml_plots(input_path, metric, plots[metric])
    
    num_subplots = len(latencies)
    latency_hist, axs = plt.subplots(num_subplots, 2, figsize=(12, 4 * num_subplots))
    latency_violin, axs = plt.subplots(num_subplots, 2, figsize=(12, 4 * num_subplots))

    for metric in time_metrics:
        hist, axs = plt.subplots(num_subplots, 1, figsize=(10, 4 * num_subplots))
        violin, ax = plt.subplots(figsize=(10, 6))
        overlay_hist, ax = plt.subplots(figsize=(10, 6))
        
        plots[metric] = [hist, violin, overlay_hist]
    i = 0
    for num_nodes in latencies:
        plot_latency_histograms(latencies[num_nodes], latency_hist, num_subplots, i, num_nodes)
        
        # for metric in time_metrics:
        plot_time_histograms(training_times[num_nodes], plots['training_times'], num_subplots, i, num_nodes)
        plot_time_histograms(train_test_times[num_nodes], plots['train_test_times'], num_subplots, i, num_nodes)
        plot_time_histograms(val_test_times[num_nodes], plots['val_test_times'], num_subplots, i, num_nodes)
        # plot_latency_statistics(latencies[num_nodes], latency_violin, num_subplots, i, num_nodes)
        i += 1
    
    save_latency_histograms(input_path, latency_hist)
    for metric in time_metrics:
        format_save_time_figs(metric, input_path, plots[metric])


if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    # ../Library/CloudStorage/GoogleDrive-kaylacomer2029@u.northwestern.edu/My Drive/FL_5G_Experiments/
    main(input_dir)