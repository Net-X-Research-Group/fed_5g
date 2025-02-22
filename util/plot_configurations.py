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


def main(input_path: str) -> None:
    experiment_dir = [d for d in listdir(input_path) if isdir(join(input_path, d))]

    # Latency
    latencies = {}

    # Times
    training_times = {}
    train_test_times = {}
    val_test_times = {}

    # ML Metrics
    val_accuracies = {}
    val_losses = {}
    train_accuracies = {}
    train_losses = {}

    # latency = ['latencies_aggregated']
    # time_metrics = ['training_time_aggregated', 'train_test_time_aggregated', 'val_test_time_aggregated']
    # ml_metrics = ['avg_val_acc', 'avg_val_loss', 'avg_train_loss', 'avg_train_acc']
    # metrics = latency + time_metrics + ml_metrics

    metrics = ['Accuracy', 'Loss']
    plots = {}
    data = {}

    for metric in metrics:
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
        latencies[num_nodes] = pd.read_csv(join(result_path, 'latencies_aggregated.csv'))
        training_times[num_nodes] = pd.read_csv(join(result_path, 'training_time_aggregated.csv'))
        train_test_times[num_nodes] = pd.read_csv(join(result_path, 'train_test_time_aggregated.csv'))
        val_test_times[num_nodes] = pd.read_csv(join(result_path, 'val_test_time_aggregated.csv'))

        # Process ML Metrics
        val_accuracies[num_nodes] = pd.read_csv(join(result_path, 'avg_val_acc.csv'))
        val_losses[num_nodes] = pd.read_csv(join(result_path, 'avg_val_loss.csv'))
        train_losses[num_nodes] = pd.read_csv(join(result_path, 'avg_train_loss.csv'))
        train_accuracies[num_nodes] = pd.read_csv(join(result_path, 'avg_train_acc.csv'))

        plot_ml_metric(val_accuracies[num_nodes], plots['Accuracy'], num_nodes + ' nodes - validation', nodes[num_nodes][0])
        plot_ml_metric(train_accuracies[num_nodes], plots['Accuracy'], num_nodes + ' nodes - train', nodes[num_nodes][1])
        
        plot_ml_metric(val_losses[num_nodes], plots['Loss'], num_nodes + ' nodes - validation', nodes[num_nodes][0])
        plot_ml_metric(train_losses[num_nodes], plots['Loss'], num_nodes + ' nodes - train', nodes[num_nodes][1])
        
        # except Exception as e:
        #     print(f"Error processing {result}: {e}")
    for metric in metrics:
        format_save_ml_plots(input_path, metric, plots[metric])


    # latencies_aggregated.csv
    # val_test_time_aggregated.csv
    # training_time_aggregated.csv
    # train_test_time_aggregated.csv
    # avg_val_acc.csv
    # avg_val_loss.csv
    # avg_train_loss.csv
    # avg_train_acc.csv



if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    # ../Library/CloudStorage/GoogleDrive-kaylacomer2029@u.northwestern.edu/My Drive/FL_5G_Experiments/
    main(input_dir)