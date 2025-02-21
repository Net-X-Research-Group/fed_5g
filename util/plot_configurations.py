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

    # assume only have Ethernet, IID for now
    # networks = []
    # nodes = []
    # distributions = []
    for result in experiment_dir:
        result_path = join(input_path, result)

        meta = result.split('_')
        if len(meta) != 3: # TODO write in a better way
            break
        try:
            network = meta[0]
            num_nodes = meta[1][0]
            distribution = meta[2]

            # Directory validation
            # if network not in networks:
            #     networks.append(network)
            # if num_nodes not in nodes:
            #     nodes.append(num_nodes)
            # if distribution not in distributions:
            #     distributions.append(distribution)

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
        
        except Exception as e:
            print(f"Error processing {result}: {e}")
    

        
        
        

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