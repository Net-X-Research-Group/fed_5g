import argparse
from typing import List, Tuple
import time
import flwr as fl
from flwr.common import Metrics
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
accuracy_plot = []
training_time_avg_plot = []
perdevice_training_time_plot = []

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--rounds",
    type=int,
    default=5,
    help="Number of rounds of federated learning (default: 5)",
)
parser.add_argument(
    "--sample_fraction",
    type=float,
    default=1.0,
    help="Fraction of available clients used for fit/evaluate (default: 1.0)",
)
parser.add_argument(
    "--min_num_clients",
    type=int,
    default=2,
    help="Minimum number of available clients required for sampling (default: 2)",
)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """This function averages the `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy and loss of each client by number of examples used
    print(metrics)
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    #training_times = [m["training_time"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    accuracy_plot.append(sum(accuracies) / sum(examples))
    #training_time_avg_plot.append(sum(training_times) / len(training_times))
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print(metrics)
    training_times = [m["training_time"] for _, m in metrics]
    perdevice_training_time_plot.append(training_times)
    training_time_avg_plot.append(sum(training_times) / len(training_times))
    return {'training_time': sum(training_times) / len(training_times)}

def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 3,  # Number of local epochs done by clients
        "batch_size": 16,  # Batch size to use by clients during fit()
    }
    return config

def plot_metrics(args, loss_plot: list):
    """Plot all tracked metrics."""
    rounds = range(1, len(accuracy_plot) + 1)

    # Create a figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot accuracy
    ax1.plot(rounds, accuracy_plot, marker='o')
    ax1.set_xlabel('Rounds')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy vs. Rounds')
    ax1.grid(True)
    # Plot loss
    ax2.plot(rounds, loss_plot, marker='o', color='red')
    ax2.set_xlabel('Rounds')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss vs. Rounds')
    ax2.grid(True)

    # Find the maximum length among all rounds
    max_num_devices = max(len(round_times) for round_times in perdevice_training_time_plot)

    # Pad each round's data with zeros to match max length
    padded_times = [round_times + [0] * (max_num_devices - len(round_times))
                    for round_times in perdevice_training_time_plot]

    # Reorganize data to group by device
    device_times = []
    for device_idx in range(max_num_devices):
        device_times.append([round_times[device_idx] for round_times in padded_times])

    # Create a color map for different devices
    colors = plt.cm.rainbow(np.linspace(0, 1, max_num_devices))

    # Plot a line for each device
    for device_idx, times in enumerate(device_times):
        ax3.plot(rounds, times, marker='o', color=colors[device_idx], linestyle='--',
                 label=f'Device {device_idx + 1}')
    ax3.plot(rounds, training_time_avg_plot, marker='^', color='red', label='Average')
    ax3.set_xlabel('Rounds')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computation Time per Device per Round')
    ax3.grid(True)
    ax3.legend()


    # Plot communication time
    ax4.plot(rounds, loss_plot, marker='o', color='purple')
    ax4.set_xlabel('Rounds')
    ax4.set_ylabel('Time (seconds)')
    #ax4.set_title('Average Communication Time per Round')
    ax4.set_title('Aint workin yet')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(f'FedAvg_{args.min_num_clients}C_{args.rounds}R_Metrics.png')
    plt.show()

def main():
    args = parser.parse_args()

    print(args)

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics
    )

    # Start Flower server
    server = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    loss_plot = [z[1] for z in server.losses_distributed]
    print(loss_plot)
    # Plot all metrics
    plot_metrics(args, loss_plot)
    print('ALL DONE')

if __name__ == "__main__":
    main()
