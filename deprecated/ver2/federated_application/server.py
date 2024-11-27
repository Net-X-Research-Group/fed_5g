from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from federated_application.task import Net, get_weights
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

accuracy_plot = []
training_time_avg_plot = []
perdevice_training_time_plot = []


# Define metric aggregation function
def weighted_average_2(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    accuracy_plot.append(sum(accuracies) / sum(examples))
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print(metrics)
    training_times = [m["training_time"] for _, m in metrics]
    perdevice_training_time_plot.append(training_times)
    training_time_avg_plot.append(sum(training_times) / len(training_times))
    return {'training_time': sum(training_times) / len(training_times)}

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


def server_fn(context: Context):
    """Flower server implementation"""

    # Read from the pyproject config file
    num_rounds = context.run_config['num-server-rounds']

    # Initialize model parameters on the central server
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=context.run_config['fraction_evaluate'],
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average_2,
        initial_parameters=parameters
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerApp(strategy=strategy, config=config)

app = ServerApp(server_fn=server_fn)