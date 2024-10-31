import argparse
from typing import List, Tuple
import time
import flwr as fl
from flwr.common import Metrics
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

from new_server import train_time_plot

accuracy_plot = []
loss_plot = []
training_time_avg_plot = []

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
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    training_times = [m["training_time"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    accuracy_plot.append(sum(accuracies) / sum(examples))
    loss_plot.append(sum(losses) / sum(examples))
    training_time_avg_plot.append(sum(training_times) / len(training_times))
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 3,  # Number of local epochs done by clients
        "batch_size": 16,  # Batch size to use by clients during fit()
    }
    return config

def plot_metrics(args):
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

    # Plot computation time
    ax3.plot(rounds, training_time_avg_plot, marker='o', color='green')
    ax3.set_xlabel('Rounds')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Average Computation Time per Round')
    ax3.grid(True)

    # Plot communication time
    ax4.plot(rounds, loss_plot, marker='o', color='purple')
    ax4.set_xlabel('Rounds')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Average Communication Time per Round')
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
    )

    # Start Flower server
    server = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
    loss_plot = [z[1] for z in server.losses_distributed]
    # Plot all metrics
    plot_metrics(args)

    '''
    print(f"{server.metrics_distributed = }")
    
    global_accuracy_distributed = server.metrics_distributed["accuracy"]
    round = [data[0] for data in global_accuracy_distributed]
    acc = [100.0 * data[1] for data in global_accuracy_distributed]
    plt.plot(round, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("MNIST - IID - 2 clients with 2 clients per round")
    plt.show()
    '''
    '''
    x1 = range(len(accuracy_plot))
    plt.figure(1)
    plt.plot(x1, accuracy_plot, marker='o', linestyle='solid')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Rounds')
    plt.savefig(f'FedAvg_{args.min_num_clients}C_{args.rounds}R_Accuracy.png')

    x2 = range(len(loss_plot))
    plt.figure(2)
    plt.plot(x2, loss_plot, marker='o', linestyle='solid')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Loss Function')
    plt.savefig(f'FedAvg_{args.min_num_clients}C_{args.rounds}R_Loss.png')
    '''

if __name__ == "__main__":
    main()
