import argparse
from typing import List, Tuple
import time
import flwr as fl
from flwr.common import Metrics
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

accuracy_plot = []
loss_plot = []

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
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    accuracy_plot.append(sum(accuracies) / sum(examples))
    loss_plot.append(sum(losses) / sum(examples))
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 3,  # Number of local epochs done by clients
        "batch_size": 16,  # Batch size to use by clients during fit()
    }
    return config


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
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

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


if __name__ == "__main__":
    main()
