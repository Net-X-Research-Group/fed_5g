import argparse
import json
import time
from typing import List, Tuple, Dict, Any
from datetime import datetime
import flwr as fl
from flwr.common import Metrics
from flwr.server.client_proxy import ClientProxy
import matplotlib.pyplot as plt


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

#Initialize metric tracking lists
accuracy_plot = []
loss_plot = []
train_time_plot = []
comm_time_plot = []

class MetricsLogger:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.metrics_history = {
            "rounds": [],
            "accuracy": [],
            "loss": [],
            "client_metrics": {},
            "training_time": [],
            "communication_time": {},  # Add communication time tracking
            "num_clients_per_round": [],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

    def log_round_metrics(self, server_round: int, metrics: Dict[str, float],
                          client_metrics: List[Tuple[int, Metrics]],
                          round_duration: float,
                          communication_times: Dict[str, float]):
        self.metrics_history["rounds"].append(server_round)
        self.metrics_history["accuracy"].append(metrics["accuracy"])
        self.metrics_history["loss"].append(metrics["loss"])
        self.metrics_history["training_time"].append(round_duration)
        self.metrics_history["num_clients_per_round"].append(len(client_metrics))
        self.metrics_history["communication_time"][server_round] = communication_times

        # Log individual client metrics
        round_client_metrics = {}
        for num_examples, metrics in client_metrics:
            client_id = metrics.get("client_id", "unknown")
            round_client_metrics[client_id] = {
                "num_examples": num_examples,
                "accuracy": metrics["accuracy"],
                "loss": metrics["loss"]
            }
        self.metrics_history["client_metrics"][server_round] = round_client_metrics

    def save_metrics(self):
        filename = f"metrics_{self.experiment_name}_{self.metrics_history['timestamp']}.json"
        with open(filename, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        self._generate_plots()

    def _generate_plots(self):
        timestamp = self.metrics_history["timestamp"]

        # Create three subplots: accuracy, loss, and communication time
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

        rounds = self.metrics_history["rounds"]

        # Accuracy plot
        ax1.plot(rounds, self.metrics_history["accuracy"], marker='o')
        ax1.set_title("Model Accuracy per Round")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy")
        ax1.grid(True)

        # Loss plot
        ax2.plot(rounds, self.metrics_history["loss"], marker='o', color='red')
        ax2.set_title("Model Loss per Round")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Loss")
        ax2.grid(True)

        # Communication time plot
        comm_times = []
        for round_num in rounds:
            round_times = self.metrics_history["communication_time"].get(str(round_num), {})
            avg_time = sum(round_times.values()) / len(round_times) if round_times else 0
            comm_times.append(avg_time)

        ax3.plot(rounds, comm_times, marker='o', color='green')
        ax3.set_title("Average Communication Time per Round")
        ax3.set_xlabel("Round")
        ax3.set_ylabel("Time (seconds)")
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(f"training_metrics_{timestamp}.png")
        plt.close()


class CommunicationTrackingStrategy(fl.server.strategy.FedAvg):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metrics_logger = MetricsLogger(
            f"fedavg_{kwargs.get('min_fit_clients')}clients"
        )
        self.round_start_time = None
        self.communication_times = {}

    def initialize_parameters(self, client_manager):
        self.round_start_time = time.time()
        return super().initialize_parameters(client_manager)

    def configure_fit(
            self, server_round: int, parameters, client_manager
    ):
        """Configure the next round of training."""
        self.communication_times[server_round] = {}
        self.round_start_time = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
            self, server_round: int, parameters, client_manager
    ):
        """Configure the next round of evaluation."""
        return super().configure_evaluate(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model updates from clients."""
        # Record communication times for successful clients
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            self.communication_times[server_round][client_id] = {
                "fit_duration": fit_res.metrics.get("fit_duration", 0),
                "communication_duration": fit_res.metrics.get("communication_duration", 0)
            }

        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from clients."""
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated is not None:
            metrics, _ = aggregated
            round_duration = time.time() - self.round_start_time

            # Get all client metrics from this round
            client_metrics = [(num_examples, metrics) for _, (num_examples, metrics, _) in results]

            # Log metrics including communication times
            self.metrics_logger.log_round_metrics(
                server_round,
                metrics,
                client_metrics,
                round_duration,
                self.communication_times.get(server_round, {})
            )

        return aggregated

    def on_evaluate_end(self, server_round: int):
        """Called after evaluate stage."""
        if self._is_last_round(server_round):
            self.metrics_logger.save_metrics()

    def _is_last_round(self, server_round: int) -> bool:
        """Determine if this is the last round."""
        return server_round == self.num_rounds


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 3,
        "batch_size": 16,
    }
    return config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics from clients."""
    # Multiply metrics by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    compute_times = [m["compute_time"] for _, m in metrics]
    comm_times_receive = [m["comm_time_receive"] for _, m in metrics]
    comm_times_send = [m["comm_time_send"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate metrics
    avg_accuracy = sum(accuracies) / sum(examples)
    avg_loss = sum(losses) / sum(examples)
    avg_compute_time = sum(compute_times) / len(compute_times)
    avg_comm_time = (sum(comm_times_receive) + sum(comm_times_send)) / len(metrics)

    # Store metrics for plotting
    accuracy_plot.append(avg_accuracy)
    loss_plot.append(avg_loss)
    train_time_plot.append(avg_compute_time)
    comm_time_plot.append(avg_comm_time)

    return {
        "accuracy": avg_accuracy,
        "loss": avg_loss,
        "compute_time": avg_compute_time,
        "communication_time": avg_comm_time
    }


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
    ax3.plot(rounds, train_time_plot, marker='o', color='green')
    ax3.set_xlabel('Rounds')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Average Computation Time per Round')
    ax3.grid(True)

    # Plot communication time
    ax4.plot(rounds, comm_time_plot, marker='o', color='purple')
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

    # Start Flower server and time the entire training process
    start_time = time.time()
    server = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    total_time = time.time() - start_time

    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per round: {total_time / args.rounds:.2f} seconds")

    # Plot all metrics
    plot_metrics(args)


if __name__ == "__main__":
    main()
