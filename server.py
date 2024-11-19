import argparse
from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters
from markdown_it.cli.parse import parse_args

from task import Net, get_weights

import pandas as pd

#accuracy_plot = []
#training_time_avg_plot = []
#fit_time_avg_plot = []
perdevice_training_time_plot = []
perdevice_fit_time_plot = []

#eval_losses_plot = []
#fit_losses_plot = []

from datetime import time, timedelta
import time

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
    # Multiply accuracy and loss of each client.py by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m['loss'] for num_examples, m in metrics]
    # training_times = [m["training_time"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    #accuracy_plot.append(sum(accuracies) / sum(examples))
    #eval_losses_plot.append(sum(losses) / sum(examples))
    # training_time_avg_plot.append(sum(training_times) / len(training_times))

    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print(metrics)
    training_times = [m["training_time"] for _, m in metrics]
    fit_times = [time.time() - m['fit_time'] for _, m in metrics]
    perdevice_training_time_plot.append(training_times)
    perdevice_fit_time_plot.append(fit_times)
    #training_time_avg_plot.append(sum(training_times) / len(training_times))
    #fit_time_avg_plot.append(sum(fit_times) / len(fit_times))
    return {'training_time': sum(training_times) / len(training_times), 'fit_time': sum(fit_times) / len(fit_times)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # Number of local epochs done by clients
        "batch_size": 8,  # Batch size to use by clients during fit()
    }
    return config


def main():
    args = parser.parse_args()

    print(args)

    # Initialize model parameters on the central server
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_evaluate=args.sample_fraction,
        min_fit_clients=args.min_num_clients,
        min_available_clients=args.min_num_clients,
        min_evaluate_clients=args.min_num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics,
        initial_parameters=parameters

    )

    # Start Flower server
    server = fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    '''metrics = {
        'distributed_losses': [z[1] for z in server.losses_distributed],
        'accuracy': accuracy_plot,
        'training_time_avg': training_time_avg_plot,
        'fit_time_avg': fit_time_avg_plot,
        'perdevice_training_time': perdevice_training_time_plot,
        'perdevice_fit_time': perdevice_fit_time_plot
    }'''

    metrics = {
        'accuracy': [x[1] for x in server.metrics_distributed['accuracy']],
        'loss': [x[1] for x in server.losses_distributed],
        'training_time': [x[1] for x in server.metrics_distributed_fit['training_time']],
        'fit_time': [x[1] for x in server.metrics_distributed_fit['fit_time']],
    }

    per_device_metrics = {
        'training_time': perdevice_training_time_plot,
        'fit_time': perdevice_fit_time_plot
    }

    server_config = f'FEDAVG_CIFAR10_{args.rounds}R_{args.min_num_clients}C_3E_16B'

    print('ALL DONE')
    return metrics, per_device_metrics, server_config


if __name__ == "__main__":
    output, per_device_outputs, config = main()


    # Extract metrics from multi-dim list
    training_times = per_device_outputs['training_time']
    fit_times = per_device_outputs['fit_time']

    # Gather into dataframes
    df_distributed = pd.DataFrame(output)
    df_individual_training = pd.DataFrame(training_times)
    df_individual_fit = pd.DataFrame(fit_times)

    # Save metrics to CSV
    df_distributed.to_csv(f'{config}.csv', index=False)
    df_individual_training.to_csv(f'{config}_PERDEVTRAINING.csv', index=False)
    df_individual_fit.to_csv(f'{config}_PERDEVFIT.csv', index=False)