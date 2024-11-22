from typing import List, Tuple
import flwr as fl
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from federated_application.task import get_weights
from federated_application.models import CNN3

import pandas as pd

perdevice_training_time = []
perdevice_fit_time = []


from datetime import time
import time

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """This function averages the `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    # Multiply accuracy and loss of each client.py by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print(metrics)
    training_times = [m["training_time"] for _, m in metrics]
    fit_times = [time.time() - m['fit_time'] for _, m in metrics]
    perdevice_training_time.append(training_times)
    perdevice_fit_time.append(fit_times)
    return {'training_time': sum(training_times) / len(training_times), 'fit_time': sum(fit_times) / len(fit_times)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 2,  # Number of local epochs done by clients
        "batch_size": 16,  # Batch size to use by clients during fit()
        "learning_rate": 0.01
    }
    return config


def server_fn(context: Context):
    sample_fraction = context.run_config['fraction-evaluate']
    min_num_clients = context.run_config['min_num_clients']
    rounds = context.run_config['rounds']
    # Initialize model parameters on the central server
    ndarrays = get_weights(CNN3())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=sample_fraction,
        fraction_evaluate=sample_fraction,
        min_fit_clients=min_num_clients,
        min_available_clients=min_num_clients,
        min_evaluate_clients=min_num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=fit_metrics,
        initial_parameters=parameters

    )

    config = ServerConfig(num_rounds=rounds)

    return ServerAppComponents(strategy=strategy, config=config)

if __name__ == "__main__":
    app = ServerApp(server_fn=server_fn)

    metrics = {
        'accuracy': [x[1] for x in app.metrics_distributed['accuracy']],
        'loss': [x[1] for x in app.losses_distributed],
        'training_time': [x[1] for x in app.metrics_distributed_fit['training_time']],
        'fit_time': [x[1] for x in app.metrics_distributed_fit['fit_time']],
    }

    per_device_metrics = {
        'training_time': perdevice_training_time,
        'fit_time': perdevice_fit_time
    }

    #server_config_name = f'FEDAVG_CIFAR10_{args.rounds}R_{args.min_num_clients}C_3E_16B'
    '''
    # Extract metrics from multi-dim list
    training_times = per_device_outputs['training_time']
    fit_times = per_device_outputs['fit_time']

    # Gather into dataframes
    df_distributed = pd.DataFrame(output)
    df_individual_training = pd.DataFrame(training_times)
    df_individual_fit = pd.DataFrame(fit_times)

    # Save metrics to CSV
    df_distributed.to_csv(f'{server_config}.csv', index=False)
    df_individual_training.to_csv(f'{server_config}_PERDEVTRAINING.csv', index=False)
    df_individual_fit.to_csv(f'{server_config}_PERDEVFIT.csv', index=False)'''
