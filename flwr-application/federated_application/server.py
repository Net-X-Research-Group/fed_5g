import logging
import time
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters, logger
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from federated_application.models import CNN3
from federated_application.strategy import MetricsFedAvg
from federated_application.task import get_weights

logger.logger.setLevel(logging.DEBUG)


# Define metric aggregation function
def fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """This function averages the `accuracy` metric sent by the clients in a `evaluate`
    stage (i.e. clients received the global model and evaluate it on their local
    validation sets)."""
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy and loss of each client.py by number of examples used
    train_accuracies = [num_examples * m["train_acc"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_acc"] for num_examples, m in metrics]
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]

    train_test_times = [m["train_test_time"] for _, m in metrics]
    val_test_times = [m["val_test_time"] for _, m in metrics]
    training_times = [m["training_time"] for _, m in metrics]
    fit_times = [time.time() - m['fit_time'] for _, m in metrics]

    # Calculate the weighted average of the metrics
    results = {
        "train_acc": sum(train_accuracies) / sum(examples),
        "val_acc": sum(val_accuracies) / sum(examples),
        "train_loss": sum(train_losses) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "training_time": sum(training_times) / len(training_times),
        "fit_time": sum(fit_times) / len(fit_times),
        'train_test_time': sum(train_test_times) / len(train_test_times),
        'val_test_time': sum(val_test_times) / len(val_test_times),
    }
    return results


def server_fn(context: Context):
    sample_fraction = context.run_config['fraction_evaluate']
    min_num_clients = context.run_config['min_num_clients']
    rounds = context.run_config['rounds']
    # Initialize model parameters on the central server
    ndarrays = get_weights(CNN3())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = MetricsFedAvg(
        run_config=context.run_config,
        fraction_fit=sample_fraction,
        fraction_evaluate=0,  # Disable Final Evaluation
        min_fit_clients=min_num_clients,
        min_available_clients=min_num_clients,
        min_evaluate_clients=min_num_clients,
        fit_metrics_aggregation_fn=fit_metrics,
        initial_parameters=parameters,
        run_id=context.run_id,
    )

    config = ServerConfig(num_rounds=rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
