import logging
import time
from typing import List, Tuple

import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

from federated_application.models import ModelWrapper
from federated_application.strategy import MetricsFedAvg
from federated_application.task import get_weights

torch.manual_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fit_metrics(metrics: List[Tuple[int, Metrics]]) -> dict:
    """
    Combine all metrics from clients into a single object. Contains both aggregated metrics and individual metrics.

    Aggregated metrics: train_acc, val_acc, train_loss, val_loss, training_time, uplink_time, downlink_time, train_test_time, val_test_time
    Individual metrics: uplink_time, downlink_time, uplink_latency, downlink_latency, train_acc, val_acc, train_loss, val_loss, train_test_time, val_test_time, training_time, train_start_time, train_end_time

    Parameters:
        metrics (List[Tuple[int, Metrics]]): A list of tuples containing the number of examples and the metrics of each client

    Returns:
        results (dict): A dictionary containing the aggregated metrics and individual metrics of each client
    """
    recv_time = time.time()  # Time when the server receives the metrics (Rough)

    # Weighted average by number of examples per client
    examples = [num_examples for num_examples, _ in metrics]

    train_accuracies = [num_examples * m["train_acc"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_acc"] for num_examples, m in metrics]
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]

    uplink_times = [recv_time - m['uplink_time'] for _, m in metrics]
    downlink_times = [m['downlink_time'] for _, m in metrics]
    train_test_times = [m["train_test_time"] for _, m in metrics]
    val_test_times = [m["val_test_time"] for _, m in metrics]
    training_times = [m["training_time"] for _, m in metrics]

    individual_metrics = {m['cid']: {
        'uplink_time': recv_time - m['uplink_time'],
        'downlink_time': m['downlink_time'],
        'train_acc': m['train_acc'],
        'val_acc': m['val_acc'],
        'train_loss': m['train_loss'],
        'val_loss': m['val_loss'],
        'train_test_time': m['train_test_time'],
        'val_test_time': m['val_test_time'],
        'training_time': m['training_time'],
        'train_start_time': m['train_start'],
        'train_end_time': m['train_start'],
        'avg_trainloss': m['avg_train_loss']} for _, m in metrics}

    results = {
        "train_acc": sum(train_accuracies) / sum(examples),
        "val_acc": sum(val_accuracies) / sum(examples),
        "train_loss": sum(train_losses) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "training_time": sum(training_times) / len(training_times),
        "uplink_time": sum(uplink_times) / len(uplink_times),
        "downlink_time": sum(downlink_times) / len(downlink_times),
        'train_test_time': sum(train_test_times) / len(train_test_times),
        'val_test_time': sum(val_test_times) / len(val_test_times),
        'individual_metrics': individual_metrics
    }
    return results


def server_fn(context: Context):
    """
    Define the server-side logic for the federated learning process.

    Parameters:
        context (Context): The context object contains the configuration and the run_id of the server

    Returns:
        ServerAppComponents: A class that contains the strategy and configuration for the server run
    """
    min_num_clients = context.run_config['min_num_clients']
    rounds = context.run_config['rounds']
    model_name = context.run_config['model']

    net = ModelWrapper.create_model(model_name, num_classes=10)

    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    # Call instance of the metrics strategy
    strategy = MetricsFedAvg(
        run_config=context.run_config,
        enable_wandb=context.run_config['enable_server_wandb'],
        fraction_fit=1,  # Use all nodes
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


app = ServerApp(server_fn=server_fn)  # Create an instance of the server application
