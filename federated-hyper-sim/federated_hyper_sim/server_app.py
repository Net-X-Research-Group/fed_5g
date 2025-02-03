from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from torchvision.models import squeezenet1_1 as net

from federated_hyper_sim.sim_strategy import MetricsFedAvg
from federated_hyper_sim.task import get_weights


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

    # Calculate the weighted average of the metrics
    results = {
        "train_acc": sum(train_accuracies) / sum(examples),
        "val_acc": sum(val_accuracies) / sum(examples),
        "train_loss": sum(train_losses) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples)
        }
    return results


def server_fn(context: Context):
    sample_fraction = context.run_config['fraction_evaluate']
    min_num_clients = context.run_config['min_num_clients']
    rounds = context.run_config['rounds']
    # Initialize model parameters on the central server
    ndarrays = get_weights(net(num_classes=10))
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = MetricsFedAvg(
        run_config=context.run_config,
        enable_wandb=context.run_config['enable_server_wandb'],
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
