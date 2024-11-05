from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from task import Net, get_weights
from typing import List, Tuple


# Define metric aggregation function
def weighted_average_2(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


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