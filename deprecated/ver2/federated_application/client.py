import time
import logging
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from federated_application.task import (
    Net,
    get_weights,
    set_weights,
    load_dataset,
    train,
    test
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate) -> None:
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config) -> tuple:
        """Train the client model on the local training dataset"""
        set_weights(self.net, parameters)
        start_time = time.time()
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.learning_rate,
            self.device
        )
        results['training_time'] = time.time() - start_time # Add training time to results
        logger.info(f"Training complete. Elapsed time: {results['elapsed_time']}")
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the client model on the local validation dataset"""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        results = {
            'accuracy': accuracy
        }
        logger.info(f"Evaluation loss: {loss}, accuracy: {accuracy}")
        return loss, len(self.valloader.dataset), results

def client_fn(context: Context):
    dataset_path = context.node_config['dataset_path']

    batch_size = context.run_config['batch_size']
    trainloader, valloader = load_dataset(dataset_path, batch_size)
    local_epochs = context.run_config['local_epochs']
    learning_rate = context.run_config['learning_rate']

    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()

app = ClientApp(client_fn)
