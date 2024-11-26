import logging
import time
import warnings
from os import path

import torch
from federated_application.models import CNN3
from federated_application.task import (
    get_weights,
    set_weights,
    load_dataset,
    train,
    test
)
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
logger = logging.getLogger(__name__)

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate) -> None:
        self.net = CNN3()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
    def fit(self, parameters, config) -> tuple:
        """Train the client model on the local training dataset"""
        get_weights(self.net)
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.learning_rate,
            self.device
        )
        logger.info(f"Training complete. Elapsed time: {results['training_time']}")
        results['fit_time'] = time.time()
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the client model on the local validation dataset"""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        metrics = {
            'accuracy': accuracy,
            'loss': loss,
            'eval_time': time.time()
        }
        return loss, len(self.valloader.dataset), metrics

def client_fn(context: Context):
    dataset_path = path.expanduser(f"{context.node_config['dataset']}_part_{context.node_config['cid']}")
    batch_size = context.run_config['batch_size']
    local_epochs = context.run_config['local_epochs']
    learning_rate = context.run_config['learning_rate']

    trainloader, valloader = load_dataset(dataset_path, batch_size)

    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


app = ClientApp(client_fn=client_fn)
