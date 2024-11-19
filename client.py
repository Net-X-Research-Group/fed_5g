import time
import logging
import torch
from flwr.client import NumPyClient, start_client
import argparse
import warnings
from os import path

from task import (
    Net,
    get_weights,
    set_weights,
    load_dataset,
    train,
    test
)

parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Dataset path to train on"
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader) -> None:
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config) -> tuple:
        """Train the client model on the local training dataset"""
        batch, epochs, learning_rate = config["batch_size"], config["epochs"], config["learning_rate"]
        get_weights(self.net)
        set_weights(self.net, parameters)
        start_time = time.time()
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            epochs,
            learning_rate,
            self.device
        )
        end_time = time.time()
        metrics = {
            "training_time": end_time - start_time,
            "fit_time": time.time()
        }

        logger.info(f"Training complete. Elapsed time: {metrics['training_time']}")
        #logger.info(f'val_loss={results["val_loss"]}, val_acc={results["val_acc"]}')
        return get_weights(self.net), len(self.trainloader.dataset), metrics

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

def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS


    dataset_path = path.expanduser(f'{args.dataset}_part_{args.cid}')

    trainloader, valloader = load_dataset(dataset_path, 16)

    # Start Flower client setting its associated data partition
    start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainloader=trainloader, valloader=valloader).to_client(),
    )


if __name__ == "__main__":
    main()
