import json
import logging
import warnings
from datetime import datetime
from os import path

import torch
import wandb
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, logger

#from federated_application.models import CNN3v2
from torchvision.models import mobilenet_v3_small as net
from federated_application.mods import comm_time_mod
from federated_application.task import (
    get_weights,
    set_weights,
    load_dataset,
    train,
    test
)

warnings.filterwarnings("ignore", category=UserWarning)

logger.logger.setLevel(logging.DEBUG)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
logger = logging.getLogger(__name__)

PROJECT_NAME = "Pytorch-5G-FLWR-CIFAR10"

class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, enable_wandb, wandb_config) -> None:
        self.net = net(num_classes=10)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.enable_wandb = enable_wandb
        self.wandb_config = wandb_config

        # Login to wandb
        if enable_wandb:
            logger.info('Enabling wandb...')
            wandb.login(key=wandb_config['wandb_api_key'])
            self.wandb_config.pop('wandb_api_key')

            # Initialize the wandb project
            self._init_wandb_project()

    def _init_wandb_project(self):
        wandb.init(project=PROJECT_NAME,
                   group = str(self.wandb_config['run_id']),
                   id = f'{self.wandb_config["run_id"]}-{self.wandb_config["cid"]}',
                   name=f'{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}_CID-{self.wandb_config["cid"]}',
                   resume='allow',
                   reinit=True,
                   config=dict(self.wandb_config))

    def fit(self, parameters, config) -> tuple:
        """Train the client model on the local training dataset"""
        #downlink_time = time.time() - config['server_timestamp']
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
        results['cid'] = self.wandb_config['cid']
        
        with open(path.expanduser('~/latency.json')) as f:
            latency = json.load(f)
        results['downlink_latency'] = latency['downlink']
        results['uplink_latency'] = latency['uplink']
        logger.info(f"Training complete. Elapsed time: {results['training_time']}")
        if self.enable_wandb:
            wandb.log(results)
        #results['uplink_time'] = time.time()
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the client model on the local validation dataset"""
        set_weights(self.net, parameters)
        loss, accuracy, eval_time = test(self.net, self.valloader, self.device)
        metrics = {
            'accuracy': accuracy,
            'loss': loss
        }
        return loss, len(self.valloader.dataset), metrics


def client_fn(context: Context):
    cid = context.node_config['cid']
    dataset_path = path.expanduser(f"{context.node_config['dataset']}_part_{cid}")
    batch_size = context.run_config['batch_size']
    local_epochs = context.run_config['local_epochs']
    learning_rate = context.run_config['learning_rate']
    trainloader, valloader = load_dataset(dataset_path, batch_size)

    # Set up the Config for wandb
    config = context.run_config.copy()
    config['cid'] = cid
    config['run_id'] = context.run_id
    enable_wandb = context.run_config['enable_client_wandb']
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate, enable_wandb, config).to_client()

app = ClientApp(client_fn=client_fn, mods=[comm_time_mod])
