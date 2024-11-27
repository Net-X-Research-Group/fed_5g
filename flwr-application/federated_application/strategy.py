import json
import logging
import os

from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg
from dotenv import load_dotenv

import wandb

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_NAME = "Pytorch-5G-FLWR-CIFAR10"

class MetricsFedAvg(FedAvg):
    def __init__(self, run_config: UserConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = run_config
        self.num_rounds = run_config['rounds']
        self.epochs = run_config['local_epochs']
        self.clients = run_config['min_num_clients']
        self.batch_size = run_config['batch_size']

        # Login to wandb using API key.
        wandb.login(key=os.getenv('WANDB_API'))

        # Initialize W&B project
        self._init_wandb_project()

        # Initialize dict to store all results
        self.results = {}

    def _init_wandb_project(self):
        wandb.init(project=PROJECT_NAME, name=f'{str(self.config)}-ServerApp', config=self.config)

    def _log_results(self, server_round, results):
        self.results[server_round] = results
        wandb.log(results, step=server_round)
        if server_round == self.num_rounds:
            with open(f"{os.path.expanduser(f'~/server.{self.clients}C.{self.epochs}E.{self.batch_size}B.{self.num_rounds}R.json')}", "w") as f:
                json.dump(self.results, f)

    # Define metric aggregation function
    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        self._log_results(server_round, metrics)
        return params, metrics


