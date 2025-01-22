import json
import logging
import os
import time
from datetime import datetime

import wandb
from flwr.common import Parameters, FitIns
from flwr.common.typing import UserConfig
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import federated_application.tshark_measurements as tshark_measurements
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_NAME = "Pytorch-5G-FLWR-CIFAR10"
ENABLE_WIRESHARK = False

class MetricsFedAvg(FedAvg):
    def __init__(self, run_config: UserConfig, run_id, enable_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = run_config
        self.num_rounds = run_config['rounds']
        self.epochs = run_config['local_epochs']
        self.clients = run_config['min_num_clients']
        self.batch_size = run_config['batch_size']
        self.init_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.run_id = run_id
        self.enable_wandb = enable_wandb
        self.tshark_process = None
        self.early_stop = False
        self.best_loss = float('inf')
        self.patience = 3
        self.dir_name = f'server.{self.clients}C.{self.epochs}E.{self.batch_size}B.{self.num_rounds}R-{self.init_time}'

        if enable_wandb:
            logger.info('Enabling wandb...')
            # Login to wandb using API key.
            wandb.login(key=run_config['wandb_api_key'])
            self.config.pop('wandb_api_key')
            # Initialize W&B project
            self._init_wandb_project()

        # Initialize dict to store all results
        self.results = {}
        self.individual_metrics = {}

        # Create logging directory
        try:
            os.mkdir(os.path.expanduser(f'~/{self.dir_name}'))  # Path should never exist
            logger.info(f'Directory {os.path.expanduser(self.dir_name)} created.')
        except FileExistsError:
            logger.info(f'Directory {os.path.expanduser(self.dir_name)} already exists.')


        # Start Tshark
        if ENABLE_WIRESHARK:
            try:
                self.tshark_process = tshark_measurements.start_tshark(self.dir_name)
                logger.info("Tshark started.")
            except Exception as e:
                logger.error(f"Error starting Tshark: {e}")

    def _init_wandb_project(self):
        wandb.init(project=PROJECT_NAME,
                   group=str(self.run_id),
                   name=f'{self.init_time}-ServerApp',
                   config=self.config)

    def _write_logs(self):
        with open(
                f"{os.path.expanduser(f'~/{self.dir_name}/agg_metrics.json')}",
                "w") as f1:
            json.dump(self.results, f1)
        with open(
                f"{os.path.expanduser(f'~/{self.dir_name}/individual_metrics.json')}",
                "w") as f2:
            json.dump(self.individual_metrics, f2)

    def _log_results(self, server_round, results):
        self.individual_metrics[server_round] = results.pop('individual_metrics')
        self.results[server_round] = results
        if self.enable_wandb:
            wandb.log(results, step=server_round)
        if server_round == self.num_rounds or self.early_stop:
            if ENABLE_WIRESHARK:
                try:
                    tshark_measurements.stop_tshark(self.tshark_process)
                    logger.info("Tshark stopped.")
                except Exception as e:
                    logger.error(f"Error stopping Tshark: {e}")


    # Define metric aggregation function
    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        self._log_results(server_round, metrics)
        if metrics['val_loss'] < self.best_loss:
            self.best_loss = metrics['val_loss']
            self.patience = 3
        else:
            self.patience -= 1
            if self.patience == 0:
                self.early_stop = True
        return params, metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[tuple[ClientProxy, FitIns]]:
        if self.early_stop:
            return []
        if server_round == self.num_rounds:
            logger.info(f"Early stopping at round {server_round} due to loss threshold.")
            self._write_logs()
        client_fitins_list = super().configure_fit(server_round, parameters, client_manager)
        update_client_fitins = []
        for client, fit_ins in client_fitins_list:
            updated_config = fit_ins.config.copy()  # Make a copy of the existing config
            updated_config["server_timestamp"] = time.time()
            updated_fit_ins = FitIns(fit_ins.parameters, updated_config)
            update_client_fitins.append((client, updated_fit_ins))
        return update_client_fitins
