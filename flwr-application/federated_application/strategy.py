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
ENABLE_EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_TOLERANCE = 0.01


class MetricsFedAvg(FedAvg):
    def __init__(self, run_config: UserConfig, run_id: int, enable_wandb: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Hyper Parameters
        self.num_rounds = run_config['rounds']
        self.epochs = run_config['local_epochs']
        self.clients = run_config['min_num_clients']
        self.batch_size = run_config['batch_size']
        self.weight_decay = run_config['weight_decay']
        self.momentum = run_config['momentum']

        # Weights & Biases and Logging Config
        self.config = run_config
        self.init_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.run_id = run_id
        self.enable_wandb = enable_wandb
        self.tshark_process = None
        self.dir_name = f'{self.run_id}'

        # Early Stopping Config
        self.early_stop = False
        self.best_loss = float('inf')
        self.patience = EARLY_STOPPING_PATIENCE

        if enable_wandb:
            logger.info('Enabling wandb...')
            # Login to wandb using API key.
            wandb.login(key=run_config['wandb_api_key'])
            self.config.pop('wandb_api_key')  # Do not push API key to W&B
            # Initialize W&B project
            self._init_wandb_project()

        # Initialize dict to store all results
        self.results = {}
        self.individual_metrics = {}

        # Create logging directory
        try:
            os.makedirs(os.path.expanduser(f'~/{self.dir_name}'), exist_ok=True)  # Path should never exist
            logger.info(f'Directory {os.path.expanduser(self.dir_name)} created.')
        except FileExistsError:
            logger.info(f'Directory {os.path.expanduser(self.dir_name)} already exists.')

        try:
            config = self.config.copy()
            config['run_id'] = self.run_id
            config['patience'] = self.patience
            config['init_time'] = self.init_time
            with open(f"{os.path.expanduser(f'~/{self.dir_name}/config.json')}", "w") as f:
                json.dump(config, f)
        except Exception as e:
            logger.error(f"Error writing config to file: {e}")

        # Start Tshark
        if ENABLE_WIRESHARK:
            try:
                self.tshark_process = tshark_measurements.start_tshark(self.dir_name)
                logger.info("Tshark started.")
            except Exception as e:
                logger.error(f"Error starting Tshark: {e}")

    def _init_wandb_project(self):
        wandb.init(project=PROJECT_NAME,
                   name=f'{self.run_id}-{self.init_time}-ServerApp',
                   config=self.config)

    def _write_logs(self):
        """
        Helper Function
        -----------------
        Writes aggregated and individual metrics to a JSON file.
        """
        os.makedirs(os.path.expanduser(f'~/{self.dir_name}'), exist_ok=True)
        with open(f"{os.path.expanduser(f'~/{self.dir_name}/agg_metrics.json')}", 'w') as f:
            json.dump(self.results, f)
        with open(f"{os.path.expanduser(f'~/{self.dir_name}/individual_metrics.json')}", 'w') as f:
            json.dump(self.individual_metrics, f)

    def _log_results(self, server_round: int, results: dict) -> None:
        """
        Helper Function
        -----------------
        Moves individual metrics to a separate dictionary and logs the aggregated results.
        Pushes the results to W&B if enabled.
        Kills Tshark process if the last round or early stopping is triggered, if enabled.

        Parameters:
            server_round (int): The current round of training.
            results (dict): The aggregated results from the clients, including individual metrics.
        """
        self.individual_metrics[server_round] = results.pop('individual_metrics')
        self.results[server_round] = results
        if self.enable_wandb:
            wandb.log(results, step=server_round)
        self._write_logs()
        if server_round == self.num_rounds or self.early_stop:
            if ENABLE_WIRESHARK:
                try:
                    tshark_measurements.stop_tshark(self.tshark_process)
                    logger.info("Tshark stopped.")
                except Exception as e:
                    logger.error(f"Error stopping Tshark: {e}")

    def aggregate_fit(self, server_round: int, results: list, failures: list) -> tuple[Parameters, dict]:
        """
        Aggregates the results from the clients and logs them. Injects early stopping logic.

        Parameters:
            server_round (int): The current round of training.
            results (list): The results from the clients.
            failures (list): The failed clients.

        Returns:
            tuple: The aggregated parameters and metrics.
        """
        params, metrics = super().aggregate_fit(server_round, results, failures)
        self._log_results(server_round, metrics)
        if ENABLE_EARLY_STOPPING:
            if metrics['val_acc'] <= 0.20 and server_round >= 15:
                self.early_stop = True
            elif metrics['val_loss'] + EARLY_STOPPING_TOLERANCE < self.best_loss:
                self.best_loss = metrics['val_loss']
                self.patience = EARLY_STOPPING_PATIENCE
            else:
                self.patience -= 1
                if self.patience == 0:
                    self.early_stop = True
            if self.enable_wandb:
                wandb.log({"patience": self.patience}, step=server_round)
        return params, metrics

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[
        tuple[ClientProxy, FitIns]]:
        """
        Configure the fit process for the next round of training. Injects the server timestamp into the client's config.
        Injects early stopping logic.
        Parameters:
            server_round (int): The current round of training.
            parameters (Parameters): The current model parameters.
            client_manager (ClientManager): The client manager object.

        Returns:
            list[tuple[ClientProxy, FitIns]]: A list of tuples containing the client and the updated FitIns object.
        """
        client_fitins_list = super().configure_fit(server_round, parameters, client_manager)

        # Early Stopping
        if self.early_stop:
            return []
        if server_round == self.num_rounds:
            self._write_logs()

        # Metrics Injection
        update_client_fitins = []
        for client, fit_ins in client_fitins_list:
            updated_config = fit_ins.config.copy()  # Make a copy of the existing config
            updated_config["server_timestamp"] = time.time()
            updated_fit_ins = FitIns(fit_ins.parameters, updated_config)
            update_client_fitins.append((client, updated_fit_ins))
        return update_client_fitins
