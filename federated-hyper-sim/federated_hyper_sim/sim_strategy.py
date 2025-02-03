import logging
from datetime import datetime

import wandb
from flwr.common import Parameters, FitIns
from flwr.common.typing import UserConfig
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ENABLE_EARLY_STOPPING = False

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
        self.tshark_process = None # REMOVE
        self.early_stop = False # REMOVE
        self.best_loss = float('inf') # REMOVE
        self.patience = 3 # REMOVE

        if enable_wandb:
            logger.info('Enabling wandb...')
            # Login to wandb using API key.
            wandb.login(key=run_config['wandb_api_key'])
            self.config.pop('wandb_api_key')
            # Initialize W&B project
            self._init_wandb_project()

        # Initialize dict to store all results
        self.results = {}

    def _init_wandb_project(self):
        wandb.init(project='hyperparam_tuning',
                   name=f'{self.init_time}-ServerApp',
                   config=self.config)

    def _log_results(self, server_round, results):
        if self.enable_wandb:
            wandb.log(results, step=server_round)
            print('Logged results to wandb...')

    # KEEP METHOD, PERFORMS WANDB LOGGING
    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        self._log_results(server_round, metrics)
        if ENABLE_EARLY_STOPPING:
            if metrics['val_loss'] < self.best_loss:
                self.best_loss = metrics['val_loss']
                self.patience = 5
            else:
                self.patience -= 1
                if self.patience == 0:
                    self.early_stop = True
        return params, metrics

    # REMOVE METHOD
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> list[tuple[ClientProxy, FitIns]]:
        if self.early_stop:
            return []
        client_fitins_list = super().configure_fit(server_round, parameters, client_manager)
        update_client_fitins = []
        for client, fit_ins in client_fitins_list:
            updated_config = fit_ins.config.copy()  # Make a copy of the existing config
            updated_fit_ins = FitIns(fit_ins.parameters, updated_config)
            update_client_fitins.append((client, updated_fit_ins))
        return update_client_fitins
