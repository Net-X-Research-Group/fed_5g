import json
import logging
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg
import os
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MetricsFedAvg(FedAvg):
    def __init__(self, run_config: UserConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indiv_training_time = []
        self.indiv_fit_time = []
        self.results = {}

    def _store_results(self, results_dict):
        print('SAVING THE STUFF')
        with open(f"{os.path.expanduser('~/results.json')}", "w", encoding="utf-8") as f:
            json.dump(self.results, f)


    # Define metric aggregation function
    def fit_metrics(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(self, server_round, results, failures)

        # Store results and log
        self._store_results(metrics)

        return params, metrics



