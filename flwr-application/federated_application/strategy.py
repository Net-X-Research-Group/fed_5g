import json
import logging
import os

from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MetricsFedAvg(FedAvg):
    def __init__(self, run_config: UserConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = dict()
        self.num_rounds = run_config['rounds']

    def _log_results(self, server_round, results):
        self.results[server_round] = results

        if server_round == self.num_rounds:
            with open(f"{os.path.expanduser('~/results.json')}", "w") as f:
                json.dump(self.results, f)

    # Define metric aggregation function
    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        self._log_results(server_round, metrics)
        return params, metrics


