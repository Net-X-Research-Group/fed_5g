"""pytorch-example: A Flower / PyTorch app."""

import json
from logging import INFO
import time
import torch
from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

PROJECT_NAME = "NU_ANL_FedAvg_5GFL"

class MetricsFedAvg(FedAvg):
    def __init__(self, run_config: UserConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indiv_training_time = []
        self.indiv_fit_time = []
        self.results = {}

    # Define metric aggregation function
    def fit_metrics(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(self, server_round, results, failures)
        return params, metrics