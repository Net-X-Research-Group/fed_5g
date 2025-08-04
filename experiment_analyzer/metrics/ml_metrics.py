from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd

from metrics.base import Metric

from data_models import Trial, Configuration


class MLMetric(Metric):
    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name
        self._json_key = self._get_json_key(metric_name)

    @property
    def name(self) -> str:
        return self._metric_name

    @staticmethod
    def _get_json_key(metric_name: str) -> str:
        matches = {
            'validation_accuracy': 'val_acc',
            'validation_loss': 'val_loss',
            'train_accuracy': 'train_acc',
            'train_loss': 'train_loss',
            'avg_trainloss': 'avg_trainloss'
        }
        return matches.get(metric_name, metric_name)


    def extract_from_trial(self, trial: Trial):
        individual = self._extract_metric_from_individual(trial=trial, json_key=self._json_key)
        aggregated = self._extract_metric_from_aggregated(trial=trial, json_key=self._json_key)

        return individual, aggregated

    def aggregate_across_trials(self, configuration: Configuration, trial_data: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> Optional[pd.DataFrame]:
        print('aggregating across trials')

    def aggregate_across_configs(self, config_dfs: Dict[str, pd.DataFrame], experiment_output_path: Path) -> Optional[pd.DataFrame]:
        pass

    def visualize_trial(self, data: Optional[pd.DataFrame], figure_path: Path) -> None:
        pass

    def visualize_across_configs(self, dfs: Dict[str, pd.DataFrame], output_path_str: str) -> None:
        pass

    def visualize_single_config(self, df: pd.DataFrame, output_path_str: str) -> None:
        pass
