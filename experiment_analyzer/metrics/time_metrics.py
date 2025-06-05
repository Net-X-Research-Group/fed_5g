from pathlib import Path

import pandas as pd
from typing import List, Optional, Dict
from experiment_analyzer.metrics.base import Metric

class TimeMetric(Metric):
    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name
        self._json_key = self._get_json_key(metric_name)

    @property
    def name(self) -> str:
        return self._metric_name

    @staticmethod
    def _get_json_key(metric_name: str) -> str:
        matches = {
            'fake_uplink_time': 'uplink_time',
            'fake_downlink_time': 'downlink_time',
            'training_time': 'training_time',
            'train_dataset_eval_time': 'train_test_time',
            'validation_dataset_eval_time': 'val_test_time',
            'train_start_timestamp': 'train_start_time',
            'train_end_timestamp': 'train_end_time',
        }
        return matches.get(metric_name, metric_name)


    def extract_from_trial(self, trial):
        individual = self._extract_metric_from_individual(trial=trial, json_key=self._json_key)
        aggregated = self._extract_metric_from_aggregated(trial=trial, json_key=self._json_key)

        return individual, aggregated

    def aggregate_across_trials(self, trials: List[pd.DataFrame], config_output_path: Path) -> Optional[pd.DataFrame]:
        pass

    def aggregate_across_configs(self, config_dfs: Dict[str, pd.DataFrame], experiment_output_path: Path) -> Optional[pd.DataFrame]:
        pass

    def visualize_trial(self, data: Optional[pd.DataFrame], figure_path: Path) -> None:
        pass

    def visualize_across_configs(self, dfs: Dict[str, pd.DataFrame], output_path_str: str) -> None:
        pass

    def visualize_single_config(self, df: pd.DataFrame, output_path_str: str) -> None:
        pass
