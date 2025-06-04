from pathlib import Path

import pandas as pd
from typing import List, Optional, Dict
from experiment_analyzer.metrics.base import Metric
import json


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
            'train_loss': 'train_loss'
        }
        return matches.get(metric_name, metric_name)


    def extract_from_trial(self, trial):
        with open(trial.path / 'individual_metrics.json', 'r') as f:
            data = json.load(f)
        cids = set()
        for round_data in data.values():
            cids.update(round_data.keys())
        cids = sorted(list(cids))

        # Get all rounds
        rounds = sorted(list(data.keys()), key=int)
        result = pd.DataFrame(index=rounds)

        # Fill in data for each CID
        for cid in cids:
            cid_values = []
            for round_num in rounds:
                if cid in data[round_num]:
                    cid_values.append(data[round_num][cid][self._json_key])
                else:
                    cid_values.append(None)  # Handle missing data
            result[f'CID_{cid}'] = cid_values

        result.to_csv(trial.get_output_path() / f'{self.name}.csv')

        return result

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
