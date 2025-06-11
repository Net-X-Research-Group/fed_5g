from pathlib import Path

import pandas as pd
from typing import List, Optional, Dict, Tuple
from experiment_analyzer.metrics.base import Metric
import matplotlib.pyplot as plt
import seaborn as sns

from experiment_analyzer.plotting_util import remove_underscore, setup_plotting
from experiment_analyzer.data_models import Trial, Configuration

class PHYMetric(Metric):
    def __init__(self, metric_name: str) -> None:
        self._metric_name = metric_name

    @property
    def name(self) -> str:
        return self._metric_name

    def extract_from_trial(self, trial: Trial) -> Tuple[pd.DataFrame, pd.DataFrame]:
        individual = self._extract_metric_from_individual(trial=trial, json_key=self.name)
        aggregated = self._extract_metric_from_aggregated(trial=trial, json_key=self.name)
        return individual, aggregated

    def aggregate_across_trials(self, configuration: Configuration, trial_data: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        individuals, aggregates = self._parse_tuple(trial_data)

        individual = pd.concat(individuals, axis=1).T.groupby(level=0).mean().T
        aggregated = pd.concat(aggregates, axis=1).T.groupby(level=0).mean().T

        return individual, aggregated

    def aggregate_across_configs(self, config_dfs: Dict[str, pd.DataFrame], experiment_output_path: Path) -> Optional[pd.DataFrame]:
        pass

    def visualize_trial(self, data: Optional[pd.DataFrame], figure_path: Path) -> None:
        individual, aggregated = data
        setup_plotting()
        individual = remove_underscore(individual)
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=individual)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_path / f'{self.name}_box_by_cid.png')
        plt.close()

    def visualize_across_configs(self, dfs: Dict[str, pd.DataFrame], output_path_str: str) -> None:
        pass

    def visualize_single_config(self, df: pd.DataFrame, output_path_str: str) -> None:
        pass
