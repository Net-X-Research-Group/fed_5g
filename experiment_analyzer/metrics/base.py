from abc import ABC, abstractmethod
from typing import Optional, List, Dict, TYPE_CHECKING, Tuple
from pathlib import Path
from json import load

import pandas as pd

if TYPE_CHECKING:
    from experiment_analyzer.processor import Trial, Configuration


class Metric(ABC):
    """ Base class for all metrics used in the experiment analyzer."""

    @property
    def name(self) -> str:
        """Returns the name of the metric."""
        return self.__class__.__name__.replace('Metric', '').lower()

    @abstractmethod
    def extract_from_trial(self, trial: "Trial") -> Optional[pd.DataFrame]:
        """Extracts metric from raw data in single trial. Should return a DataFrame, saving is optional here."""
        pass

    @abstractmethod
    def aggregate_across_trials(self, configuration: "Configuration", trial_data: Dict) -> Optional[pd.DataFrame]:
        """Aggregate a metric across multiple trials inside a configuration. Saves to output dir in config path and returns aggregated DataFrame."""
        pass

    @abstractmethod
    def aggregate_across_configs(self, config_dfs: Dict[str, pd.DataFrame], experiment_output_path: Path) -> Optional[pd.DataFrame]:
        """
        Aggregate/combine metrics across configurations for comparison. Saves to output dir in experiment path.
        Returns a DataFrame suitable for comparison plotting.
        """
        pass

    @abstractmethod
    def visualize_trial(self, data: Optional[pd.DataFrame], figure_path: Path) -> None:
        pass

    @abstractmethod
    def visualize_single_config(self, df: pd.DataFrame, output_path_str: str) -> None:
        """Visualizes a single configuration metric. Saves to output path."""
        pass

    @abstractmethod
    def visualize_across_configs(self, dfs: Dict[str, pd.DataFrame], output_path_str: str) -> None:
        """Visualizes aggregated metric across configurations. Saves to output path."""
        pass

    def _extract_metric_from_individual(self, trial: "Trial", json_key: str) -> Optional[pd.DataFrame]:
        with open(trial.path / 'individual_metrics.json', 'r') as f:
            data = load(f)
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
                    cid_values.append(data[round_num][cid][json_key])
                else:
                    cid_values.append(None)  # Handle missing data
            result[f'CID_{cid}'] = cid_values

        result.to_csv(trial.get_output_path() / f'{self.name}.csv')
        return result if not result.empty else None

    def _extract_metric_from_aggregated(self, trial: "Trial", json_key: str) -> Optional[pd.DataFrame]:
        """No CID, just round based aggregation parsing"""
        with open(trial.path / 'agg_metrics.json', 'r') as f:
            data = load(f)
        rounds = sorted(list(data.keys()), key=int)
        result = pd.DataFrame(index=rounds)

        values = []
        for round_num in rounds:
            if data[round_num]:
                values.append(data[round_num][json_key])
            else:
                values.append(None)
        result['avg'] = values
        result.to_csv(trial.get_output_path() / f'{self.name}_server_agg_.csv')
        return result if not result.empty else None

    @staticmethod
    def _parse_tuple(dataset: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[List, List]:
        """Parses a list of tuples and returns two lists containing all the data"""
        firsts = []
        seconds = []
        for data in dataset:
            first, second = data # Tuple of uplink and downlink dfs
            firsts.append(first)
            seconds.append(second)

        return firsts, seconds
