from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from pathlib import Path # Added for type hinting

import pandas as pd

class Metric(ABC):
    " Base class for all metrics used in the experiment analyzer."
    @property
    @abstractmethod
    def name(self) -> str:
        "Returns the name of the metric."
        pass

    @abstractmethod
    def extract_from_trial(self, trial) -> Optional[pd.DataFrame]:
        "Extracts metric from raw data in single trial. Should return a DataFrame, saving is optional here."
        pass

    @abstractmethod
    def aggregate_across_trials(self, trials: List[pd.DataFrame], config_output_path: Path) -> Optional[pd.DataFrame]:
        "Aggregate a metric across multiple trials inside a configuration. Saves to output dir in config path and returns aggregated DataFrame."
        pass

    @abstractmethod
    def aggregate_across_configs(self, config_dfs: Dict[str, pd.DataFrame], experiment_output_path: Path) -> Optional[pd.DataFrame]:
        """
        Aggregate/combine metrics across configurations for comparison. Saves to output dir in experiment path.
        Returns a DataFrame suitable for comparison plotting.
        """
        pass

    @abstractmethod
    def visualize_trial(self, data: Optional[pd.DataFrame]) -> None:
        pass

    @abstractmethod
    def vizualize_single_config(self, df: pd.DataFrame, output_path_str: str) -> None:
        "Visualizes a single configuration metric. Saves to output path."
        pass

    @abstractmethod
    def vizualize_across_configs(self, dfs: Dict[str, pd.DataFrame], output_path_str: str) -> None:
        "Visualizes aggregated metric across configurations. Saves to output path."
        pass
