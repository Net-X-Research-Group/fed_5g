# This is the main pipeline processor for the experiment analyzer.

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from experiment_analyzer.metrics.base import Metric
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trial:
    """ This represents a single trial within a configuration."""
    path: Path

    @property
    def identifier(self) -> str:
        return self.path.name

    def get_output_path(self) -> Path:
        """Returns the output path for this trial."""
        output_path = self.path / "output"
        output_path.mkdir(exist_ok=True)
        return output_path

    def get_figure_path(self) -> Path:
        figure_path = self.path / "figures"
        figure_path.mkdir(exist_ok=True)
        return figure_path

    def process_trial(self, output: Path, metrics: List[Metric]) -> Dict[str, pd.DataFrame]:
        results = {}
        for metric in metrics:
            df = metric.extract_from_trial(self)
            metric.visualize_trial(df)
            results[metric.name] = df
        return results

@dataclass
class Configuration:
    """This represents a configuration, containing multiple trials, within an experiment."""
    path: Path

    @property
    def identifier(self) -> str:
        """Gets the name of the trial directory"""
        return self.path.name

    def get_output_path(self) -> Path:
        """Returns the output path for this trial."""
        output_path = self.path / "output"
        output_path.mkdir(exist_ok=True)
        return output_path

    def get_figure_path(self) -> Path:
        figure_path = self.path / "figures"
        figure_path.mkdir(exist_ok=True)
        return figure_path

    def get_trials(self) -> List[Trial]:
        """Get all trials inside a configuration
            Returns: A list of trials"""
        return [Trial(item) for item in self.path.iterdir() if item.is_dir() and item not in ['output', 'figures']]

@dataclass
class Experiment:
    """This represents an experiment, it contains multiple configurations."""
    path: Path

    @property
    def identifier(self) -> str:
        return self.path.name

    def get_output_path(self) -> Path:
        """Returns the output path for this trial."""
        output_path = self.path / "output"
        output_path.mkdir(exist_ok=True)
        return output_path

    def get_figure_path(self) -> Path:
        figure_path = self.path / "figures"
        figure_path.mkdir(exist_ok=True)
        return figure_path

    def get_configurations(self) -> List[Configuration]:
        """Returns a list of all configurations"""
        return [Configuration(item) for item in self.path.iterdir() if item.is_dir() and item not in ['output', 'figures']]

