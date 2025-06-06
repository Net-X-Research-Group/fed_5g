# This is the main pipeline processor for the experiment analyzer.
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import pandas as pd

import logging_setup
from experiment_analyzer.metrics.base import Metric

logger = logging_setup.setup_logging('info')

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
        figure_path = self.get_figure_path()
        for metric in metrics:
            df = metric.extract_from_trial(self)
            #metric.visualize_trial(df, figure_path) # TODO: UNDO COMMENT OUT, or build in switch
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
        return [Trial(item) for item in self.path.iterdir() if item.is_dir() and item.name not in ['output', 'figures']]

    def aggregate_trials(self, trial_data, metrics):
        figure_path = self.get_figure_path()
        aggregated_metrics = {}
        for metric in metrics:
            metric_data_over_trials = []
            for trial_id, trial_metrics in trial_data.items():
                if metric.name in trial_metrics:
                    metric_data_over_trials.append(trial_metrics[metric.name])

            if metric_data_over_trials:
                aggregated_df = metric.aggregate_across_trials(self, metric_data_over_trials)
                aggregated_metrics[metric.name] = aggregated_df
                #metric.visualize_trial(aggregated_df, figure_path)
        return aggregated_metrics


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
        return [Configuration(item) for item in self.path.iterdir() if
                item.is_dir() and item.name not in ['output', 'figures']]

    def process(self, configuration_data, metrics: List[Metric]):
        figure_path = self.get_figure_path()
        for metric in metrics:
            experiment_data = {}
            for configuration_id, configuration_metrics in configuration_data.items():
                if metric.name in configuration_metrics:
                    experiment_data[configuration_id] = configuration_metrics[metric.name]
            output_path = self.get_output_path()
            experiment_aggregated = metric.aggregate_across_configs(experiment_data, output_path)
            metric.visualize_across_configs(experiment_aggregated, figure_path)