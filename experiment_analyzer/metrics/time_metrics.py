from pathlib import Path

import pandas as pd
from typing import List, Optional, Dict
from metrics.base import Metric

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from plotting_util import setup_plotting, remove_underscore

from data_models import Trial, Configuration

import logging_setup
logger = logging_setup.setup_logging('debug') # Custom logging setup for the module
plt.set_loglevel('WARNING')

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
        # aggregated = self._extract_metric_from_aggregated(trial=trial, json_key=self._json_key)

        return individual#, aggregated

    def aggregate_across_trials(self, configuration: Configuration, trials: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        output_dir = configuration.get_output_path()
        
        # individual_dfs = [trial[0] for trial in trials]

        aggregated = pd.concat(trials, axis=1).T.groupby(level=0).mean().T
        aggregated.to_csv(output_dir / f'aggregated_{self.name}.csv', index=False)

        return aggregated

    def aggregate_across_configs(self, config_dfs: Dict[str, pd.DataFrame], experiment_output_path: Path) -> Optional[pd.DataFrame]:
        configurations = config_dfs.keys()

        results = {}
        for configuration in configurations:
            # individual_aggregated, early_aggregated = config_dfs[configuration]
            # individual_aggregated = individual_aggregated.mean(axis=1)
            # early_aggregated = early_aggregated.mean(axis=1)

            # results[configuration] = individual_aggregated, early_aggregated

            results[configuration] = config_dfs[configuration].mean(axis=1)

        return results

    def visualize_trial(self, data: Optional[pd.DataFrame], figure_path: Path) -> None:
        setup_plotting()

        # individual, aggregated = data
        # df = remove_underscore(individual)
        df = remove_underscore(data)

        # Line graph
        fig = plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, dashes=False)
        plt.xlabel('Round Number')
        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))
        plt.ylabel('Time (s)')
        plt.title(self._metric_name)
        plt.legend(title="Client ID")
        plt.tight_layout()
        plt.savefig(figure_path / f'{self.name}_linegraph')
        plt.close()

        # Box plot
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_path / f'{self.name}_box_by_cid.png')
        plt.close()

        logger.info(f'{self._metric_name}: Outputted line graph and box plot for trial(s).')

        return

    def visualize_across_configs(self, dfs: Dict[str, pd.DataFrame], output_path_str: str) -> None:
        setup_plotting()

        plot_data = []
        sorted_configs = sorted(dfs.keys())

        for config_name in sorted_configs:
            df = dfs[config_name]
            for _, value in df.items():
                plot_data.append({
                    'Configuration': config_name,
                    'Time (s)': value
                })

        plot_df = pd.DataFrame(plot_data)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_df, x='Configuration', y='Time (s)', showfliers=False)
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path_str / f'{self.name}_box_plot.png')
        plt.close()

        logger.info(f'{self._metric_name} Metric: Outputted box plot for configurations.')

        return

    def visualize_single_config(self, df: pd.DataFrame, output_path_str: str) -> None:
        """this is identical functionality to visualize_single_trial. Pipeline visualize_single_trial instead
         WILL NOT BE IMPLEMENTED"""
        logger.warning('This function is not implemented; PASSING')
        pass
