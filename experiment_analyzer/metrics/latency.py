import re
from os import listdir
from os.path import join
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import logging_setup
from metrics.base import Metric
from plotting_util import setup_plotting, remove_underscore
from data_models import Trial, Configuration

logger = logging_setup.setup_logging('debug') # Custom logging setup for the module


class LatencyMetric(Metric):
    @property
    def name(self) -> str:
        """
        Returns the name of the metric.
        """
        return "communication_time"

    def extract_from_trial(self, trial: Trial) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts the uplink and downlink latency data from the trial directory .csv file.
        FileIO: Outputs two csv files in the trial output path, one for uplink and one for downlink.
        Parameters:
            trial (Trial): The trial object containing the path to the trial data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing uplink and downlink latencies respectively.
        """
        files = [f for f in listdir(trial.path) if f.startswith('latency')]
        if len(files) == 0:
            logger.error(f'No latency files found in trial: {trial}')
            return pd.DataFrame(), pd.DataFrame()
        results = {}
        for file in files:
            cid = re.search(r'latency_\d+_CID(\d+)', file) # Filename Format: latency_xxxx_CIDx.csv
            if not cid:
                continue
            cid = cid.group(1)
            results[f'CID_{cid}'] = pd.read_csv(join(trial.path, file), names=['Round', 'Downlink', 'Uplink'])

        # Sort, separate, and output
        uplink_df = pd.DataFrame()
        downlink_df = pd.DataFrame()

        for cid, df in results.items():
            uplink_df[cid] = df['Uplink']
            downlink_df[cid] = df['Downlink']

        output_dir = trial.get_output_path()

        # Sort the df's by CID
        uplink_df = uplink_df.reindex(sorted(uplink_df.columns), axis=1)
        downlink_df = downlink_df.reindex(sorted(uplink_df.columns), axis=1)

        uplink_df.to_csv(output_dir / f'uplink_{self.name}.csv', index=False)
        downlink_df.to_csv(output_dir / f'downlink_{self.name}.csv', index=False)

        return uplink_df, downlink_df

    def aggregate_across_trials(self, configuration: Configuration,
                                trial_data: List[Tuple[pd.DataFrame, pd.DataFrame]]) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Aggregates uplink and downlink latencies across multiple trials within a configuration.
        FileIO: two csv files in the configuration output path, one for uplink and one for downlink.
        Parameters:
            configuration (Configuration): The configuration object.
            trial_data (List[Tuple[pd.DataFrame, pd.DataFrame]]): List of tuples containing uplink and downlink DataFrames from each trial.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing aggregated uplink and downlink latencies respectively.
        """
        # Parse
        uplinks, downlinks = self._parse_tuple(trial_data)

        # Modify
        uplink_aggregated = pd.concat(uplinks, axis=1).T.groupby(level=0).mean().T
        downlink_aggregated = pd.concat(downlinks, axis=1).T.groupby(level=0).mean().T

        # Output
        output_dir = configuration.get_output_path()
        uplink_aggregated.to_csv(output_dir / f'uplink_aggregated_{self.name}.csv', index=False)
        downlink_aggregated.to_csv(output_dir / f'downlink_aggregated_{self.name}.csv', index=False)

        return uplink_aggregated, downlink_aggregated

    def aggregate_across_configs(self, config_dfs: Dict[str, pd.DataFrame], experiment_output_path: Path) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Aggregates uplink and downlink latencies across multiple configurations. The individual node times are averaged
        row wise for each config. Then returned {Config: pd.DataFrame}
        FileIO: N/A
        Parameters:
            config_dfs (Dict[str, pd.DataFrame]): A dictionary where keys are configuration names and values are tuples of uplink and downlink DataFrames.
            experiment_output_path (Path): The path to save the aggregated results.
        Returns:
            Dict: A dictionary where keys are configuration names and values are tuples of aggregated uplink and downlink DataFrames.
        """
        """WE do NOT aggregate across configurations. Here we are just combining the CID's together in each configuration.
        The individual node times are averaged row wise for each config. Then returned {Config: pd.DataFrame}"""
        configurations = config_dfs.keys()

        results = {}
        for configuration in configurations:
            uplink, downlink = config_dfs[configuration]
            uplink = uplink.mean(axis=1)
            downlink = downlink.mean(axis=1)

            results[configuration] = uplink, downlink

        return results

    def visualize_trial(self, data: Optional[pd.DataFrame], figure_path: Path) -> None:
        """
        Visualizes the uplink and downlink latencies for a single trial and configuration. (Dual Use)
        FileIO: Outputs two plots in the figure path, one for uplink and one for downlink.
        Parameters:
            data (Optional[pd.DataFrame]): A tuple containing two DataFrames, one for uplink and one for downlink latencies.
            figure_path (Path): The path to save the visualizations.
        Returns:
            None
        """
        setup_plotting()

        uplink_df, downlink_df = data
        uplink_df = remove_underscore(uplink_df)
        downlink_df = remove_underscore(downlink_df)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Uplink plot
        sns.lineplot(data=uplink_df, ax=ax1, dashes=False)
        ax1.set_title("Uplink Communication Time")
        ax1.set_ylabel("Communication Time (s)")
        ax1.set_xlabel("Round Number")
        ax1.legend(title="Client ID")

        # Downlink plot
        sns.lineplot(data=downlink_df, ax=ax2, dashes=False)
        ax2.set_title("Downlink Communication Time")
        ax2.set_ylabel("Communication Time (s)")
        ax2.set_xlabel("Round Number")
        ax2.legend(title="Client ID")

        plt.tight_layout()
        plt.savefig(figure_path / f'{self.name}_split_linegraph')
        plt.close()

        # Boxplot for uplink latencies
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=uplink_df)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_path / f'uplink_{self.name}_box_by_cid.png')
        plt.close()


        logger.info('Latency Metric: Outputted split line graph and box plot for trial.')

        return

    def visualize_across_configs(self, dfs: Dict[str, Dict], figure_path: Path) -> None:
        """
        dfs is {Config: pd.Series} representing each configs average. This function visualizes the uplink and downlink latencies across configurations.
        FileIO: Outputs a combined box plot in the figure path.
        Parameters:
            dfs (Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]): A dictionary where keys are configuration names and values are tuples of uplink and downlink DataFrames.
            figure_path (Path): The path to save the visualizations.
        Returns:
            None
        """
        setup_plotting()

        plot_data = []
        sorted_configs = sorted(dfs.keys())

        for config_name in sorted_configs:
            uplink_df, downlink_df = dfs[config_name]
            for _, uplink_value in uplink_df.items():
                plot_data.append({
                    'Configuration': config_name,
                    'Direction': 'Uplink',
                    'Time (s)': uplink_value
                })
            for _, downlink_value in downlink_df.items():
                plot_data.append({
                    'Configuration': config_name,
                    'Direction': 'Downlink',
                    'Time (s)': downlink_value
                })

        plot_df = pd.DataFrame(plot_data)
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=plot_df, x='Configuration', y='Time (s)', hue='Direction', showfliers=False)
        plt.grid(True, alpha=0.3)
        plt.savefig(figure_path / f'{self.name}_combined_box_plot.png')
        plt.close()

        logger.info('Latency Metric: Outputted combined box plot for configurations.')
        return

    def visualize_single_config(self, df: pd.DataFrame, output_path_str: str) -> None:
        """this is identical functionality to visualize_single_trial. Pipeline visualize_single_trial instead
         WILL NOT BE IMPLEMENTED"""
        logger.warning('This function is not implemented; PASSING')
        pass
