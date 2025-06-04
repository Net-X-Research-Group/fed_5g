import re
from os import listdir
from os.path import join
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiment_analyzer.metrics.base import Metric
from experiment_analyzer.plotting_util import setup_plotting, remove_underscore
from experiment_analyzer.processor import Trial, Configuration


class LatencyMetric(Metric):
    @property
    def name(self) -> str:
        return "communication_time"

    def extract_from_trial(self, trial: Trial) -> Tuple[pd.DataFrame, pd.DataFrame]:
        files = [f for f in listdir(trial.path) if f.startswith('latency')]
        results = {}
        for file in files:
            cid = re.search(r'latency_\d+_CID(\d+)', file)
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

    def aggregate_across_trials(self, configuration: Configuration) -> Optional[pd.DataFrame]:
        # TODO
        pass

    def aggregate_across_configs(self, config_dfs: Dict[str, pd.DataFrame], experiment_output_path: Path) -> Optional[
        pd.DataFrame]:
        # TODO
        pass

    def visualize_trial(self, data: Optional[pd.DataFrame], figure_path: Path) -> None:
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
        plt.savefig(figure_path / f'{self.name}_split_linegraph.png')
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=uplink_df)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figure_path / f'{self.name}_box_by_cid.png')
        plt.close()

    def visualize_across_configs(self, dfs: Dict[str, pd.DataFrame], output_path_str: str) -> None:
        # TODO
        pass

    def visualize_single_config(self, df: pd.DataFrame, output_path_str: str) -> None:
        # TODO
        pass
