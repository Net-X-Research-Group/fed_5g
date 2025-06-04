from os import listdir
from os.path import join
from typing import List
import re
import pandas as pd

from experiment_analyzer.metrics.base import Metric


class LatencyMetric(Metric):
    @property
    def name(self) -> str:
        return "latency"

    def extract_from_trial(self, trial_path: str) -> List[pd.DataFrame]:
        files = [f for f in listdir(trial_path) if f.startswith('latency')]
        results = {}
        for file in files:
            cid = re.search(r'latency_\d+_CID(\d+)', file)
            if not cid:
                continue
            cid = cid.group(1)
            results[f'CID_{cid}'] = pd.read_csv(join(trial_path, file), names=['Round', 'Downlink', 'Uplink'])

        uplink_df = pd.DataFrame()
        downlink_df = pd.DataFrame()

        for cid, df in results.items():
            uplink_df[cid] = df['Uplink']
            downlink_df[cid] = df['Downlink']

        uplink_df.to_csv()

        return [uplink_df, downlink_df]