import json
import re
from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from experiment_analyzer import process_trial_latency


def plot_latency_histograms(non_tshark, tshark, output_dir):
    """Create histogram subplots for each CID's latencies"""
    num_cids = len(non_tshark)
    fig, axs = plt.subplots(num_cids, 2, figsize=(12, 4 * num_cids))
    fig2, axs2 = plt.subplots(num_cids, 2, figsize=(12, 4 * num_cids))

    for idx, (cid, df) in enumerate(non_tshark.items()):
        # Downlink histogram
        axs[idx, 0].hist(df['Downlink'], alpha=0.75, color='blue')
        axs[idx, 0].set_title(f'{cid.replace("_", " ")} - Non-Tshark')
        axs[idx, 0].set_xlabel('Latency (s)')
        axs[idx, 0].set_ylabel('Frequency')
        axs[idx, 0].grid(True, alpha=0.3)

        # Uplink histogram
        axs2[idx, 0].hist(df['Uplink'], alpha=0.75, color='blue')
        axs2[idx, 0].set_title(f'{cid.replace("_", " ")} - Non-Tshark')
        axs2[idx, 0].set_xlabel('Latency (s)')
        axs2[idx, 0].set_ylabel('Frequency')
        axs2[idx, 0].grid(True, alpha=0.3)
    
    for idx, (cid, df) in enumerate(tshark.items()):
        # Downlink histogram
        axs[idx, 1].hist(df['Downlink'], alpha=0.75, color='green')
        axs[idx, 1].set_title(f'{cid.replace("_", " ")} - Tshark')
        axs[idx, 1].set_xlabel('Latency (s)')
        axs[idx, 1].set_ylabel('Frequency')
        axs[idx, 1].grid(True, alpha=0.3)

        # Uplink histogram
        axs2[idx, 1].hist(df['Uplink'], alpha=0.75, color='green')
        axs2[idx, 1].set_title(f'{cid.replace("_", " ")} - Tshark')
        axs2[idx, 1].set_xlabel('Latency (s)')
        axs2[idx, 1].set_ylabel('Frequency')
        axs2[idx, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Downlink')
    fig2.suptitle('Uplink')

    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(join(output_dir, 'tshark_comparison_downlink_histograms'))
    fig2.savefig(join(output_dir, 'tshark_comparison_uplink_histograms'))


def process(trials, input_path):
    data = {}

    for trial in trials:
        trial_path = join(input_path, trial)
    
        trial_results = process_trial_latency(trial_path)
        for cid, df in trial_results.items():
            if cid not in data:
                data[cid] = []
            data[cid].append(df)
    
    agg_data = {}
    for cid, dfs in data.items():
        agg_df = pd.concat(dfs, axis=1).T.groupby(level=0).mean().T.drop('Round', axis=1)
        agg_data[cid] = agg_df
    agg_data = dict(sorted(agg_data.items(), key=lambda x: int(x[0].split('_')[1]))) # Sort the dict by CID

    return agg_data


def main(input_path: str) -> None:
    trial_dirs = [d for d in listdir(input_path) if isdir(join(input_path, d))]
    
    non_tshark_trials = [trial for trial in trial_dirs if 'TSHARK' not in trial]
    tshark_trials = [trial for trial in trial_dirs if 'TSHARK' in trial]

    non_tshark_data = process(non_tshark_trials, input_path)
    tshark_data = process(tshark_trials, input_path)

    plot_latency_histograms(non_tshark_data, tshark_data, input_path)

if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)