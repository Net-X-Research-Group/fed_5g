import json
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd

from experiment_analyzer import plot_latency_histograms, save_latency_histograms
from util.experiment_analyzer import process_trial_latency, _aggregate_metrics
import numpy as np
import seaborn as sns

def _read_json(file):
    with open(file, 'r') as f:
        raw = json.load(f)
    return raw

def _extract_latencies(tshark_data: dict, direction: str) -> pd.DataFrame:
    source_ips = sorted(list(set(item['destination_ip'] for item in tshark_data.values()))) if direction == 'downlink' else sorted(list(set(item['source_ip'] for item in tshark_data.values())))

    # Define mapping from IP to CID
    ip_to_cid = {
        '129.105.6.17': 'CID1',
        '129.105.6.18': 'CID2',
        '129.105.6.19': 'CID3'
    }

    latency_data = {ip_to_cid[ip]: [] for ip in source_ips}

    for item in tshark_data.values():
        source_ip = item['destination_ip'] if direction == 'downlink' else item['source_ip']
        latency = item['latency']
        cid = ip_to_cid[source_ip]
        latency_data[cid].append(latency)

    max_length = max(len(latencies) for latencies in latency_data.values())

    for ip in source_ips:
        cid = ip_to_cid[ip]
        if len(latency_data[cid]) < max_length:
            latency_data[cid].extend([np.nan] * (max_length - len(latency_data[cid])))

    df = pd.DataFrame(latency_data)

    return df

def main(input_path: str) -> None:
    uplink_file = _read_json(join(input_path, 'http2_data_analysis_UPLINK.json'))
    downlink_file = _read_json(join(input_path, 'http2_data_analysis_DOWNLINK.json'))

    tshark_uplink = _extract_latencies(uplink_file, 'uplink')
    tshark_downlink = _extract_latencies(downlink_file, 'downlink')

    tshark_uplink['$t_u$'] = tshark_uplink.mean(axis=1)
    tshark_downlink['$t_d$'] = tshark_downlink.mean(axis=1)


    flower_latency = process_trial_latency(input_path)
    flower_latency = pd.concat(flower_latency.values(), axis=1).T.groupby(level=0).mean().T
    flower_latency['$t_d$'] = flower_latency['Downlink']
    flower_latency['$t_u$'] = flower_latency['Uplink']

    # Create plots with 2 rows, 1 column
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot uplink data (top plot)
    sns.histplot(tshark_uplink['$t_u$'].dropna(), ax=axs[0], color='blue', label='Tshark')
    sns.histplot(flower_latency['$t_u$'].dropna(), ax=axs[0], color='red', label='Flower')
    #axs[0].set_title('Uplink Latency Distribution')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()

    # Plot downlink data (bottom plot)
    sns.histplot(tshark_downlink['$t_d$'].dropna(), ax=axs[1], color='blue', label='Tshark')
    sns.histplot(flower_latency['$t_d$'].dropna(), ax=axs[1], color='red', label='Flower')
    #axs[1].set_title('Downlink Latency Distribution')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    plt.tight_layout()

    save_latency_histograms(input_path, fig)

    # Calculate average differences between methods
    tshark_uplink_mean = tshark_uplink['$t_u$'].mean()
    flower_uplink_mean = flower_latency['$t_u$'].mean()
    uplink_diff = abs(tshark_uplink_mean - flower_uplink_mean)

    tshark_downlink_mean = tshark_downlink['$t_d$'].mean()
    flower_downlink_mean = flower_latency['$t_d$'].mean()
    downlink_diff = abs(tshark_downlink_mean - flower_downlink_mean)

    # Calculate percentage differences
    uplink_percent_diff = (uplink_diff / tshark_uplink_mean) * 100
    downlink_percent_diff = (downlink_diff / tshark_downlink_mean) * 100

    # Print the results to console as well
    print(f"Uplink - Tshark mean: {tshark_uplink_mean:.5f}s, Flower mean: {flower_uplink_mean:.5f}s")
    print(f"Uplink - Absolute difference: {uplink_diff:.5f}s ({uplink_percent_diff:.2f}%)")
    print(f"Downlink - Tshark mean: {tshark_downlink_mean:.5f}s, Flower mean: {flower_downlink_mean:.5f}s")
    print(f"Downlink - Absolute difference: {downlink_diff:.5f}s ({downlink_percent_diff:.2f}%)")


if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)