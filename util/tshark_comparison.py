import json
import re
from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np

from experiment_analyzer import plot_latency_histograms, save_latency_histograms

def _read_json(file):
    with open(file, 'r') as f:
        raw = json.load(f)
    return raw


def main(input_path: str) -> None:
    uplink_file = _read_json(join(input_path, 'http2_data_analysis_UPLINK.json'))
    downlink_file = _read_json(join(input_path, 'http2_data_analysis_DOWNLINK.json'))

    tshark_uplink = []
    tshark_downlink = []

    for key in uplink_file:
        tshark_uplink.append(uplink_file[key]['latency'])
    for key in downlink_file:
        tshark_downlink.append(downlink_file[key]['latency'])

    flower_uplink = pd.read_csv(join(input_path, 'uplink.csv'))
    flower_downlink = pd.read_csv(join(input_path, 'downlink.csv'))

    fig, axs = plt.subplots(2, 2, figsize=(12, 4 * 2))
    plot_latency_histograms(tshark_uplink, tshark_downlink, fig, 0, 'Wireshark')
    plot_latency_histograms(flower_uplink, flower_downlink, fig, 1, 'Flower')

    save_latency_histograms(input_path, fig)


if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)