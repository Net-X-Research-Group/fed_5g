import json
import re
from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import numpy as np

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
    
    avg_uplink = np.mean(tshark_uplink)
    avg_downlink = np.mean(tshark_downlink)

    print(f'average uplink latency: {avg_uplink}')
    print(f'average downlink latency: {avg_downlink}')

    


if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)