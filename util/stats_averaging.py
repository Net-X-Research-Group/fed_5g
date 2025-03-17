import json
import re
from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import csv


def main(input_path: str) -> None:
    experiment_dir = [d for d in listdir(input_path) if isdir(join(input_path, d))]

    time_metrics = ['Training Time', 'Train Test Time', 'Val Test Time', 'Round Time']
    data = {}

    for result in experiment_dir:
        result_path = join(input_path, result)
        result = result.replace("_", " ")

        try:
            data[result] = {}

            latency = pd.read_csv(join(result_path, 'latencies_aggregated.csv'), skiprows=1)
            data[result]['Uplink'] = pd.concat([latency[df] for df in latency if 'Uplink' in df], axis=1).mean(axis=1)
            data[result]['Uplink'].name = result
            data[result]['Downlink'] = pd.concat([latency[df] for df in latency if 'Downlink' in df], axis=1).mean(axis=1)
            data[result]['Downlink'].name = result

            for metric in time_metrics:
                filename = f'{metric.lower().replace(" ", "_")}_aggregated.csv'
                data[result][metric] = pd.read_csv(join(result_path, filename)).mean(axis=1)
                data[result][metric].name = result

            data[result]['Validation Accuracy'] = pd.read_csv(join(result_path, 'avg_val_acc.csv')).mean(axis=1)
            data[result]['Validation Loss'] = pd.read_csv(join(result_path, 'avg_val_loss.csv')).mean(axis=1)

        except Exception as e:
            data.popitem()
            print(f"Error processing {result}: {e}")

    # mydict = {k: v for k, v in blahs}
    table = {result: {} for result in data}
    for result in data:
        for metric in data[result]:
            table[result][metric] = data[result][metric].mean()
        
        filename = f'{result}_stats.csv'
        with open(filename, 'w', newline='') as f:
            w = csv.DictWriter(f, table[result].keys())
            w.writeheader()
            w.writerow(table[result])
    


if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    main(input_dir)