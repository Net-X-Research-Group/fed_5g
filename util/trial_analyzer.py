from os import listdir
from os.path import join, isfile, isdir
import re
import pandas as pd
import matplotlib.pyplot as plt

def process_trial_latency(path: str):
    files = [f for f in listdir(path) if f.startswith('latency')]
    results = {}
    for file in files:
        cid = re.search(r'latency_\d+_CID(\d+)', file)
        if not cid:
            continue
        cid = cid.group(1)
        results[f'CID_{cid}'] = pd.read_csv(join(path, file), names=['Round', 'Downlink', 'Uplink'])
    return results


def plot_latency_histograms(agg_latencies, output_dir):
    """Create histogram subplots for each CID's latencies"""
    num_cids = len(agg_latencies)
    fig, axs = plt.subplots(num_cids, 2, figsize=(12, 4 * num_cids))

    for idx, (cid, df) in enumerate(agg_latencies.items()):
        # Downlink histogram
        axs[idx, 0].hist(df['Downlink'], alpha=0.75, color='blue')
        axs[idx, 0].set_title(f'{cid.replace("_", " ")} - Downlink Latency')
        axs[idx, 0].set_xlabel('Latency (ms)')
        axs[idx, 0].set_ylabel('Frequency')
        axs[idx, 0].grid(True, alpha=0.3)

        # Uplink histogram
        axs[idx, 1].hist(df['Uplink'], alpha=0.75, color='green')
        axs[idx, 1].set_title(f'{cid.replace("_", " ")} - Uplink Latency')
        axs[idx, 1].set_xlabel('Latency (ms)')
        axs[idx, 1].set_ylabel('Frequency')
        axs[idx, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    trial_dirs = [d for d in listdir(input_dir) if isdir(join(input_dir, d))]

    latencies = {}

    for trial_dir in trial_dirs:
        trial_path = join(input_dir, trial_dir)
        try:
            trial_results = process_trial_latency(trial_path)
            for cid, df in trial_results.items():
                if cid not in latencies:
                    latencies[cid] = []
                latencies[cid].append(df)
        except Exception as e:
            print(f"Error processing {trial_dir}: {e}")

    agg_latencys = {}
    for cid, dfs in latencies.items():
        agg_df = pd.concat(dfs, axis=1).groupby(level=0, axis=1).mean().drop('Round', axis=1)
        agg_latencys[cid] = agg_df

    plot_latency_histograms(agg_latencys, input_dir)
    print(latencies)




