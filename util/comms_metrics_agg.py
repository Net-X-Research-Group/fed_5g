import json
from matplotlib import pyplot as plt
import numpy as np


def plot_metrics(data: dict, direction: str):
    # Get source addresses in the data
    addresses = list(data.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), tight_layout=True)
    fig.suptitle(f'{direction} Communication Metrics', fontsize=16)

    colors = plt.colormaps.get_cmap('tab10')(np.linspace(0, 1, len(addresses)))

    for i, address in enumerate(addresses):
        address_data = data[address]

        rounds = list(address_data.keys())
        throughputs = []
        durations = []

        for round_id, round_data in address_data.items():
            throughputs.append(round_data['throughput'])
            durations.append(round_data['duration'])

        ax1.plot(rounds, durations, label=address, color=colors[i])
        ax2.plot(rounds, throughputs, label=address, color=colors[i])


    ax1.legend()
    ax1.set_title('Latency per Round')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Time (s)')

    ax2.set_title('Throughput per Round')
    ax2.legend()
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Throughput (Mbps)')


    plt.savefig(f'{direction}_communication_metrics.png')
    plt.show()

def group_by_source(data):
    """
    Transform data to group elements by their source IP. To only be used on uplink data.
    """
    result = {}
    for n, entry in data.items():
        source = entry['source']
        if source not in result:
            result[source] = {}
        result[source][n] = entry
    return result

def group_by_destination(data):
    """
    Transform data to group elements by their destination IP. To only be used on downlink data.
    """
    result = {}
    for n, entry in data.items():
        destination = entry['destination']
        if destination not in result:
            result[destination] = {}
        result[destination][n] = entry
    return result

def label_round(data):
    """
    Label each element of the data inside source/dest groups by round number
    rename each key to an incrementing number starting at 1.
    """
    result = {}
    for source, entries in data.items():
        result[source] = {}
        for i, entry in enumerate(entries.values(), 1):
            result[source][i] = entry
    return result


def main():
    uplink_file = 'http2_data_analysis_UPLINK.json'
    downlink_file = 'http2_data_analysis_DOWNLINK.json'
    with open(uplink_file, 'r') as f:
        uplink_raw = json.load(f)
    uplink_data = group_by_source(uplink_raw)

    with open(downlink_file, 'r') as f:
        downlink_raw = json.load(f)
    downlink_data = group_by_destination(downlink_raw)

    uplink_data = label_round(uplink_data)
    downlink_data = label_round(downlink_data)
    plot_metrics(uplink_data, 'Uplink')
    plot_metrics(downlink_data, 'Downlink')



if __name__ == '__main__':
    main()