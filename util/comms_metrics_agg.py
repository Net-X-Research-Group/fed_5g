import json

from matplotlib import pyplot as plt


def get_run_time():
    with open('config.json', 'r') as f:
        config_metadata = json.load(f)
    with open('../elapsed.json') as f:
        trials_metadata = json.load(f)
    run_id = str(config_metadata['run_id'])
    time = trials_metadata[run_id]
    ftr = [3600, 60, 1]
    elapsed = sum([a * b for a, b in zip(ftr, map(int, time.split(':')))])
    return elapsed


def plot_metrics(data: dict, direction: str, elapsed: int):
    # Get source addresses in the data
    addresses = list(data.keys())

    # Plot latency
    fig_latency, axs_latency = plt.subplots(len(addresses), 1, figsize=(10, 5 * len(addresses)), tight_layout=True)
    fig_latency.suptitle(f'{direction} Latency, Run Time: {elapsed} seconds', fontsize=16)

    # Plot throughput
    fig_throughput, axs_throughput = plt.subplots(len(addresses), 1, figsize=(10, 5 * len(addresses)),
                                                  tight_layout=True)
    fig_throughput.suptitle(f'{direction} Throughput', fontsize=16)

    for i, address in enumerate(addresses):
        address_data = data[address]
        rounds = list(address_data.keys())
        throughputs = []
        durations = []

        for round_id, round_data in address_data.items():
            throughputs.append(round_data['throughput'])
            durations.append(round_data['latency'])

        # Plot latency
        axs_latency[i].hist(durations, label=address)
        axs_latency[i].set_title(f'{address}')
        axs_latency[i].set_xlabel('Time (s)')
        axs_latency[i].set_ylabel('Frequency')
        axs_latency[i].legend()

        # Plot throughput
        axs_throughput[i].hist(throughputs, label=address)
        axs_throughput[i].set_title(f'{address}')
        axs_throughput[i].set_xlabel('Throughput (Mbps)')
        axs_throughput[i].set_ylabel('Frequency')
        axs_throughput[i].legend()

    fig_latency.savefig(f'{direction}_latency.png')
    fig_throughput.savefig(f'{direction}_throughput.png')

    plt.close(fig_latency)
    plt.close(fig_throughput)


def group_by_source(data):
    """
    Transform data to group elements by their source IP. To only be used on uplink data.
    """
    result = {}
    for n, entry in data.items():
        source = entry['source_ip']
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
        destination = entry['destination_ip']
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
    elapsed = get_run_time()
    print('The trial took:', elapsed, 'seconds')
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
    plot_metrics(uplink_data, 'Uplink', elapsed)
    plot_metrics(downlink_data, 'Downlink', elapsed)


if __name__ == '__main__':
    main()
