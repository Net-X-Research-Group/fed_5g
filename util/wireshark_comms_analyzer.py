import argparse
import time

import numpy as np
import pandas as pd
import pyshark
import yaml
from tqdm import tqdm


def _load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def analyze_http2_data_streams(pcap_file: str, ip_addresses: dict) -> pd.DataFrame:
    """
    Analyze HTTP2 DATA stream packets and calculate latency and throughput

    Args:
        pcap_file (str): Path to the pcap file
        ip_addresses (dict): Dictionary containing downlink and uplink IP addresses of the network
    Returns:
        pandas.DataFrame: Pruned trace containing cherry-picked columns
    """
    # Initialize capture with display filter for HTTP2 DATA frames
    # Create display filter for both uplink and downlink IP addresses
    uplink_filter = ' || '.join([f'ip.addr == {ip}' for ip in ip_addresses['uplink']])
    downlink_filter = f'ip.addr == {ip_addresses["downlink"]}'
    display_filter = f'http2.type == 0 && ({uplink_filter} || {downlink_filter})'

    capture = pyshark.FileCapture(
        pcap_file,
        display_filter=display_filter,
        decode_as={'tcp.port==9092': 'http2'},
        keep_packets=False
    )

    # Lists to store packet data
    data = []
    start = time.time()
    capture.load_packets()
    print(f"Loaded {len(capture)} packets in {time.time() - start:.2f} seconds")
    # Process each packet
    for packet in tqdm(capture, desc="Processing packets", unit="packets"):
        try:
            # Extract HTTP2 and TCP information
            timestamp = float(packet.frame_info.time_epoch)
            stream_id = packet.http2.streamid  # Corresponds to the HTTP2 stream ID which is unique to each Tx group
            packet_number = int(packet.frame_info.number)  # Int
            http2_length = int(packet.http2.length)  # Bytes
            source = packet.get_multiple_layers('ip')[-1].src
            destination = packet.get_multiple_layers('ip')[-1].dst
            route = (source, destination)
            # Filter packets based on size and direction
            if any(ip in route for ip in ip_addresses['uplink']) and ip_addresses['downlink'] in route:
                direction = 'uplink' if source in ip_addresses['uplink'] else 'downlink'
                if http2_length >= 100:
                    data.append({
                        'packet_number': packet_number,
                        'timestamp': timestamp,
                        'stream_id': stream_id,
                        'tcp_bytes': http2_length,
                        'source': source,
                        'destination': destination,
                        'direction': direction
                    })

        except AttributeError as e:
            print(f"Error processing packet: {e}")
            continue
    capture.close()
    # Create DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("No HTTP2 DATA frames found in capture")

    consolidated_df = df.groupby(['stream_id', 'source', 'destination', 'direction']).agg(
        start_time=('timestamp', 'first'),
        end_time=('timestamp', 'last'),
        total_bytes=('tcp_bytes', 'sum'),
        total_packets=('packet_number', 'count')
    ).reset_index()
    consolidated_df['total_bytes'] = consolidated_df['total_bytes'] * 8 / 1e6  # Convert to Mbits from Bytes
    consolidated_df['duration'] = consolidated_df['end_time'] - consolidated_df['start_time']
    consolidated_df['throughput'] = (consolidated_df['total_bytes'] / consolidated_df['duration'])  # Mbps
    consolidated_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    consolidated_df.dropna(inplace=True)
    return consolidated_df


def save_results_to_json(data, output_file, config):
    """
    Save analysis results to a CSV file

    Args:
        data (pandas.DataFrame): Analysis results
        output_file (str): Output CSV file path
        config (dict): Dictionary containing downlink and uplink IP addresses of the network
    """

    downlink_df = data[data['source'] == config['network']['downlink']]
    uplink_df = data[data['destination'] == config['network']['downlink']]

    downlink_df.transpose().to_json(f'{output_file}_DOWNLINK.json', index=False)
    uplink_df.transpose().to_json(f'{output_file}_UPLINK.json', index=False)
    data.transpose().to_json(f'{output_file}.json', index=False)
    print("Results exported to JSON")


def main(pcap_file, config):
    # File paths
    output_file = "http2_data_analysis"
    config = _load_config(config)
    network = config['network']

    print("Analyzing HTTP2 DATA streams...")
    data = analyze_http2_data_streams(pcap_file, network)

    # Save results to CSV
    print(f"Saving results to {output_file}")
    save_results_to_json(data, output_file, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pcap_file", help="Path to the pcap file")
    parser.add_argument("-c", "--config", help="Path to the config file")

    args = parser.parse_args()

    main(args.pcap_file, args.config)
