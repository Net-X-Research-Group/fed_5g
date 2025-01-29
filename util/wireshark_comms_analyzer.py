import argparse
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from subprocess import call
import json


def _load_config(config_file) -> dict:
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def _load_json(file_path) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def _transform_pcap(pcap_file: str, display_filter: str) -> str:
    """
    Transform the pcap file into a pandas DataFrame

    Args:
        pcap_file (str): Path to the pcap file
        ip_addresses (dict): Dictionary containing downlink and uplink IP addresses of the network
    Returns:
        pandas.DataFrame: Pruned trace containing
    """
    # tshark -r output.pcapng -Y "http2.type == 0 && (ip.addr == 10.0.0.3 || ip.addr == 10.0.0.2 || ip.addr == 192.168.70.129)" -d "tcp.port==9092,http2" -T json > output.json
    output_file = 'output.json'
    command = ['tshark',
               '-o', 'tcp.desegment_tcp_streams:TRUE',
               '-r', pcap_file,
               '-J', 'tcp ip http2 frame',
               '-T', 'json',
               '-Y', display_filter,
               '-d', 'tcp.port==9092,http2']

    f = open(output_file, 'w')
    call(command, stdout=f)
    f.close()
    print('Sucessfully converted pcapng file.')
    return output_file

def analyze_data_streams(data: dict, ip_addresses: dict) -> pd.DataFrame:
    data = [d['_source']['layers'] for d in data]
    parsed = []
    for packet in tqdm(data):
        try:
            timestamp = float(packet['frame']['frame.time_epoch'])
            source_ip = str(packet['ip']['ip.src'])
            destination_ip = str(packet['ip']['ip.dst'])
            stream_id = str(packet['http2']['http2.stream']['http2.streamid'])
            packet_length = float(packet['http2']['http2.stream']['http2.length'])
            route = (source_ip, destination_ip)
            if any(ip in route for ip in ip_addresses['uplink']) and ip_addresses['downlink'] in route:
                direction = 'uplink' if source_ip in ip_addresses['uplink'] else 'downlink'
                if packet_length >= 100:
                    parsed.append({'timestamp': timestamp,
                                 'source_ip': source_ip,
                                 'destination_ip': destination_ip,
                                 'stream_id': stream_id,
                                 'packet_length': packet_length,
                                 'direction': direction})
        except ValueError as e:
            raise('Value Error:', e)
    df = pd.DataFrame(parsed).groupby(['stream_id', 'source_ip', 'destination_ip', 'direction']).agg(
        start_time=('timestamp', 'min'),
        end_time=('timestamp', 'max'),
        http_bytes=('packet_length', 'sum'),
        total_packets=('stream_id', 'count')
    ).reset_index()
    df['latency'] = df['end_time'] - df['start_time']
    df['throughput'] = ((df['http_bytes'] * 8) / df['latency']) / 1e6
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.sort_values(by='start_time').reset_index()
    return df

def save_results_to_json(data, output_file, config):
    """
    Save analysis results to a CSV file

    Args:
        data (pandas.DataFrame): Analysis results
        output_file (str): Output CSV file path
        config (dict): Dictionary containing downlink and uplink IP addresses of the network
    """

    downlink_df = data[data['source_ip'] == config['network']['downlink']]
    uplink_df = data[data['destination_ip'] == config['network']['downlink']]

    downlink_df.transpose().to_json(f'{output_file}_DOWNLINK.json', index=False)
    uplink_df.transpose().to_json(f'{output_file}_UPLINK.json', index=False)
    data.transpose().to_json(f'{output_file}.json', index=False)
    print("Results exported to JSON")


def main(pcap_file, config):
    # File paths
    output_file = "http2_data_analysis"
    config = _load_config(config)
    network = config['network']

    uplink_filter = ' || '.join([f'ip.addr == {ip}' for ip in network['uplink']])
    downlink_filter = f'ip.addr == {network["downlink"]}'
    display_filter = f'http2.type == 0 && ({uplink_filter} || {downlink_filter})'

    raw_data = _load_json(_transform_pcap('output.pcapng', display_filter))

    processed_data = analyze_data_streams(raw_data, network)
    save_results_to_json(processed_data, output_file, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pcap_file", help="Path to the pcap file")
    parser.add_argument("-c", "--config", help="Path to the config file")

    args = parser.parse_args()

    main(args.pcap_file, args.config)
