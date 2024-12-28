import pyshark
import pandas as pd
import numpy as np
from tqdm import tqdm

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
        keep_packets=True,
    )

    # Lists to store packet data
    data = []
    # Process each packet
    for packet in tqdm(capture, desc="Processing packets", unit="packets"):
        try:
            # Extract HTTP2 and TCP information
            timestamp = float(packet.frame_info.time_epoch)
            stream_id = packet.http2.streamid # Corresponds to the HTTP2 stream ID which is unique to each Tx group
            packet_number = int(packet.frame_info.number) # Int
            http2_length = int(packet.http2.length) # Bytes
            source = packet.get_multiple_layers('ip')[-1].src
            destination = packet.get_multiple_layers('ip')[-1].dst
            route = (source, destination)
            # Filter packets based on size and direction
            if any(ip in route for ip in ip_addresses['uplink']) and ip_addresses['downlink'] in route:
                if http2_length >= 100:
                    direction = 'uplink' if source in ip_addresses['uplink'] else 'downlink'
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

    # Create DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("No HTTP2 DATA frames found in capture")

    consolidated_df = df.groupby(['stream_id', 'source', 'destination']).agg(
        start_time=('timestamp', 'first'),
        end_time=('timestamp', 'last'),
        total_bytes=('tcp_bytes', 'sum'),
        total_packets=('packet_number', 'count')
    ).reset_index()
    consolidated_df['total_bytes'] = consolidated_df['total_bytes'] / 1e6  # Convert to KB
    consolidated_df['duration'] = consolidated_df['end_time'] - consolidated_df['start_time']
    consolidated_df['throughput'] = (consolidated_df['total_bytes'] / consolidated_df['duration'])  # Mbps
    consolidated_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    consolidated_df.dropna(inplace=True)
    return consolidated_df

def save_results_to_csv(data, output_csv):
    """
    Save analysis results to a CSV file

    Args:
        data (pandas.DataFrame): Analysis results
        output_csv (str): Output CSV file path
    """
    data.to_csv(output_csv, index=False)
    print("Results saved to CSV")

def main():
    # File paths
    pcap_file = "ethernet_1c_2e_cifar10_cnn3_3R_nowandb_5G.pcapng"  # Replace with your pcap file path
    output_csv = "http2_data_analysis.csv"

    downlink_address = '192.168.70.129'
    uplink_addresses = ['10.0.0.2']

    network = {'downlink': downlink_address, 'uplink': uplink_addresses}


    print("Analyzing HTTP2 DATA streams...")
    data = analyze_http2_data_streams(pcap_file, network)

    # Save results to CSV
    print(f"Saving results to {output_csv}")
    save_results_to_csv(data, output_csv)

if __name__ == "__main__":
    main()