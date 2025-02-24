import os
import subprocess
import logging
import signal

ETHERNET = False

def start_tshark(output_directory):
    output_dir = os.path.expanduser(f'~/{output_directory}')
    os.makedirs(output_dir, exist_ok=True)
    if ETHERNET:
        tshark_cmd = [
            'tshark',
            '-n',
            '-i', 'enp0s31f6',
            '-f', 'tcp port 9092 or tcp port 9091 or tcp port 9093',
            '-w', os.path.join(output_dir, 'output.pcapng')
        ]
    else: 
        tshark_cmd = [
            'tshark',
            '-n',
            '-i', 'oai-cn5g',
            '-w', os.path.join(output_dir, 'output.pcapng')
        ]
    # Capture stderr to detect startup errors
    log_file = open(os.path.join(output_dir, 'tshark.log'), 'w')
    process = subprocess.Popen(tshark_cmd, stderr=log_file)
    return process

def stop_tshark(tshark_process):
    try:
        # Send SIGINT (like Ctrl+C) for graceful termination
        tshark_process.send_signal(signal.SIGINT)
        tshark_process.wait(timeout=10)  # Wait for process to terminate
    except subprocess.TimeoutExpired:
        logging.warning("Tshark did not exit gracefully, forcing termination.")
        tshark_process.kill()
        tshark_process.wait()
    except ProcessLookupError:
        logging.info("Tshark process already terminated.")
    logging.info("Tshark measurement stopped.")
