import os
import subprocess
import logging
from datetime import datetime

def _ensure_tshark():
    try:
        subprocess.run(['tshark', '-v'], check=True)
    except FileNotFoundError:
        logging.error("tshark not found. Please install Wireshark.")
        exit(1)
    except subprocess.CalledProcessError:
        logging.error("tshark not found. Please install Wireshark.")
        exit(1)

def start_tshark(output_file):
    _ensure_tshark()
    tshark_cmd = [
        'tshark',
        '-i', 'oai-cn5g',  # Attach to OAI CN interface
        '-f', '\"tcp port 9092\"',
        '-w', os.path.expanduser(f'~/{output_file}.pcapng')
    ]
    return subprocess.Popen(tshark_cmd)

# Stop tshark measurement
def stop_tshark(tshark_process):
    tshark_process.terminate()
    tshark_process.wait()
    logging.info("Tshark measurement stopped.")