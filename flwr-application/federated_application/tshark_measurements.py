import os
import subprocess
import logging
from datetime import datetime

def start_tshark(output_file):
    tshark_cmd = [
        'tshark',
        '-n',
        '-i', 'oai-cn5g',  # Attach to OAI CN interface
        '-f', 'tcp port 9092',
        '-w', os.path.expanduser(f'~/{output_file}.pcapng')
    ]
    return subprocess.Popen(tshark_cmd)

# Stop tshark measurement
def stop_tshark(tshark_process):
    tshark_process.terminate()
    tshark_process.wait()
    logging.info("Tshark measurement stopped.")