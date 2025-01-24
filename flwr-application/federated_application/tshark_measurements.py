import os
import subprocess
import logging
from datetime import datetime

CELLULAR = False

def start_tshark(output_directory) -> Popen[bytes]:
    if CELLULAR:
        tshark_cmd = [
            'tshark',
            '-n',
            '-i', 'oai-cn5g',  # Attach to OAI CN interface
            '-w', os.path.expanduser(f'~/{output_directory}/output.pcapng')
        ]
    else:
        tshark_cmd = [
            'tshark',
            'f', 'tcp port 9092',
            '-n',
            '-i', 'enp0s31f6',  # Attach to OAI CN interface
            '-w', os.path.expanduser(f'~/{output_directory}/output.pcapng')
        ]
    pid = subprocess.Popen(tshark_cmd)
    return pid

# Stop tshark measurement
def stop_tshark(tshark_process):
    tshark_process.kill()
    tshark_process.wait()
    logging.info("Tshark measurement stopped.")