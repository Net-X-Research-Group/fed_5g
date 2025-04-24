import subprocess
import re
import datetime
import time

def process_line(line: str):
    return 0

def stream_gnb_log(path: str):
    """Stream gNB logs from a redirected output file"""
    try:
        print(f'Reading gNB output from {path}...')
        with open(path, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    # IF EOF, wait
                    time.sleep(0.1)
                    continue
                process_line(line)
    except KeyboardInterrupt:
        print('Stopped reading gNB output from {path}')

if __name__ == '__main__':
    stream_gnb_log('gnb_log')