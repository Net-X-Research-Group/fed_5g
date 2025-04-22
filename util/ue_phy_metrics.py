from typing import List

import serial
import time
import csv
import re
import os

ser_device = serial.Serial

def signal_handler(sig, frame):
    global ser_device
    if ser_device is not None:
        ser_device.close()



def setup_serial(port: str = '/dev/ttyUSB2', baudrate: int = 115200, parity: str = 'N', bits: int = 8):
    """
    Sets up the serial connection to the UE PHY metrics device.
    :param port: The serial port to connect to.
    :param baudrate: The baud rate for the serial connection.
    :return: The serial connection object.
    """
    try:
        ser = serial.Serial(port=port, baudrate=baudrate)
        time.sleep(2)
        return ser
    except serial.SerialException as e:
        assert f"Error opening serial port {port}: {e}"
        return None

def send_command(ser: serial.Serial, command: str):
    response = ""
    ser.write(f'{command}\r'.encode())

    timeout = 0.1
    quantity = ser.in_waiting

    while True:
        if quantity > 0:
            response += ser.read(quantity).decode()
        else:
            time.sleep(timeout)
        quantity = ser.in_waiting
        if quantity == 0:
            break
    return response.splitlines()

def save_response(data: dict, filename: str = 'at_command.csv'):
    exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())

        if not exists:
            writer.writeheader()

        writer.writerow(data)

def parse_rfsts(response: List[str]):
    extracted_response = response[2]
    pattern = r'#RFSTS:\s*\"([^\"]+)\",(\d+),(\d+),(-?\d+),(-?\d+),(-?\d+),(\d+),(\d+),(\d+),(\d+)'

    match = re.search(pattern, extracted_response)
    if match:
        response = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'plmn': match.group(1),
            'dl_channel': match.group(2),
            'ul_channel': match.group(3),
            'rsrp': match.group(4),
            'rssi': match.group(5),
            'rsrq': match.group(6),
            'band': match.group(7),
            'dl_bandwidth': match.group(8),
            'ul_bandwidth': match.group(9),
            'tx_power': match.group(10)
        }
        save_response(response, 'rfsts.csv')

def main():
    global ser_device
    ser_device = setup_serial()
    rsp = send_command(ser_device, 'at#rfsts')
    # Command Ran, Line Br, Response, Line Br, Status
    # Response: PLMN, NR_CH, NR_ULCH, NR_RSRP, NR_RSSI, NR_RSRQ, NR_BAND, NR_BW, NR_ULBW, NR_TXPWR
    parse_rfsts(rsp)


if __name__ == '__main__':
    main()