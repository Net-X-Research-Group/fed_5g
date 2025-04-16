import serial
import time
import csv
import re

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


def main():
    ser_device = setup_serial()

    ser_device.write('AT+CIMI'.encode())
    print(ser_device.readall().decode())

if __name__ == '__main__':
    main()