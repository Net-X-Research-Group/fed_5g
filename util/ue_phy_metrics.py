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

def send_command(ser: serial.Serial, command: str) -> str:
    response = ""
    ser.write(f'{command}\r'.encode())

    timeout = 0.1
    quantity = ser.in_waiting

    while True:
        if quantity > 0:
            response += ser.read(quantity)
        else:
            time.sleep(timeout)
        quantity = ser.in_waiting
        if quantity == 0:
            break
    return response



def main():
    ser_device = setup_serial()

    rsp = send_command(ser_device, 'at+cimi')

    print(rsp)


if __name__ == '__main__':
    main()