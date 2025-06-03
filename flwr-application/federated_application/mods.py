import logging
import re
import time
from datetime import datetime

import wandb
from flwr.client.typing import ClientAppCallable
from flwr.common import ConfigsRecord
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.message import Message
import serial

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
logger = logging.getLogger(__name__)



PROJECT_NAME = "Pytorch-5G-FLWR-CIFAR10"

def _send_serial_command(ser: serial.Serial, command: str):
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

def phy_layer_measurement_mod(message: Message, context: Context, app: ClientAppCallable) -> Message:
    """
    This modification is uses qmi or at commands to measure the physical layer parameters on the UE

    Hardcoded for Telit 980m on /dev/cdc-wdm0 interface with serial interface /dev/ttyUSB2
    """
    reply = app(message, context)
    ser = None

    try:
        ser = serial.Serial(port='/dev/ttyUSB2', baudrate=115200)
        try:
            rsp = _send_serial_command(ser, 'at#rfsts')
            extracted_response = rsp[2]
            pattern = r'#RFSTS:\s*\"([^\"]+)\",(\d+),(\d+),(-?\d+),(-?\d+),(-?\d+),(\d+),(\d+),(\d+),(\d+)'

            match = re.search(pattern, extracted_response)
            if match:
                reply.content.configs_records['fitres.metrics']['rsrp'] = float(match.group(4))
                reply.content.configs_records['fitres.metrics']['rssi'] = float(match.group(5))
                reply.content.configs_records['fitres.metrics']['rsrq'] = float(match.group(6))
            else:
                logger.error("No match found in the response")
                reply.content.configs_records['fitres.metrics']['rsrp'] = 0
                reply.content.configs_records['fitres.metrics']['rssi'] = 0
                reply.content.configs_records['fitres.metrics']['rsrq'] = 0
        except Exception as e:
            logger.error(f"Error executing AT command: {e}")
            return reply
    except Exception as e:
        logger.error(f"Error opening serial port: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
    return reply

def comm_time_mod(message: Message, context: Context, app: ClientAppCallable) -> Message:
    downlink_time = time.time() - message.content.configs_records['fitins.config']['server_timestamp']
    reply = app(message, context)
    if reply.metadata.message_type == MessageType.TRAIN:
        reply.content.configs_records['fitres.metrics']['downlink_time'] = downlink_time
        reply.content.configs_records['fitres.metrics']['uplink_time'] = time.time()
    return reply


def wandb_metrics_mod(message: Message, context: Context, app: ClientAppCallable) -> Message:
    current_round = int(message.metadata.group_id)
    # Authenticate with wandb
    wandb.login(key=context.run_config['wandb_api_key'])

    # Initialize the wandb project
    run_id = message.metadata.run_id
    group_name = f'Run ID: {run_id}'
    node_id = context.node_config['cid']
    run_name = f'{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}_CID-{node_id}'
    wandb.init(
        project=PROJECT_NAME,
        group=group_name,
        name=run_name,
        id=f'{run_id}-{node_id}',
        resume='allow',
        reinit=True,
        config={'rounds': context.run_config['rounds'],
                'fraction_evaluate': context.run_config['fraction_evaluate'],
                'local_epochs': context.run_config['local_epochs'],
                'learning_rate': context.run_config['learning_rate'],
                'batch_size': context.run_config['batch_size'],
                'min_num_clients': context.run_config['min_num_clients']
                }
    )

    start = time.time()

    reply = app(message, context)

    end = time.time()

    if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():
        metrics = reply.content.configs_records
        logged_results = dict(metrics.get('fitres.metrics', ConfigsRecord()))
        logged_results['fit_time'] = end - start
        wandb.log(logged_results, step=int(current_round), commit=True)

    return reply


