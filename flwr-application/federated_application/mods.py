from logging import INFO
import os

from flwr.common.logger import log
from flwr.client.typing import ClientAppCallable, Mod
from flwr.common.context import Context
from flwr.common.message import Message
from datetime import datetime
import json

global_results = {}

def ensure_log_dir(log_dir: str = f"{os.path.expanduser('~/logs/')}") -> str:
    """
    Ensure log directory exists

    Args:
        log_dir (str): Directory path for logs
    """
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def save_json_log(data: dict, filename: str, log_dir: str = f"{os.path.expanduser('~/logs/')}"):
    """
    Save log data to a JSON file

    Args:
        data (dict): Log data to save
        filename (str): Name of the log file
        log_dir (str): Directory to save logs
    """
    log(INFO, "Saving log to JSON file")
    log_dir = ensure_log_dir(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(log_dir, full_filename)
    global_results[timestamp] = data
    try:
        with open(filepath, 'a') as f:
            json.dump(data, f, indent=2)
        log(INFO, f"Log saved to {filepath}")
    except Exception as e:
        log(INFO, f"Failed to save log: {e}")


def message_size_mod(msg: Message, ctxt: Context, call_next: ClientAppCallable) -> Message:
    """Message size mod.

    This mod logs the size in bytes of the message being transmited.
    """
    message_size_log = {
        "timestamp": datetime.now().isoformat(),
        "message_type": msg.metadata.message_type,
        'cid': ctxt.node_config['cid'],
        "run_id": msg.metadata.run_id,
        "node_id": msg.metadata.dst_node_id,
        "message_sizes": {
            "parameters": 0,
            "configs": 0,
            "metrics": 0,
            "total": 0
        }
    }

    # Calculate sizes for different message components
    parameters_size = sum(p_record.count_bytes() for p_record in msg.content.parameters_records.values())
    configs_size = sum(c_record.count_bytes() for c_record in msg.content.configs_records.values())
    metrics_size = sum(m_record.count_bytes() for m_record in msg.content.metrics_records.values())
    total_size = parameters_size + configs_size + metrics_size

    # Update log with size details
    message_size_log["message_sizes"] = {
        "parameters": parameters_size,
        "configs": configs_size,
        "metrics": metrics_size,
        "total": total_size
    }

    '''for p_record in msg.content.parameters_records.values():
        message_size_in_bytes += p_record.count_bytes()

    for c_record in msg.content.configs_records.values():
        message_size_in_bytes += c_record.count_bytes()

    for m_record in msg.content.metrics_records.values():
        message_size_in_bytes += m_record.count_bytes()'''

    log(INFO, "Message size: %i bytes", message_size_log["message_sizes"]["total"])

    save_json_log(message_size_log, "message_size_log.json")

    return call_next(msg, ctxt)

'''def fit_time(msg: Message, context: Context, app: ClientAppCallable) -> Message:
    """Flower Mod that logs the metrics dictionary returned by the client's fit
    function to Weights & Biases."""
    server_round = int(msg.metadata.group_id)

    if server_round == 1 and msg.metadata.message_type == MessageType.TRAIN:
        run_id = msg.metadata.run_id
        group_name = f"Run ID: {run_id}"
        node_id = str(msg.metadata.dst_node_id)
        run_name = f"Node ID: {node_id}"
    start_time = time.time()

    reply = app(msg, context)

    time_diff = time.time() - start_time

    # if the `ClientApp` just processed a "fit" message, let's log some metrics to W&B
    if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():
        metrics = reply.content.configs_records
        results_to_log = dict(metrics.get("fitres.metrics", ConfigsRecord()))
        results_to_log["fit_time"] = time_diff

    return reply'''