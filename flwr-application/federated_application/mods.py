from logging import INFO
import os

from flwr.common.logger import log
from flwr.client.typing import ClientAppCallable, Mod
from flwr.common.context import Context
from flwr.common.message import Message
from datetime import datetime
import json

global_metrics = {'message_sizes': []}


def save_json_log(data: dict):
    log(INFO, "Saving log to JSON file")
    cid = next(iter(data.values()))['cid']
    path = os.path.expanduser(f'~/flwr_logs_client{cid}.json')
    try:
        with open(path, 'a') as f:
            json.dump(data, f)
        log(INFO, f"Log saved to {path}")
    except Exception as e:
        log(INFO, f"Failed to save log: {e}")


def message_size_mod(msg: Message, ctxt: Context, call_next: ClientAppCallable) -> Message:
    server_round = int(msg.metadata.group_id)
    num_rounds = int(ctxt.run_config['rounds'])
    message_size_log = {
        server_round: {
            "timestamp": datetime.now().isoformat(),
            'cid': ctxt.node_config['cid'],
            "message_type": msg.metadata.message_type,
            'num_rounds': num_rounds,
            "message_sizes": {
                "parameters": 0,
                "configs": 0,
                "metrics": 0,
                "total": 0
            }
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

    log(INFO, "Message size: %i bytes", message_size_log["message_sizes"]["total"])

    global_metrics['message_sizes'].append(message_size_log)

    if server_round == int(ctxt.run_config['rounds']):
        save_json_log(global_metrics['message_sizes'])

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