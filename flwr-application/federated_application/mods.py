import os
import json
from logging import INFO
from datetime import datetime
from typing import Dict, Any

from flwr.common.logger import log
from flwr.client.typing import ClientAppCallable
from flwr.common.context import Context
from flwr.common.message import Message


class MessageSizeLogger:
    def __init__(self, log_dir: str = '~/flwr_logs'):
        """
        Initialize the MessageSizeLogger with configurable log directory.

        :param log_dir: Directory to store log files, defaults to ~/flwr_logs
        """
        self.log_dir = os.path.expanduser(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

    def _calculate_message_sizes(self, msg: Message) -> Dict[str, int]:
        """
        Calculate sizes for different message components.

        :param msg: Flower Message object
        :return: Dictionary with size calculations
        """
        try:
            parameters_size = sum(p_record.count_bytes() for p_record in msg.content.parameters_records.values())
            configs_size = sum(c_record.count_bytes() for c_record in msg.content.configs_records.values())
            metrics_size = sum(m_record.count_bytes() for m_record in msg.content.metrics_records.values())
            total_size = parameters_size + configs_size + metrics_size

            return {
                "parameters": parameters_size,
                "configs": configs_size,
                "metrics": metrics_size,
                "total": total_size
            }
        except Exception as e:
            log(INFO, f"Error calculating message sizes: {e}")
            return {"parameters": 0, "configs": 0, "metrics": 0, "total": 0}

    def save_log(self, log_data: Dict[str, Any], cid: str):
        """
        Save log data to a JSON file with better file naming and error handling.

        :param log_data: Log data to save
        :param cid: Client ID
        """
        try:
            # Create a more structured filename
            filename = f"client_{cid}_message_sizes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.log_dir, filename)

            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)

            log(INFO, f"Log saved to {filepath}")
        except Exception as e:
            log(INFO, f"Failed to save log: {e}")


def message_size_mod(msg: Message, ctxt: Context, call_next: ClientAppCallable) -> Message:
    """
    Middleware function to log message sizes in Flower FL framework.

    :param msg: Incoming message
    :param ctxt: Flower context
    :param call_next: Next callable in the middleware chain
    :return: Processed message
    """
    # Create logger instance
    logger = MessageSizeLogger()

    # Extract context information
    server_round = int(msg.metadata.group_id)
    num_rounds = int(ctxt.run_config.get('rounds', 0))
    cid = str(ctxt.node_config.get('cid', 'unknown'))

    # Create log entry
    message_size_log = {
        "timestamp": datetime.now().isoformat(),
        "cid": cid,
        "message_type": msg.metadata.message_type,
        "num_rounds": num_rounds,
        "message_sizes": logger._calculate_message_sizes(msg)
    }

    # Prepare log result
    result = {server_round: message_size_log}

    # Save log
    logger.save_log(result, cid)

    return call_next(msg, ctxt)