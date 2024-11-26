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
    def __init__(self, log_dir: str = '~/flwr_logs', cid: str = 'unknown'):
        """
        Initialize the MessageSizeLogger with a consistent log file.

        :param log_dir: Directory to store log files
        :param cid: Client ID for filename
        """
        self.log_dir = os.path.expanduser(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Create a consistent filename based on client ID and start time
        start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_filename = f"client_{cid}_message_sizes.json"
        self.log_filepath = os.path.join(self.log_dir, self.log_filename)

        # Initialize the log file with an empty structure
        self._initialize_log_file()

    def _initialize_log_file(self):
        """
        Create an initial JSON structure for the log file.
        """
        try:
            with open(self.log_filepath, 'w') as f:
                json.dump({}, f, indent=2)
        except Exception as e:
            log(INFO, f"Failed to initialize log file: {e}")

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

    def save_log(self, log_data: Dict[str, Any]):
        """
        Update the existing log file with new log data.

        :param log_data: Log data to save
        """
        try:
            # Read existing log data
            with open(self.log_filepath, 'r') as f:
                existing_data = json.load(f)

            # Update existing data with new log data
            existing_data.update(log_data)

            # Write back the updated data
            with open(self.log_filepath, 'w') as f:
                json.dump(existing_data, f, indent=2)

            log(INFO, f"Log updated in {self.log_filepath}")
        except Exception as e:
            log(INFO, f"Failed to update log: {e}")


def message_size_mod(msg: Message, ctxt: Context, call_next: ClientAppCallable) -> Message:
    """
    Middleware function to log message sizes in Flower FL framework.

    :param msg: Incoming message
    :param ctxt: Flower context
    :param call_next: Next callable in the middleware chain
    :return: Processed message
    """
    # Extract context information
    server_round = int(msg.metadata.group_id)
    num_rounds = int(ctxt.run_config.get('rounds', 0))
    cid = str(ctxt.node_config.get('cid', 'unknown'))

    # Create logger instance with specific client ID
    logger = MessageSizeLogger(cid=cid)

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
    logger.save_log(result)

    return call_next(msg, ctxt)