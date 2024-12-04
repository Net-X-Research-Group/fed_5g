import logging
import time
from datetime import datetime

import wandb
from flwr.client.typing import ClientAppCallable
from flwr.common import ConfigsRecord
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.message import Message

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
logger = logging.getLogger(__name__)



PROJECT_NAME = "Pytorch-5G-FLWR-CIFAR10"

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


