import time
from datetime import datetime
import wandb
from flwr.client.typing import ClientAppCallable
from flwr.common import ConfigsRecord
from flwr.common.constant import MessageType
from flwr.common.context import Context
from flwr.common.message import Message

PROJECT_NAME = "Pytorch-5G-FLWR-CIFAR10"

def wandb_metrics_mod(message: Message, context: Context, app: ClientAppCallable) -> Message:
    config = context.run_config
    current_round = int(message.metadata.group_id)
    if current_round == 1 and message.metadata.message_type == MessageType.TRAIN:
        # Authenticate with wandb
        wandb.login(key=config['wandb_api_key'])

        # Initialize the wandb project
        run_id = message.metadata.run_id
        group_name = f'Run ID: {run_id}'
        node_id = context.node_config['cid']
        run_name = f'{datetime.now().strftime("%Y-%m-%d/%H-%M-%S")}_CID-{node_id}'
        config.pop('wandb_api_key')
        wandb.init(
            project=PROJECT_NAME,
            group=group_name,
            name=run_name,
            id=f'{run_id}-{node_id}',
            resume='allow',
            reinit=True,
            config=config
        )

    start = time.time()

    reply = app(message, context)

    end = time.time()

    if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():
        metrics = reply.content.configs_records
        logged_results = dict(metrics.get('fitres.metrics', ConfigsRecord()))
        logged_results['client_fit_time'] = end - start

        wandb.log(logged_results, step=int(current_round), commit=True)

    return reply


