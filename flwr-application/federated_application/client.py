import time
from os import path

import torch
from flwr.client import ClientApp
from flwr.common import Context, Message, ArrayRecord, MetricRecord, RecordDict

from federated_application.models import ModelWrapper
from federated_application.task import load_dataset, train_fn, test_fn

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    train_timestamp = time.time()
    # Load the model and initialize it with the received weights
    model = ModelWrapper.create_model(model=context.run_config['model'], num_classes=10)
    model.load_state_dict(msg.content['arrays'].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    train_eval: bool = context.run_config['train_eval']
    cid: str = context.node_config['cid']
    batch_size: int = context.run_config['batch_size']
    dataset_path: str = path.expanduser(f"{context.node_config['dataset']}_part_{cid}")
    trainloader, valloader = load_dataset(dataset_path, batch_size)

    eval_loss = eval_acc = eval_time = train_loss = train_time = -1
    if train_eval:
        eval_loss, eval_acc, eval_time = test_fn(net=model,
                                                 testloader=valloader,
                                                 device=device)
    # Call the training function
    train_loss, train_time = train_fn(
        net=model,
        trainloader=trainloader,
        epochs=context.run_config['epochs'],
        learning_rate=context.run_config['learning_rate'],
        momentum=context.run_config['momentum'],
        weight_decay=context.run_config['weight_decay'],
        device=device
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        'cid': cid,
        'timestamp': train_timestamp,
        'train_loss': train_loss,
        'train_time': train_time,
        'eval_loss': eval_loss,
        'eval_acc': eval_acc,
        'eval_time': eval_time,
        'num-examples': len(trainloader.dataset),
        'num-eval-examples': len(valloader.dataset)
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({'arrays': model_record, 'metrics': metric_record})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Load the model and initialize it with the received weights
    model = ModelWrapper.create_model(model=context.run_config['model'], num_classes=10)
    model.load_state_dict(msg.content['arrays'].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    cid: str = context.node_config['cid']
    batch_size: int = context.run_config['batch_size']
    dataset_path: str = path.expanduser(f"{context.node_config['dataset']}_part_{cid}")
    _, valloader = load_dataset(dataset_path, batch_size)
    eval_loss, eval_acc, eval_time = test_fn(model, valloader, device)

    metrics = {
        'cid': cid,
        'eval_loss': eval_loss,
        'eval_acc': eval_acc,
        'eval_time': eval_time,
        'num-examples': len(valloader.dataset)
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({'metrics': metric_record})
    return Message(content=content, reply_to=msg)