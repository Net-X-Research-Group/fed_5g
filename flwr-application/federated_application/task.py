import logging
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_weights(net) -> list:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, params) -> None:
    params_dict = zip(net.state_dict().keys(), params)
    state_dict = OrderedDict(
        {
            k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
            for k, v in params_dict
        }
    )
    net.load_state_dict(state_dict, strict=True)


def load_dataset(dataset_path: str, batch_size: int) -> tuple:
    """Load the dataset from disk"""
    dataset = load_from_disk(dataset_path)

    pytorch_transforms_cifar10 = Compose([ToTensor(), Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms_cifar10(img) for img in batch["img"]]
        return batch

    partition_train_test = dataset.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, valloader


def train(net, trainloader, valloader, epochs, learning_rate, device) -> dict:
    """Train the model on the training dataset"""
    net.to(device)
    criterion = nn.CrossEntropyLoss()  # Use classification cross-entropy loss
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)  # SGD with momentum
    net.train()  # Inform PyTorch that we are training the model

    logger.info(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    tr_start = time.time()
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        for batch in trainloader:
            images, labels = batch['img'].to(device), batch['label'].to(device)
            optimizer.zero_grad()  # Zero the parameter gradients
            loss = criterion(net(images), labels)
            loss.backward()  # Forward, backward, and optimize
            optimizer.step()
    tr_end = time.time()
    train_loss, train_acc, train_test_time = test(net, trainloader, device)
    val_loss, val_acc, val_test_time = test(net, valloader, device)
    logger.info(
        f"Finished training. Training loss: {train_loss}, Validation loss: {val_loss}, Validation accuracy: {val_acc}, Training accuracy: {train_acc}")
    results = {
        'train_test_time': train_test_time,
        'val_test_time': val_test_time,
        'training_time': tr_end - tr_start,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    return results


def test(net, testloader, device) -> tuple[float, float, float]:
    """Test the model on the test dataset"""
    criterion = nn.CrossEntropyLoss()  # Use classification cross-entropy loss
    correct, loss = 0, 0.0
    net.eval()
    start = time.time()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch['img'].to(device), batch['label'].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    end = time.time()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy, end - start
