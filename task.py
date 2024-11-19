from collections import OrderedDict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, 120)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(self.bn1(f.relu(self.conv1(x))))
        x = self.pool(self.bn2(f.relu(self.bn2(self.conv2(x)))))
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = self.dropout(x)
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader

def train(net, trainloader, valloader, epochs, learning_rate, device) -> dict:
    """Train the model on the training dataset"""
    net.to(device)
    criterion = nn.CrossEntropyLoss() # Use classification cross-entropy loss
    optimizer = optim.SGD(net.parameters(), lr=learning_rate) # SGD with momentum
    net.train() # Inform PyTorch that we are training the model

    logger.info(f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        for batch in trainloader:
            images, labels = batch['img'].to(device), batch['label'].to(device)
            optimizer.zero_grad() # Zero the parameter gradients
            loss = criterion(net(images), labels)
            loss.backward() # Forward, backward, and optimize
            optimizer.step()
            #assert loss.dim() == 0
            # print statistics
            running_loss += loss.item()

    avg_train_loss = running_loss / len(trainloader)
    val_loss, val_acc = test(net, valloader, device)
    logger.info(f"Finished training. Training loss: {avg_train_loss}, Validation loss: {val_loss}, Validation accuracy: {val_acc}")
    results = {
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    return results


def test(net, testloader, device) -> tuple[float, float]:
    """Test the model on the test dataset"""
    criterion = nn.CrossEntropyLoss() # Use classification cross-entropy loss
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch['img'].to(device), batch['label'].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
