import torch
import torch.nn as nn
import torch.optim as optim
from plato.clients import simple
from plato.trainers import basic
import torchvision.transforms as transforms
from pathlib import Path
import sys
from plato.datasources import base
from datasets import load_from_disk
import asyncio
# Import the CNN3 model
sys.path.append(str(Path(__file__).resolve().parents[1] / 'flwr-application' / 'federated_application'))
from models import CNN3

class CifarDataSource(base.DataSource):
    """Handles loading of partitioned CIFAR10 dataset."""
    
    def __init__(self, client_id):
        super().__init__()
        self.client_id = client_id
        partition_path = Path.home() / f"cifar10_{client_id}"
        self.partition = load_from_disk(str(partition_path))
        
    def get_train_set(self):
        return self.partition['train']

    def get_test_set(self):
        return self.partition['test']

class CifarTrainer(basic.Trainer):
    def __init__(self):
        super().__init__()
        self.loss_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.batch_size = 32
        self.epochs = 1

    def train_model(self, config):
        """Train the model on local data."""
        self.model.train()
        for _ in range(self.epochs):
            for batch in self.trainloader:
                inputs, labels = batch
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
        return True

    def test_model(self, config):
        """Test the model on local data."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in self.testloader:
                inputs, labels = batch
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

class CifarClient(simple.Client):
    def __init__(self, model=CNN3, trainer=CifarTrainer, client_id=0):
        self.client_id = client_id
        datasource = CifarDataSource(client_id)
        super().__init__(model=model, trainer=trainer, datasource=datasource)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, default=0)
    args = parser.parse_args()
    
    client = CifarClient(client_id=args.client_id)
    client.configure()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(client.start())
    client.start()
