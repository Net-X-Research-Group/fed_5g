import argparse
import warnings
from collections import OrderedDict
import logging
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import time
parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help=f"gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--mnist",
    action="store_true",
    help="If you use Raspberry Pi Zero clients (which just have 512MB or RAM) use "
    "MNIST",
)

warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()
        logger.info(f'Finished epoch')


def test(net, testloader, device):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in tqdm(testloader):
            batch = list(batch.values())
            images, labels = batch[0], batch[1]
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    logger.info(f"Evaluation complete. Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    return loss, accuracy


def prepare_dataset(use_mnist: bool):
    """Get MNIST/CIFAR-10 and return client partitions and global testset."""
    if use_mnist:
        fds = FederatedDataset(dataset="mnist", partitioners={"train": NUM_CLIENTS})
        img_key = "image"
        norm = Normalize((0.1307,), (0.3081,))
    else:
        fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
        img_key = "img"
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pytorch_transforms = Compose([ToTensor(), norm])

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[img_key] = [pytorch_transforms(img) for img in batch[img_key]]
        return batch

    trainsets = []
    validsets = []
    for partition_id in range(NUM_CLIENTS):
        partition = fds.load_partition(partition_id, "train")
        # Divide data on each node: 90% train, 10% test
        partition = partition.train_test_split(test_size=0.1, seed=42)
        partition = partition.with_transform(apply_transforms)
        trainsets.append(partition["train"])
        validsets.append(partition["test"])
    testset = fds.load_split("test")
    testset = testset.with_transform(apply_transforms)
    return trainsets, validsets, testset


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, use_mnist):
        self.trainset = trainset
        self.valset = valset
        if use_mnist:
            self.model = Net()
        else:
            self.model = mobilenet_v3_small(num_classes=10)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Initialized FlowerClient on device: {self.device}")

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        # Measure communication time (receiving parameters)
        comm_start = time.time()
        self.set_parameters(parameters)
        comm_time_receive = time.time() - comm_start

        # Read hyperparameters from config
        batch, epochs = config["batch_size"], config["epochs"]
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

        # Measure computation time
        compute_start = time.time()

        # Modified train function to track per-epoch metrics
        train_metrics = self.train(trainloader, optimizer, epochs)

        compute_time = time.time() - compute_start

        # Measure communication time (sending parameters)
        comm_start = time.time()
        parameters = self.get_parameters({})
        comm_time_send = time.time() - comm_start

        metrics = {
            "comm_time_receive": comm_time_receive,
            "comm_time_send": comm_time_send,
            "compute_time": compute_time,
            **train_metrics
        }

        return parameters, len(trainloader.dataset), metrics

    def train(self, trainloader, optimizer, epochs):
        """Modified train function that tracks metrics per epoch."""
        criterion = torch.nn.CrossEntropyLoss()
        epoch_losses = []
        epoch_accuracies = []

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):
                batch = list(batch.values())
                images, labels = batch[0].to(self.device), batch[1].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / len(trainloader)
            epoch_accuracy = 100. * correct / total
            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_accuracy)

            logger.info(f'Epoch {epoch + 1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%')

        return {
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            "train_accuracy": sum(epoch_accuracies) / len(epoch_accuracies)
        }

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        # Measure communication time (receiving parameters)
        comm_start = time.time()
        self.set_parameters(parameters)
        comm_time_receive = time.time() - comm_start

        # Measure computation time
        compute_start = time.time()
        valloader = DataLoader(self.valset, batch_size=64)
        loss, accuracy = test(self.model, valloader, device=self.device)
        compute_time = time.time() - compute_start

        # Measure communication time (sending results)
        comm_start = time.time()
        metrics = {
            "accuracy": float(accuracy),
            "loss": float(loss),
            "comm_time_receive": comm_time_receive,
            "compute_time": compute_time,
            "comm_time_send": time.time() - comm_start
        }

        return float(loss), len(valloader.dataset), metrics


def main():
    args = parser.parse_args()
    print(args)

    assert args.cid < NUM_CLIENTS

    use_mnist = args.mnist
    # Download dataset and partition it
    trainsets, valsets, _ = prepare_dataset(use_mnist)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid], use_mnist=use_mnist
        ).to_client(),
    )


if __name__ == "__main__":
    main()
