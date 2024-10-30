import time
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


# Previous imports and setup remain the same...

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