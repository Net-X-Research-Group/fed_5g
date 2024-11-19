import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from task import Net
import torch.nn as nn
import logging
from typing import List
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

class Centralized:
    def __init__(self, batch_size=16, learning_rate=0.001, momentum=0.9, val_split=0.2):
        self.device = 'cpu'
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.trainloader = None
        self.valloader = None
        self.val_split = val_split
        # Track losses
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
    def setup_data(self):
        """Set up data transformations and dataloaders with validation split"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        # Calculate split sizes
        val_size = int(len(full_trainset) * self.val_split)
        train_size = len(full_trainset) - val_size

        # Split into train and validation sets
        trainset, valset = torch.utils.data.random_split(
            full_trainset,
            [train_size, val_size]
        )

        # Create data loaders
        self.trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

        self.valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        self.testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )


    def setup_model(self):
        self.net = Net().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def train(self, num_epochs):
        logger.info('Training for {} epochs'.format(num_epochs))
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

        logger.info('Finished Training')

    def save_model(self, path='./cifar_net.pth'):
        torch.save(self.net.state_dict(), path)

    def test(self):
        if self.net is None:
            raise Exception('Model has not been trained yet')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct // total
        print(f'Accuracy of the network on the 10000 test images: {accuracy}%')
        return accuracy


def main():
    # Create trainer instance
    trainer = Centralized(batch_size=16, val_split=0.2)

    # Setup
    trainer.setup_data()
    trainer.setup_model()

    # Train and get history
    history = trainer.train(num_epochs=100)

    # Save the model and history
    trainer.save_model()

    # Evaluate
    trainer.evaluate()

    return history


if __name__ == "__main__":
    main()

