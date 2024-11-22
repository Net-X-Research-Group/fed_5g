import torch
import torch.nn as nn
import torch.nn.functional as f

class BasicCNN(nn.Module):
    def __init__(self) -> None:
        super(BasicCNN, self).__init__()
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


class CNN3(nn.Module):
    def __init__(self) -> None:
        super(CNN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5, 512)
        self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        x = self.pool(f.relu((self.conv1(x)))) # 32x32x3 -> 28x28x32 -> 14x14x32
        x = self.pool(f.relu(self.conv2(x))) # 14x14x32 -> 10x10x64 -> 5x5x64
        x = torch.flatten(x, 1) # Flatten all dimensions except batch (5x5x64 -> 1600)
        x = f.relu(self.fc1(x)) # 1600 -> 512
        x = self.fc2(x) # 512 -> 10
        return x