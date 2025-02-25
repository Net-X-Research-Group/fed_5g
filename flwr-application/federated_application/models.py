import torch
import torch.nn as nn
import torch.nn.functional as f

from torchvision.models import squeezenet1_1, mobilenet_v3_small, efficientnet_b0, shufflenet_v2_x1_0

from torchvision.models.quantization import mobilenet_v3_large as mobilenet_v3_large_quant


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
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
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
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(f.relu((self.conv1(x))))  # 32x32x3 -> 28x28x32 -> 14x14x32
        x = self.pool(f.relu(self.conv2(x)))  # 14x14x32 -> 10x10x64 -> 5x5x64
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch (5x5x64 -> 1600)
        x = f.relu(self.fc1(x))  # 1600 -> 512
        x = self.fc2(x)  # 512 -> 10
        return x


class CNN3v2(nn.Module):
    """
    Link: https://nvsyashwanth.github.io/machinelearningmaster/cifar-10/
    """

    def __init__(self):
        super(CNN3v2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = self.pool(f.relu(self.conv3(x)))
        x = self.pool(f.relu(self.conv4(x)))
        x = x.view(-1, 128 * 2 * 2)
        x = self.dropout(x)
        x = self.dropout(f.relu(self.fc1(x)))
        x = self.dropout(f.relu(self.fc2(x)))
        x = self.out(x)
        return x


class ModelWrapper:
    @staticmethod
    def create_model(model: str, num_classes: int = 10) -> nn.Module:
        match model:
            case 'squeezenet1_1':
                return squeezenet1_1(num_classes=num_classes)
            case 'basic_cnn':
                return BasicCNN()
            case 'cnn3v2':
                return CNN3v2()
            case 'cnn3':
                return CNN3()
            case 'mobilenet_v3_small':
                return mobilenet_v3_small(num_classes=num_classes)
            case 'mobilenet_v3_large_quantized':
                return mobilenet_v3_large_quant(quantize=True, num_classes=num_classes)
            case 'efficientnet_b0':
                return efficientnet_b0(num_classes=num_classes)
            case 'shufflenet_v2_x1_0':
                return shufflenet_v2_x1_0(num_classes=num_classes)
            case _:
                raise ValueError(f'Model {model} is not supported... Exiting :(')
