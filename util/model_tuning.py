import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
from tqdm import tqdm
import logging
from datetime import datetime
import os
import wandb

# Load WandB API key from environment variables
WANDB_PROJECT = "squeezenet_hyperparam_tuning_server"
WANDB_API_KEY = ''

sweep_configuration = {
    'method': 'bayes',  # Bayesian optimization
    'name': 'federated_sweep',
    'metric': {
        'name': 'global_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'fed_rounds': {
            'values': [10, 20]
        },
        'local_epochs': {
            'values': [2, 5]
        },
        'learning_rate': {
            'distribution': 'uniform',
            'min': 1e-4,
            'max': 1e-1
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.85,
            'max': 0.99
        },
        'weight_decay': {
            'values': [1e-4, 5e-4, 1e-3]
        },
        'batch_size': {
            'values': [64, 128, 256]
        }
    }
}

class FederatedTrainer:
    def __init__(self, config):
        self.num_nodes = 3  # Fixed number of nodes
        self.fed_rounds = config['fed_rounds']
        self.local_epochs = config['local_epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.momentum = config['momentum']
        self.weight_decay = config['weight_decay']

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Using device:', self.device)

        # Setup logging
        self.setup_logging()
        self.logger.info(f"Using device: {self.device}")

        # Initialize dataset and model
        self.setup_data()
        self.setup_model()

    def setup_logging(self):
        if not os.path.exists('logs'):
            os.makedirs('logs')

        self.logger = logging.getLogger(f'Federated_Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_data(self):
        self.logger.info("Setting up datasets...")

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        indices = np.arange(len(trainset))
        np.random.shuffle(indices)

        self.node_data = []
        samples_per_node = len(trainset) // self.num_nodes

        for i in range(self.num_nodes):
            start_idx = i * samples_per_node
            end_idx = start_idx + samples_per_node if i < self.num_nodes - 1 else len(trainset)
            node_indices = indices[start_idx:end_idx]
            node_dataset = Subset(trainset, node_indices)
            self.node_data.append(DataLoader(node_dataset, batch_size=self.batch_size,
                                             shuffle=True, num_workers=2))

        self.test_loader = DataLoader(self.testset, batch_size=100,
                                      shuffle=False, num_workers=2)

    def setup_model(self):
        model = squeezenet1_1()
        model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1))  # Adjust output to 10 classes
        model.num_classes = 10
        self.global_model = model.to(self.device)
        self.node_models = [copy.deepcopy(self.global_model) for _ in range(self.num_nodes)]

    def train_node(self, node_id, round_num):
        model = self.node_models[node_id]
        model.train()

        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate,
                              momentum=self.momentum, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in tqdm(range(self.local_epochs), desc=f"Node {node_id} Local Epochs"):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in self.node_data[node_id]:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / len(self.node_data[node_id])
            epoch_acc = 100. * correct / total

            # Log metrics to wandb
            wandb.log({
                f'node_{node_id}_loss': epoch_loss,
                f'node_{node_id}_accuracy': epoch_acc,
                'round': round_num,
            })

    def aggregate_models(self):
        global_dict = self.global_model.state_dict()

        for k in global_dict.keys():
            if isinstance(global_dict[k], torch.Tensor):
                global_dict[k] = torch.stack([self.node_models[i].state_dict()[k]
                                              for i in range(self.num_nodes)], dim=0).mean(dim=0)

        self.global_model.load_state_dict(global_dict)

        for node_model in self.node_models:
            node_model.load_state_dict(global_dict)

    def evaluate(self):
        self.global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.global_model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total

    def train(self):
        best_acc = 0

        for fed_round in range(self.fed_rounds):
            for node_id in range(self.num_nodes):
                self.train_node(node_id, fed_round)

            self.aggregate_models()
            test_acc = self.evaluate()

            wandb.log({'global_accuracy': test_acc, 'round': fed_round, 'best_accuracy': max(best_acc, test_acc)})
            best_acc = max(best_acc, test_acc)

        return best_acc

def train_sweep():
    with wandb.init() as run:
        trainer = FederatedTrainer(wandb.config)
        final_acc = trainer.train()
        wandb.log({'final_accuracy': final_acc})

def main():
    wandb.login(key=WANDB_API_KEY)
    sweep_id = wandb.sweep(sweep_configuration, project=WANDB_PROJECT)
    wandb.agent(sweep_id, function=train_sweep, count=50)

if __name__ == "__main__":
    main()
