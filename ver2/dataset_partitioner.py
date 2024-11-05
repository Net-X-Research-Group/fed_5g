import argparse

from flwr_datasets import FederatedDataset
from flwr_datasets.visualization import plot_label_distributions

def main(arguments):
    print(arguments)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Configuration")

    # Add arguments
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Number of client devices to participate in federated learning.")
    partition = parser.add_mutually_exclusive_group(required=True)
    partition.add_argument('--iid', action='store_true', help="Partition as IID dataset.")
    partition.add_argument('--niid', action='store_true', help="Partition as Non-IID dataset.")
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument("--mnist", action="store_true", help="Use MNIST dataset for training.")
    dataset_group.add_argument("--cifar10", action="store_true", help="Use CIFAR-10 dataset for training.")
    dataset_group.add_argument("--cifar100", action="store_true", help="Use CIFAR-100 dataset for training.")

    arguments = parser.parse_args()

    main(arguments)