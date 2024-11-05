import argparse

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions

DATASET_DIRECTORY = "datasets"


def main(arguments):
    num_clients = arguments.num_clients
    partition_type = arguments.partition
    dataset = arguments.dataset

    # IID Partitioning
    if partition_type == 'iid':
        fds = FederatedDataset(
            dataset=dataset,
            partitioners={
                "train": IidPartitioner(num_partitions=num_clients),
            },
        )
    #Non-IID Partition Types
    if partition_type == 'dirichlet':
        fds = FederatedDataset(
            dataset=dataset,
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=num_clients,
                    partition_by="label",
                    alpha=1.0,
                    min_partition_size=0,
                ),
            },
        )
    partitioner = fds.partitioners['train']

    fig, ax, df = plot_label_distributions(
        partitioner,
        label_name="label",
        plot_type="bar",
        size_unit="percent",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        cmap="rainbow",
        title=f"{dataset} Per Partition Labels Distribution",
    )
    fig.show()
    fig.savefig(f'{dataset}_{num_clients}_partitions_{partition_type}', bbox_inches='tight')

    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id)
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        file_path = f"./{DATASET_DIRECTORY}/{dataset}_part_{partition_id + 1}"
        partition_train_test.save_to_disk(file_path)
        print(f"Written: {file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Configuration")

    # Add arguments
    parser.add_argument("--num-clients", type=int, default=2,
                        help="Number of client devices to participate in federated learning.")
    parser.add_argument('-p', '--partition', help="Partition as Non-IID dataset. Default is IID.", default='iid')
    parser.add_argument('-d', '--dataset', required=True, choices={'mnist', 'cifar10', 'cifar100'}, help='Select the dataset.')

    arguments = parser.parse_args()

    main(arguments)