import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner, ShardPartitioner
from flwr_datasets.visualization import plot_label_distributions

DATASET_DIRECTORY = "datasets"

# Literal types for partition strategies and dataset options
PartitionStrategy = Literal[
    "iid",
    "dirichlet",
    "shard"
]

Dataset = Literal[
    'mnist',
    'cifar10',
    'cifar100'
]

# Pull choices fromm literal types
VALID_PARTITION_STRATEGIES: tuple[str, ...] = get_args(PartitionStrategy)
VALID_DATASETS: tuple[str, ...] = get_args(Dataset)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class PartitionConfiguration:
    """Dataclass to store all partition configuration parameters"""
    num_clients: int
    alpha: float
    seed: int
    min_partition_size: int
    test_split: float

    partition_strategy: PartitionStrategy
    dataset: Dataset
    output_dir: Path = Path(os.path.expanduser(f'~/{DATASET_DIRECTORY}'))
    save_plots: bool = True

    def __post_init__(self):
        if self.num_clients < 1:
            raise ValueError("Number of clients must be greater than or equal to 1")

    def __validate_dataset(self):
        if self.dataset not in VALID_DATASETS:
            raise ValueError(f"Invalid dataset: {self.dataset}. Valid datasets are: {VALID_DATASETS}")

    def __validate_partition_strategy(self):
        if self.partition_strategy not in VALID_PARTITION_STRATEGIES:
            raise ValueError(
                f"Invalid partition strategy: {self.partition_strategy}. Valid strategies are: {VALID_PARTITION_STRATEGIES}")


class DatasetPartitioner:
    """Class to partition a dataset for federated learning"""

    def __init__(self, config: PartitionConfiguration):
        self.config = config
        self._set_output_directory()

    def _set_output_directory(self) -> None:
        """Create output directory if it does not exist"""
        self.config.output_dir = self.config.output_dir / self.config.partition_strategy
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.config.output_dir}")

    def _get_partition_configuration(self) -> dict:
        """Return partition configuration based on the partition strategy"""
        match self.config.partition_strategy:
            case "iid":
                return {"train": IidPartitioner(num_partitions=self.config.num_clients)}
            case "dirichlet":
                return {"train": DirichletPartitioner(
                    num_partitions=self.config.num_clients,
                    partition_by="label",
                    alpha=1.0,
                    min_partition_size=0,
                )}
            case 'shard':
                return {'train': ShardPartitioner(
                    num_partitions=self.config.num_clients,
                    partition_by='label',
                    shard_size=1_000
                )}
            case _:
                raise ValueError(f"Invalid partition strategy: {self.config.partition_strategy}")

    def partition_dataset(self) -> None:
        """Partition the dataset, save the partitions to disk and save a plot"""
        logger.info(f"Partitioning dataset: {self.config.dataset} with {self.config.num_clients} clients")
        partitioners = self._get_partition_configuration()
        fds = FederatedDataset(
            dataset=self.config.dataset,
            partitioners=partitioners
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
            title=f"{self.config.partition_strategy} {self.config.dataset} Per Partition Labels Distribution",
        )
        fig.show()
        fig.savefig(f'{self.config.dataset}_{self.config.num_clients}_partitions_{self.config.partition_strategy}',
                    bbox_inches='tight')

        for partition_id in range(self.config.num_clients):
            partition = fds.load_partition(partition_id)
            partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
            file_path = f"{self.config.output_dir}/{self.config.dataset}_{self.config.partition_strategy}_part_{partition_id + 1}"
            partition_train_test.save_to_disk(file_path)
            print(f"Written: {file_path}")


def main():
    """Main entry point for the dataset partitioner."""
    parser = argparse.ArgumentParser(description="Enhanced Federated Learning Dataset Partitioner")

    parser.add_argument(
        '-n',
        "--num-clients",
        type=int,
        default=2,
        help="Number of client devices for federated learning"
    )
    parser.add_argument(
        "-p",
        "--partition",
        type=str,
        choices=VALID_PARTITION_STRATEGIES,  # Using values from Literal type
        default="iid",
        help="Partition strategy to use"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=VALID_DATASETS,  # Using values from Literal type
        required=True,
        help="Dataset to partition"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Concentration parameter for Dirichlet distribution"
    )
    parser.add_argument(
        "--min-partition-size",
        type=int,
        default=0,
        help="Minimum size for each partition"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="save_plots",
        help="Disable saving of distribution plots"
    )

    args = parser.parse_args()

    config = PartitionConfiguration(
        num_clients=args.num_clients,
        partition_strategy=args.partition,
        dataset=args.dataset,
        alpha=args.alpha,
        min_partition_size=args.min_partition_size,
        test_split=args.test_split,
        seed=args.seed,
        # output_dir=args.output_dir,
        save_plots=args.save_plots
    )

    partitioner = DatasetPartitioner(config)
    partitioner.partition_dataset()


if __name__ == '__main__':
    main()
