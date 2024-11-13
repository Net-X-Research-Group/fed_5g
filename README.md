
Here is the documentation for your project in the `README.md` file:

# Federated Learning Dataset Partitioner

This project provides tools for partitioning datasets for federated learning experiments. It supports various partitioning strategies and datasets.

## Requirements

- Python
- pip

## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Partitioning a Dataset

Dataset partitions are load to each client using `scp` manually.


To partition a dataset, run the `dataset_partitioner.py` script with the desired options:

```bash
python dataset_partitioner.py -d <dataset> -n <num_clients> -p <partition_strategy>
```

#### Options

- `-d`, `--dataset`: Dataset to partition. Valid options are `mnist`, `cifar10`, `cifar100`.
- `-n`, `--num-clients`: Number of client devices for federated learning.
- `-p`, `--partition`: Partition strategy to use. Valid options are `iid`, `dirichlet`, `shard`.
- `--alpha`: Concentration parameter for Dirichlet distribution (default: 1.0).
- `--min-partition-size`: Minimum size for each partition i.e the number of clients (default: 0).
- `--test-split`: Proportion of data to use for testing (default: 0.2).
- `--seed`: Random seed for reproducibility (default: 42).
- `--no-plots`: Disable saving of distribution plots.

### Federated Learning Server
The `server.py` script is used to standup a server using the flower library. 
The server will listen for connections from clients and coordinate the federated learning process using the FedAvg strategy.

To start the server, run the `server.py` script with the desired options:

```bash
python server.py --server_address 127.0.0.1:8080 --rounds 30 --min_num_clients 3
```

### Federated Learning Client
The `client.py` script is used to connect to the server and participate in the federated learning process.

To start a client, run the `client.py` script with the desired options:

```bash
python client.py --server_address 127.0.0.1:8080 --dataset ~/dataset_name --cid 1
```

## License

This project is licensed under the MIT License.