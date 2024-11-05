# 5GFed

A federated learning system using Flower. To be deployed with the SDR controller and Raspberry Pi network over 5G.

## Instructions to Run

1. Launch the superlink:
   ```sh
   flower-superlink --insecure --port=9092
   ```

2. Launch the supernode:
   ```sh
   flower-supernode --insecure --superlink="SUPERLINK_IP:9092" \
                    --isolation="subprocess" \
                    --node-config="dataset-path='path/to/cifar10_part_1'"
   ```

3. Launch the Flower apps:
   ```sh
   flwr run . federated_application
   ```

## Project Structure

- `federated_application/`: Contains the main application code.
  - `client.py`: Client-side implementation.
  - `server.py`: Server-side implementation.
  - `task.py`: Task-specific functions and model definitions.
- `dataset_partitioner.py`: Script for partitioning datasets for federated learning.
- `pyproject.toml`: Project configuration file.

## Dependencies

- flwr
- flwr-datasets
- torch
- torchvision

## Configuration

The configuration for the federated learning system is specified in the `pyproject.toml` file under `[tool.flwr.app.config]`:

- `num-server-rounds`: Number of federated learning rounds.
- `fraction-evaluate`: Fraction of clients to evaluate.
- `local-epochs`: Number of local training epochs.
- `learning-rate`: Learning rate for local training.
- `batch-size`: Batch size for local training.

## License

This project is licensed under the Apache-2.0 License.