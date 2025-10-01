import logging
from pathlib import Path

import paramiko
import torch
from flwr.common import Context, ArrayRecord
from flwr.server import ServerApp, Grid
from scp import SCPClient

from federated_application.models import ModelWrapper
from federated_application.strategy import CellFedAvg

torch.manual_seed(42)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_ssh_client(server, port, user):
    """
    Helper Function to create an SSH client connection
    """
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, allow_agent=True, look_for_keys=True, key_filename='/app/.ssh/id_ed25519')
    return client


def transfer_latency_measurements(num_clients, save_path, run_id):
    """
    Helper Function
    -----------------
    Uses scp to transfer the latency measurements from each client to the server output directory.
    """
    DEVICE_NAME_PREFIX = "commnetpi0"
    IP_PREFIX = "129.105.6."
    IP_SUFFIXES = [17, 18, 19, 20, 21, 22]

    for cid in range(1, num_clients + 1):
        login = f"{DEVICE_NAME_PREFIX}{cid}"
        ip = f"{IP_PREFIX}{IP_SUFFIXES[cid - 1]}"
        hostname = ip
        username = login

        logger.info(f"Transferring latency measurements from {login} ({ip})...")
        try:
            ssh = create_ssh_client(hostname, 22, username)
            with SCPClient(ssh.get_transport()) as scp:
                stdin, stdout, stderr = ssh.exec_command("ls latency_*.csv")
                file_name = stdout.read().decode().strip()
                if not file_name:
                    print(f"No latency file found on {hostname}")
                    ssh.close()
                    continue
                local_path = save_path / f"{run_id}_CID{cid}.csv"
                scp.get(f'latency_{run_id}.csv', str(local_path))

        except Exception as e:
            logger.error(f"Failed to connect to {login} ({ip}): {e}")
            continue


app = ServerApp()

@app.main()
def main(grid: Grid, context: Context):
    """
    Main function to start the federated learning server using Flower's ServerApp.
    """
    # Read run config
    min_num_clients = context.run_config['min_num_clients']
    rounds = context.run_config['rounds']
    model_name = context.run_config['model']
    run_id = context.run_id

    # Load the model
    global_model = ModelWrapper.create_model(model_name, num_classes=10)
    arrays = ArrayRecord(global_model.state_dict())

    # Save path is based on the current directory
    save_path = Path.home() / 'flwr_output' / str(run_id)

    save_path.mkdir(parents=True, exist_ok=False)


    strategy = CellFedAvg(fraction_train=1,
                          fraction_evaluate=1,
                          min_train_nodes=min_num_clients,
                          min_evaluate_nodes=min_num_clients,
                          min_available_nodes=min_num_clients)

    strategy.set_save_path(save_path)


    result = strategy.start(grid=grid,
                            initial_arrays=arrays,
                            num_rounds=rounds)

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"{save_path}/final_model.pt")
    logger.info(f"Final model saved to {save_path}")

    # Grab the latency metrics from the clients (Calls helper script)
    logger.info(f'Grabbing latency metrics from clients...')
    if not context.run_config['debug']:
        transfer_latency_measurements(num_clients=min_num_clients, save_path=save_path, run_id=run_id)