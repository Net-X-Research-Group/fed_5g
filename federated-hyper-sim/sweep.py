import subprocess

import toml
import wandb


def update_pyproject_toml(config):
    # Load the pyproject.toml file
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)

    # Update the [tool.flwr.app.config] section with the new hyperparameters
    pyproject["tool"]["flwr"]["app"]["config"]["rounds"] = config.rounds
    pyproject["tool"]["flwr"]["app"]["config"]["local_epochs"] = config.epochs
    pyproject["tool"]["flwr"]["app"]["config"]["learning_rate"] = config.learning_rate
    pyproject["tool"]["flwr"]["app"]["config"]["batch_size"] = config.batch_size
    pyproject["tool"]["flwr"]["app"]["config"]["momentum"] = config.momentum
    pyproject["tool"]["flwr"]["app"]["config"]["weight_decay"] = config.weight_decay

    # Save the updated pyproject.toml file
    with open("pyproject.toml", "w") as f:
        toml.dump(pyproject, f)

def main():
    # Initialize WandB
    wandb.init(project="hyperparam_tuning")

    # Get the hyperparameters from WandB
    config = wandb.config

    # Update the pyproject.toml file with the new hyperparameters
    update_pyproject_toml(config)

    # Execute the flwr run . command
    subprocess.run(["flwr", "run", ".", "--stream"])
    wandb.finish()

if __name__ == "__main__":
    main()