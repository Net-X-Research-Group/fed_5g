import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from pathlib import Path

def _read_json(file):
    with open(file, 'r') as f:
        raw = json.load(f)
    return raw

def export_metrics_to_csv(data, output_dir="individual_metrics_parsed"):
    """
    Export each metric to a separate CSV file where columns are CIDs and rows are rounds.

    Args:
        data (dict): Nested dictionary with structure {Round#: {CID: {metrics}}}
        output_dir (str): Directory to save the CSV files
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all metrics from the first round and first CID
    first_round = list(data.keys())[0]
    first_cid = list(data[first_round].keys())[0]
    metrics = list(data[first_round][first_cid].keys())

    # Get all unique CIDs
    cids = set()
    for round_data in data.values():
        cids.update(round_data.keys())
    cids = sorted(list(cids))

    # Get all rounds
    rounds = sorted(list(data.keys()), key=int)

    # For each metric, create a DataFrame and save to CSV
    for metric in metrics:
        # Create empty DataFrame with rounds as index
        df = pd.DataFrame(index=rounds)

        # Fill in data for each CID
        for cid in cids:
            cid_values = []
            for round_num in rounds:
                if cid in data[round_num]:
                    cid_values.append(data[round_num][cid][metric])
                else:
                    cid_values.append(None)  # Handle missing data
            df[f'CID_{cid}'] = cid_values
        df.index.name = 'Round'
        # Save to CSV
        filename = f"{metric}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)
        print(f"Saved {filename}")

def plot_agg_metrics(file_name):
    data = _read_json(file_name)
    agg_df = pd.DataFrame(data).transpose()
    fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(10, 10))
    axs[0].plot(agg_df.index, agg_df['train_loss'], label='Train')
    axs[0].plot(agg_df.index, agg_df['val_loss'], label='Validation')
    axs[0].set_xlim(1, agg_df.index[-1])
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend(['Train', 'Validation'])
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(25, integer=True))

    axs[1].plot(agg_df.index, agg_df['train_acc'], label='Train')
    axs[1].plot(agg_df.index, agg_df['val_acc'], label='Validation')
    axs[1].set_xlim(1, agg_df.index[-1])
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True)
    axs[1].legend(['Train', 'Validation'])
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(25, integer=True))

    plt.savefig('metrics_plots/server_agg_metrics.png')

def plot_individual_metrics(metrics_path):
    try:
        os.mkdir('metrics_plots')
    except FileExistsError:
        pass

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(metrics_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        # Read the CSV file
        df = pd.read_csv(os.path.join(metrics_path, csv_file))

        # Drop the 'Round' column if it exists
        if 'Round' in df.columns:
            df.drop('Round', axis=1, inplace=True)

        # Determine the number of columns (CIDs)
        cols = len(df.columns)

        if '_latency' in csv_file or '_time' in csv_file:
            # Create a figure for histograms
            fig, ax = plt.subplots(cols, 1, figsize=(8, 4 * cols))
            fig.suptitle(csv_file.replace('.csv', '').replace('_', ' ').title())
            if cols == 1:
                ax = [ax]

            # Plot histograms for each column
            for i, col in enumerate(df.columns):
                ax[i].hist(df[col], label=col)
                ax[i].set_title(col)
                ax[i].set_xlabel('Value')
                ax[i].set_ylabel('Frequency')
                ax[i].legend()
        else:
            # Create a figure for line plots
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.suptitle(csv_file.replace('.csv', '').replace('_', ' ').title())

            # Plot line plot for each column
            for col in df.columns:
                ax.plot(df[col], label=col)
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.savefig(f"metrics_plots/indiv_{csv_file.replace('.csv', '')}.png")
        plt.close(fig)



def main():
    individual_metrics_path = 'individual_metrics.json'
    aggregate_metrics_path = 'agg_metrics.json'
    flwr_config_path = 'config.json'
    network_config = 'network.yml'

    # Parse and return the individual metrics.
    export_metrics_to_csv(_read_json(individual_metrics_path))
    plot_individual_metrics('individual_metrics_parsed')
    plot_agg_metrics(aggregate_metrics_path)


if __name__ == '__main__':
    main()