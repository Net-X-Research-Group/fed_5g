import os.path

import matplotlib.pyplot as plt
import pandas as pd
import json
import argparse

def plot_agg_metrics(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data).transpose()

    fig, axs = plt.subplots(2, 1, tight_layout=True, figsize=(10, 10))
    axs[0].plot(df.index, df['train_loss'], label='Train')
    axs[0].plot(df.index, df['val_loss'], label='Validation')
    axs[0].set_xlim(1, df.index[-1])
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Round')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend(['Train', 'Validation'])
    axs[0].xaxis.set_major_locator(plt.MaxNLocator(25, integer=True))

    axs[1].plot(df.index, df['train_acc'], label='Train')
    axs[1].plot(df.index, df['val_acc'], label='Validation')
    axs[1].set_xlim(1, df.index[-1])
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Round')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True)
    axs[1].legend(['Train', 'Validation'])
    axs[1].xaxis.set_major_locator(plt.MaxNLocator(25, integer=True))

    plt.savefig('server_agg_metrics.png')
    #plt.show()

def main(file: str):
    plot_agg_metrics(os.path.expanduser(file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-f", "--file", help="Path to the metrics_file")
    args = parser.parse_args()

    main(args.file)
