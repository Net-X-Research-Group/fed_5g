from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })


def main(input_path: str) -> None:
    experiment_dir = [d for d in listdir(input_path) if isdir(join(input_path, d))]

    # assume only have IID for now
    for result in experiment_dir:
        result_path = join(input_path, result)

        meta = result.split('_')
        print(meta)

    # val_test_time_aggregated.csv
    # training_time_aggregated.csv
    # train_test_time_aggregated.csv
    # avg_val_acc.csv
    # avg_val_loss.csv
    # avg_train_loss.csv
    # avg_train_acc.csv



if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    # ../Library/CloudStorage/GoogleDrive-kaylacomer2029@u.northwestern.edu/My Drive/FL_5G_Experiments/
    main(input_dir)