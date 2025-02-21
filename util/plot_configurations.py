from os import listdir
from os.path import join, isdir

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def main(input_path: str) -> None:
    trial_dirs = [d for d in listdir(input_path) if isdir(join(input_path, d))]


if __name__ == '__main__':
    input_dir = input("Enter the path to the directory containing the trials: ")
    # ../Library/CloudStorage/GoogleDrive-kaylacomer2029@u.northwestern.edu/My Drive/FL_5G_Experiments/
    main(input_dir)