import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas import json_normalize
import re

# If you give it a file path to an experimental dataset folder like “flwr_output_10-23-25” 
#it can put all of the aggregated data into one pandas dataframe with information about the 
#experiment parameters including frequency, number of nodes, and TDD (D/U). I didn’t implement
#a column for MIMO or SISO. It will also create another pandas dataframe for the individual metrics 
#with the same information about experiment parameters and include the CID of the UE.

# The only visualization implemented right now is the grids of graphs where the x-axis is 
#always server round. The indexing issue from last week is fixed so the graphs should have 
#accurate titles now. I was manually switching which experimental parameters were compared.

def func():
    #file setup
    folder_path = input("Enter the file path: ")
    files = [f for f in os.listdir(folder_path) if f != ".DS_Store"]

    #big agg dataframe setup
    agg = pd.DataFrame(columns=["Number of Nodes", "Bandwidth (MHz)", "TDD (D/U)", "Server Round", "timestamp", "Train Loss",
                                "Train Time", "Evaluation Loss", "Evaluation Accuracy", "Evaluation Time"])

    #big ind dataFrame setup
    ind = pd.DataFrame(columns=["Number of Nodes", "Bandwidth (MHz)", "TDD (D/U)", "Server Round", "CID", "timestamp", "Train Loss",
                                "Train Time", "Evaluation Loss", "Evaluation Accuracy", "Evaluation Time"])

    for i in files:
        #getting general experiment info
        parts = i.split("_")
        num_nodes = parts[1]
        MHz = parts[2]
        TDD = parts[3]
    
        #file import csv
        tam = pd.read_csv(folder_path + '%s/train_agg_metrics.csv' % i)

        #putting agg metrics into big agg dataframe
        new_rows_agg = pd.DataFrame(columns=agg.columns)
        new_rows_agg.iloc[:, 3:10] = tam
        new_rows_agg.iloc[:, 0] = num_nodes
        new_rows_agg.iloc[:, 1] = MHz
        new_rows_agg.iloc[:, 2] = TDD
        agg = pd.concat([agg, new_rows_agg], ignore_index=True)
        
        #file import json
        data = pd.read_json(folder_path + '%s/individual_metrics.json' % i)
        temp = pd.DataFrame()
        for j in data:
            lm = json_normalize(data[j])
            for k in lm:
                temp = pd.concat([temp, json_normalize(lm[k])], ignore_index=True)

        #putting ind metrics into big ind dataframe
        new_rows_ind = pd.DataFrame(columns=ind.columns)
        new_rows_ind.iloc[:, 3:10] = temp
        new_rows_ind.iloc[:, 0] = num_nodes
        new_rows_ind.iloc[:, 1] = MHz
        new_rows_ind.iloc[:, 2] = TDD
        ind = pd.concat([ind, new_rows_ind], ignore_index=True)


    # switch whitch stuff is fixed and unfixed variables
    metric_cols = agg.columns[agg.columns.get_loc("Server Round") + 1:]
    x_col = "Server Round"
    hue_col = "Number of Nodes"
    row_col = "Bandwidth (MHz)"
    col_col = "TDD (D/U)"
    for metric in metric_cols:
       g = sns.relplot(
           data=agg,
           x=x_col,
           y=metric,
           hue=hue_col,
           row=row_col,
           col=col_col,
           kind="line",
           facet_kws={'sharey': False, 'sharex': True},
           height=3.5,
           aspect=1.2,
       )
       g.fig.suptitle(metric, fontsize=10, y=1.02)
       g.set_axis_labels("Server Round", metric)
       g.add_legend(title=hue_col)
       plt.tight_layout()
       plt.subplots_adjust(hspace=0.4, wspace=0.4)
       g.set(ylim=(agg[metric].min(), agg[metric].max()))

    plt.show()
    return()

func()