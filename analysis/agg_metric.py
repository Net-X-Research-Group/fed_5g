import csv
import pandas as pd
import re
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def parse_experiment_name(name):
    params = {
        'bandwidth': None,
        'tdd': None,
        'nodes': None,
        'rank': '1x1',
    }

    parts = name.split('_')

    for part in parts:
        if re.search(r"\d+MHz", part, re.IGNORECASE):
            match = re.match(r"(\d+)(MHz)", part, re.IGNORECASE)
            if match:
                params['bandwidth'] = match.group(1) + 'MHz'
        elif re.match(r"\d+N$", part):
            params['nodes'] = part
        elif re.match(r"\d+-\d+$", part):
            params['tdd'] = part
        elif re.search(r"MIMO", part, re.IGNORECASE):
            params['rank'] = '2x2'

    return params

def analyze_model(model_path):
    model = torch.load(model_path)

def load(experiment_path):
    """
    Load a single experiment return metrics
    Args:
        experiment_path:

    Returns:

    """

    metrics = {}


    # Get latencies from a single experiment
    latencies =  list(experiment_path['path'].glob('latency_*_CID*.csv'))
    if latencies:
        latency_dfs = []
        for latency in latencies:
            cid_match = re.search(r"CID(\d+)", latency.name)
            if cid_match:
                cid = int(cid_match.group(1))
                df = pd.read_csv(latency)
                df['cid'] = cid
                df['server_round'] = range(1, len(df) + 1)
                latency_dfs.append(df)
        if latency_dfs:
            metrics['latency'] = pd.concat(latency_dfs, ignore_index=True)


    # Get aggregated metrics from csv
    agg_file = experiment_path['path'] / 'train_agg_metrics.csv'
    if agg_file.exists():
        metrics['server_agg_metric'] = pd.read_csv(agg_file)

    # Load the .json individual metrics
    individual_file = experiment_path['path'] / 'individual_metrics.json'
    if individual_file.exists():
        with open(individual_file, 'r') as f:
            metrics['individual_metrics'] = json.load(f)

    # Load the exec time
    exec_time_file = experiment_path['path'] / 'execution_time.txt'
    if exec_time_file.exists():
        with open(exec_time_file, 'r') as f:
            time_str = f.read().strip().rstrip('s')
            metrics['execution_time'] = float(time_str)

    # Load start time
    start_time_file = experiment_path['path'] / 'start_time.txt'
    if start_time_file.exists():
        with open(start_time_file, 'r') as f:
            time_str = f.read().strip().rstrip('s')
            metrics['start_time'] = float(time_str)


    """# Load the model
    model_file = experiment_path['path'] / 'final_model.pt'
    if model_file.exists():
        metrics['model_metric'] = analyze_model(model_file)"""

    return metrics





if __name__ == '__main__':
    directory = Path("/Users/roberthayek/hayekr@ieee.org - Google Drive/My Drive/5G Experiment Data/IMC_Analysis")
    all_experiments = []
    for exp in directory.iterdir():
        if not exp.is_dir():
            continue
        params = parse_experiment_name(exp.name)
        all_experiments.append({'path': exp, **params})

   """ # 6N, 40Mhz, MIMO, sweep TDD
    target_experiments = [
        exp for exp in all_experiments
        if exp['rank'] == '2x2' and exp['bandwidth'] == '100MHz' and exp['nodes'] == '6N'
    ]

    compare(all_experiments, filters={'bandwidth': '100MHz', 'nodes': '6N', 'rank': '2x2'}, sweep_param='tdd')"""




