import json
import re

import pandas as pd


def parse_experiment_name(name):
    params = {
        'bandwidth': None,
        'tdd': None,
        'nodes': None,
        'rank': '1x1',
        'distribution': 'dirichlet',
        'congestion': False,
        'network': 'wwan'
    }

    parts = name.split('_')

    for part in parts:
        if re.search(r"\d+MHz", part, re.IGNORECASE):
            match = re.match(r"(\d+)(MHz)", part, re.IGNORECASE)
            if match:
                params['bandwidth'] = match.group(1) + ' MHz'
        elif re.match(r"\d+N$", part):
            params['nodes'] = part
        elif re.match(r"\d+-\d+$", part):
            params['tdd'] = part
        elif re.search(r"MIMO", part, re.IGNORECASE):
            params['rank'] = '2x2'
        elif re.search(r"SISO", part, re.IGNORECASE):
            params['rank'] = '1x1'
        elif re.search(r"Dirichlet", part, re.IGNORECASE):
            params['distribution'] = 'dirichlet'
        elif re.search(r"IID", part, re.IGNORECASE):
            params['distribution'] = 'iid'
        elif re.search('Congestion', part, re.IGNORECASE):
            params['congestion'] = True
        elif re.search(r"wlan", part, re.IGNORECASE):
            params['network'] = 'wlan'
        elif re.search(r"wwan", part, re.IGNORECASE):
            params['network'] = 'wwan'
        elif re.search(r"lan", part, re.IGNORECASE):
            params['network'] = 'lan'

    return params


def load(experiment_path: dict) -> dict:
    """Load a single experiment return metrics"""
    metrics = {}

    # Get latencies from a single experiment
    latencies = list(experiment_path['path'].glob('latency_*_CID*.csv'))
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
            individual_metrics = json.load(f)
            individual_metrics = {k: v['train'] for k, v in individual_metrics.items()}
            metrics['individual_metrics'] = individual_metrics

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

    return metrics




def prepare_individual_metrics(metrics, sweep_param):
    """Flatten individual metrics and add sweep parameter"""
    processed = []
    for exp in metrics:
        rows = []
        for server_round, clients in exp['metrics'].items():
            for client in clients:
                row = {'server_round': int(server_round), **client}
                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(['cid', 'server_round'])
        df['round_duration'] = df.groupby('cid')['timestamp'].diff()
        median_duration = df['round_duration'].median()
        df['round_duration'] = df['round_duration'].apply(
            lambda x: median_duration if (pd.notna(x) and x > 200) else x
        )

        df[sweep_param] = exp[sweep_param]
        processed.append(df)

    return pd.concat(processed, ignore_index=True)