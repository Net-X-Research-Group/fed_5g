import json
import re
from os import listdir, makedirs
from os.path import join, isdir, exists

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.size': 18,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.format': 'svg',
        'savefig.bbox': 'tight'
    })

class Parser:
    """
    Given a configuration directory, grab and combine the trials, output to configuration directory.
    """
    def __init__(self, config_dir):
        self.config_dir = config_dir

    @staticmethod
    def read_json(file):
        with open(file, 'r') as f:
            raw = json.load(f)
        return raw

    def process(self):
        """
        Process the configuration directory, parse the data, and export to CSV.

        Args:
            data (dict): Nested dictionary with structure {Round#: {CID: {metrics}}}
            output_dir (str): Directory to save the CSV files
        """
        trial_dirs = [d for d in listdir(self.config_dir) if isdir(join(self.config_dir, d)) and d not in ['output', 'figures']]

        for trial_dir in trial_dirs:
            trial_path = join(self.config_dir, trial_dir)

            individual_raw_data = self.read_json(join(trial_path, "individual_metrics.json"))

            # Create output directory if it doesn't exist
            try:
                makedirs(join(trial_path, 'output'))
            except FileExistsError:
                pass

            # Export before doing anything
            self.export_individual(individual_raw_data, output_dir=join(trial_path, 'output'))

            self.compile_latency(trial_path, output_dir=join(trial_path, 'output'))
            self.parse_rfsts(trial_path, output_dir=join(trial_path, 'output'))

        # Compile the trials into one output dir in each configuration.
        self.aggregate_trials(trial_dirs)

    def aggregate_trials(self, trial_dirs):
        output_path = join(self.config_dir, 'output')
        makedirs(output_path, exist_ok=True)

        first_trial_output = join(self.config_dir, trial_dirs[0], 'output')
        metrics_files = [f for f in listdir(str(first_trial_output)) if f.endswith('.csv')]

        aggregator = Aggregator(self.config_dir)

        for metric_file in metrics_files:
            metrics_dict = {}
            for trial_dir in trial_dirs:
                trial_csv_path = join(self.config_dir, trial_dir, 'output', metric_file)
                if exists(trial_csv_path):
                    try:
                        metrics_dict[trial_dir] = pd.read_csv(trial_csv_path)
                    except Exception as e:
                        print(f"Error reading {trial_csv_path}: {e}")
            if metrics_dict:
                metric_name = metric_file.replace('.csv', '') # Extract name of metric
                if metric_name in ['train_loss', 'val_loss', 'train_acc', 'val_acc']:
                    aggregator.aggregate_ml_metrics(metrics_dict, output_path, metric_name)
                elif metric_name in ['rssi', 'rsrq', 'rsrp']:
                    print(f'Skipping {metric_name}, TODO IMPLEMENT PHY AGGREGATION')
                else:
                    aggregator.aggregate_time_metrics(metrics_dict, output_path, metric_name)


    @staticmethod
    def export_individual(data, output_dir):
        """
        Export each metric to a separate CSV file where columns are CIDs and rows are rounds.

        Args:
            data (dict): Nested dictionary with structure {Round#: {CID: {metrics}}}
            output_dir (str): Directory to save the CSV files
        """

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

        ''' Just repeat the code from the following for loop (for metric in metrics) to inelegantly calculate round time'''
        round_time_df = pd.DataFrame(index=rounds)
        for cid in cids:
            cid_values = []
            for round_num in rounds:
                if cid in data[round_num]:
                    value = 0
                    for metric in list(data[first_round][first_cid].keys()):
                        if 'time' in metric and 'start' not in metric and 'end' not in metric:
                            # cid_values.append(data[round_num][cid][metric])
                            value += data[round_num][cid][metric]
                    cid_values.append(value)
                else:
                    cid_values.append(None)  # Handle missing data
            round_time_df[f'CID_{cid}'] = cid_values
        round_time_df.index.name = 'Round'
        # Save to CSV
        filename = f"round_time.csv"
        filepath = join(output_dir, filename)
        round_time_df.to_csv(filepath, index=False)

        ''' End of repeated code section (repeats following code for one new metric)'''

        comm_round_time = round_time_df.max(axis=1, numeric_only=True) # Get max time for each round, corresponds to one comm round (straggler)
        comm_round_time.to_csv(join(output_dir, 'comm_' + filename), index=False)

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
            filepath = join(output_dir, filename)
            df.to_csv(filepath, index=False)

    @staticmethod
    def compile_latency(path, output_dir):
        files = [f for f in listdir(path) if f.startswith('latency')]
        results = {}
        for file in files:
            cid = re.search(r'latency_\d+_CID(\d+)', file)
            if not cid:
                continue
            cid = cid.group(1)
            results[f'CID_{cid}'] = pd.read_csv(join(path, file), names=['Round', 'Downlink', 'Uplink'])

        uplink_df = pd.DataFrame()
        downlink_df = pd.DataFrame()

        for cid, df in results.items():
            uplink_df[cid] = df['Uplink']
            downlink_df[cid] = df['Downlink']

        uplink_df.to_csv(join(output_dir, 'latency_uplink.csv'), index=False)
        downlink_df.to_csv(join(output_dir, 'latency_downlink.csv'), index=False)

    @staticmethod
    def parse_rfsts(path, output_dir):
        files = [f for f in listdir(path) if f.startswith('rfsts')]
        results = {}
        for file in files:
            cid = re.search(r'rfsts_\d+_CID(\d+).csv', file)
            if not cid:
                continue
            cid = cid.group(1)
            results[f'CID_{cid}'] = pd.read_csv(join(path, file))

        rsrq_df = pd.DataFrame()
        rssi_df = pd.DataFrame()
        rsrp_df = pd.DataFrame()
        for cid, df in results.items():
            # Include timestamp
            rsrq_df['Timestamp'] = df['timestamp']
            rssi_df['Timestamp'] = df['timestamp']
            rsrp_df['Timestamp'] = df['timestamp']
            rsrq_df[cid] = df['rsrq']
            rssi_df[cid] = df['rssi']
            rsrp_df[cid] = df['rsrp']
        rsrq_df.to_csv(join(output_dir, 'rsrq.csv'), index=False)
        rssi_df.to_csv(join(output_dir, 'rssi.csv'), index=False)
        rsrp_df.to_csv(join(output_dir, 'rsrp.csv'), index=False)


class Aggregator:
    def __init__(self, config_dir):
        self.config_dir = config_dir

    def process(self):
        """
        Process the configuration directory, aggregate the data, and export to CSV.
        """
        output_path = join(self.config_dir, 'output')
        makedirs(output_path, exist_ok=True)

        # Read all metrics files from the output directory
        metrics_files = [f for f in listdir(output_path) if f.endswith('.csv')]

        metrics_dict = {}
        for metric_file in metrics_files:
            metric_name = metric_file.replace('.csv', '')

    def aggregate_ml_metrics(self, metrics_dict, output_path, metric_name):
        """Aggregate metrics across trials by computing the mean for each round."""
        try:
            dfs = list(metrics_dict.values())
            agg_dfs = []
            for df in dfs:
                try:
                    df = df.drop('Round', axis=1)
                except KeyError:
                    pass
                df['avg'] = df.mean(axis=1)
                # Drop all but 'avg'
                agg_dfs.append(df[['avg']])
            pd.concat(agg_dfs, axis=1).to_csv(join(output_path, f'avg_{metric_name}.csv'),
                                              index=False)  # Concat and save as csv
        except Exception as e:
            print(f"Error aggregating ML metrics for {metric_name}: {e}")


    def aggregate_time_metrics(self, metrics_dict, output_path, metric_name):
        try:
            df = pd.concat(metrics_dict.values(), axis=1).T.groupby(level=0).mean().T
            df.to_csv(join(output_path, f'{metric_name}_aggregated.csv'), index=False)
        except Exception as e:
            print(f"Error aggregating time metrics for {metric_name}: {e}")

    def aggregate_phy_metrics(self, metrics_dict, output_path, metric_name):
        pass


class Vizualizer:
    def __init__(self, config_dir):
        self.config_dir = config_dir

    def process(self):
        pass

def process_configuration(config_dir):
    parser = Parser(config_dir)
    parser.process()

    aggregator = Aggregator(config_dir)
    aggregator.process()

def main() -> None:
    input_dir = '/Users/roberthayek/Documents/git_repos/fed_5g/5G_Tests/'

    configuration_dirs = [d for d in listdir(input_dir) if isdir(join(input_dir, d))]

    for config_dir in configuration_dirs:
        config_path = join(input_dir, config_dir)
        process_configuration(config_path)

if __name__ == '__main__':
    main()