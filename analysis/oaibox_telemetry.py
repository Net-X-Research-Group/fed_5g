import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import polars as pl
import seaborn as sns

SAVE_LOCALLY = False # whether to save in working directory (lower stakes for debugging)

def parse_gnb_telemetry(trial_data, savepath):
    trial_data = trial_data.with_row_index('segment').drop('_id')

    common = trial_data.drop('ues')
    common.write_csv(f'{savepath}/common.csv')

    unpacked = trial_data.explode('ues').unnest('ues')

    for rnti, group in unpacked.group_by('rnti'):
        # Remove all duplicate fields (except 'segment', by which we'll merge) and the rnti column since it's in the filename
        ue_data = group.drop('rnti', set(common.columns) - {'segment'})
        
        # Save to separate file
        ue_data.write_csv(f'{savepath}/ue_{rnti[0]}.csv')


def filter_out_inactivity(agg_metrics_file, trial_data):
    agg_metrics = pd.read_csv(agg_metrics_file)
    agg_metrics['timestamp'] = pd.to_datetime(agg_metrics['timestamp'], unit='s', utc=True)
    
    agg_metrics = pl.from_pandas(agg_metrics)
    agg_metrics = agg_metrics.with_columns(pl.col('timestamp').dt.cast_time_unit("ms")).set_sorted('timestamp')

    # Initial filter: remove values not within first and last timestamps of agg_metrics
    trial_data.filter(pl.col('timestamp').is_between(agg_metrics['timestamp'][0], agg_metrics['timestamp'][-1]))

    # Identify active intervals using FL data, which collected once per round (agg_metrics). If consecutive data points > 200 seconds apart, the system was down
    intervals = agg_metrics.with_columns(
            time_diff=pl.col('timestamp').diff().dt.total_seconds()
        ).filter(pl.col('time_diff') <= 200).with_columns(
            interval_start=pl.col('timestamp'),
            interval_end=pl.col('timestamp').shift(-1,fill_value=agg_metrics['timestamp'][0])
        ).drop('time_diff')
    print(f'Detected {len(agg_metrics)-len(intervals)} period(s) of downtime')

    # Filter all trial data to exclude data collected during active intervals
    original_size = len(trial_data)
    trial_data = trial_data.join_asof(intervals, on='timestamp')
    trial_data = trial_data.filter(pl.col('timestamp').is_between(pl.col('interval_start'), pl.col('interval_end'), closed='left'))
    trial_data = trial_data.drop('interval_start', 'interval_end')
    print(f'Filtered out {original_size-len(trial_data)} data points collected during downtime')

    return trial_data

def sort_telemetry_into_trials(runs, telemetry_df, runs_dir, parser, file):
    telemetry_df=telemetry_df.with_columns(timestamp=pl.from_epoch(telemetry_df['timestamp'], time_unit="ms").dt.replace_time_zone(time_zone="UTC"))

    for _, trial in runs.iterrows():
        try:
            telemetry_df = telemetry_df.filter(pl.col('timestamp') >= trial['Created At']) # we can always move forward since all datasets are sorted by datetime
        except TypeError as exception:
            print(f'\033[91mException: {exception}. Data for trial {trial['Run ID']} cannot be parsed.\033[0m')
            continue

        if telemetry_df.is_empty():
            print(f'Finished parsing {file}')
            return
        
        trial_data = telemetry_df.filter(pl.col('timestamp') <= trial['Finished At'])
        if not trial_data.is_empty():
            # Look for a matching trial run directory (if provided)
            run_dir = None
            if Path(runs_dir).exists():
                for item in Path(runs_dir).iterdir():
                    if str(trial['Run ID']) in item.name:
                        print(f'Found data for trial {trial['Run ID']}')
                        run_dir = item
                        break
            else:
                print(f'Directory with trials {runs_dir} does not exist')

            # If we found a trial dir, filter out inactivity using its agg metrics
            if run_dir:
                trial_data = filter_out_inactivity(f'{run_dir}/train_agg_metrics.csv', trial_data)
                if SAVE_LOCALLY:
                    run_dir = run_dir.name

                # Create a save path and save the filtered telemetry
                savepath = f"{run_dir}/phys_layer/"
                Path(savepath).mkdir(parents=True, exist_ok=True)

                parser(trial_data, savepath)
            else:
                print(f'Warning: Run dir not found for {trial["Run ID"]}; not saving telemetry')


def read_data_from_csvs(main_fp, secondary_fp, columns=None):
    if secondary_fp:
        df = pl.read_csv(main_fp, columns=['timestamp'], try_parse_dates=True).with_row_index('segment')

        if columns is not None:
            secondary_df = pl.read_csv(secondary_fp, columns=['segment']+columns)
        else:
            secondary_df = pl.read_csv(secondary_fp)
        df = df.join(secondary_df, on='segment', how='right') # here
    else:
        if columns is not None:
            df = pl.read_csv(main_fp, columns=['timestamp']+columns, try_parse_dates=True)
        else:
            df = pl.read_csv(main_fp, try_parse_dates=True)

    return df

def plot_rntis_by_time(ue_dfs, metric, metric_units, run_id, pts_to_plot):
    plt.figure(figsize=(10, 6))
    for rnti, ue_df in ue_dfs.items():
        if ue_df['segment'][0] <= pts_to_plot:
            interval = min(pts_to_plot, len(ue_df))
            plt.scatter(ue_df['timestamp'][:interval], ue_df[metric][:interval], marker='.', label=rnti)
    
    plt.xlabel('Time (s)')
    plt.ylabel(metric + metric_units)
    plt.title(f'{metric} over time for all UEs, trial {run_id}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rntis_distribution(ue_dfs, metric, metric_units, run_id):
    plt.figure(figsize=(10, 6))
    for rnti, ue_df in ue_dfs.items():
        # plt.hist(ue_df[metric], label=rnti)
        sns.kdeplot(data=ue_df[metric], label=rnti)
    
    # plt.xlabel('Time (s)')
    plt.ylabel(metric + metric_units)
    plt.title(f'{metric} distribution for all UEs, trial {run_id}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_agg_distribution(df_agg, metric, metric_units):
    plt.figure(figsize=(10, 6))
    plt.hist(df_agg[metric])
    
    # plt.xlabel('Time (s)')
    plt.ylabel(metric + metric_units)
    plt.title(f'agg {metric} distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_agg_rntis_by_time(df_agg, metric, metric_units, pts_to_plot):
    plt.figure(figsize=(10, 6))
    interval = min(pts_to_plot, len(df_agg))
    plt.scatter(df_agg['timestamp'][:interval], df_agg[metric][:interval], marker='.')
    plt.xlabel('Time (s)')
    plt.ylabel(metric + metric_units)
    plt.title(f'agg {metric} over time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_over_trials(agg_dfs, metric, metric_units):
    plt.figure(figsize=(10, 6))
    for run_id, agg_df in agg_dfs.items():
        plt.hist(agg_df[metric], label=run_id)
    
    # plt.xlabel('Time (s)')
    plt.ylabel(metric + metric_units)
    plt.title(f'{metric} distribution across trials')
    plt.legend()
    plt.grid(True)
    plt.show()

def gather_metrics_by_rnti(savepath, metrics=None):
    try:
        phys_layer = savepath.joinpath('phys_layer')
    except FileNotFoundError as exception:
        print(f'phys_layer directory not found on savepath {savepath}. Exception: {exception}')
        return
    
    ue_dfs = {}
    for file in phys_layer.iterdir():
        filename = file.name
        if 'ue' in filename and 'common' not in filename:
            rnti = (filename.split('_')[1]).split('.')[0]
            ue_dfs[rnti] = read_data_from_csvs(str(phys_layer)+'/common.csv', str(file), metrics)
    
    return ue_dfs

def combine_rntis(savepath):
    ue_dfs = gather_metrics_by_rnti(savepath)
    order = sorted(ue_dfs, key=lambda rnti: ue_dfs[rnti]['segment'][-1]) # sort by last element of segment
    for rnti in order:
        # print(f'{df, ue_dfs[df]['segment'][0], ue_dfs[df]['segment'][-1]}')
        first_pt = ue_dfs[rnti]['segment'][0]
        if first_pt != 0:
            possible_pairs = [ue_df for ue_df in ue_dfs if ue_dfs[ue_df]['segment'][-1] <= first_pt]
            if len(possible_pairs) == 1:
                pair = possible_pairs[0]
                ue_dfs[pair+'_'+rnti] = pl.concat([ue_dfs[pair],ue_dfs[rnti]])
                del ue_dfs[pair]
                del ue_dfs[rnti]
            # print(f'{rnti} possible pairs: {possible_pairs}')

def get_runs_list(path):
    runs_brief = pd.DataFrame()
    for file in Path(path).iterdir():
        filename = file.name
        if "Runs" in filename:
            runs = pd.read_csv(file)#, columns=['Run ID', 'Created At', 'Finished At'])
            runs_brief = runs.loc[:, ['Run ID', 'Created At', 'Finished At']]
            runs_brief['Created At'] = pd.to_datetime(runs_brief['Created At'].str.strip())
            runs_brief['Finished At'] = pd.to_datetime(runs_brief['Finished At'].str.strip())
    
    if runs_brief.empty: # pd dataframe
        raise FileNotFoundError('No file containing list of runs in directory')
    
    return runs_brief

def plot(dir):
    # metric = 'ulMcs' 
    agg_dfs = {}
    # metrics = ['ulMcs', 'dlMcs', 'rssi', 'rsrp', 'rsrq', 'dlBler', 'ulQm', 'dlQm', 'ulBler', 'phr', 'pcmax', 'sinr', 'pucchSnr', 'cqi', 'puschSnr']
    metrics = ['rsrp']
    pts_to_plot = 500
    avgs = {metric: {} for metric in metrics}
    stds = {metric: {} for metric in metrics}
    for path in Path(dir).iterdir():
        if path.is_dir():
            ue_dfs = gather_metrics_by_rnti(path, metrics)

            if ue_dfs:
                for metric in metrics:
                    plot_rntis_by_time(ue_dfs, metric, metric_units='', run_id=path.name, pts_to_plot=pts_to_plot)
                    # plot_rntis_distribution(ue_dfs, metric, metric_units='', run_id=path.name)
    
    for metric in avgs:
        print(f'trial, \t\t\t mean \t\t\t std \t\t\t (metric: {metric})')
        for trial in avgs[metric]:
            print(f'{trial} \t {avgs[metric][trial]} \t {stds[metric][trial]}')
    # plot_over_trials(agg_dfs, metric, metric_units='')

def parse(dir):
    runs = get_runs_list(dir)

    source = [file.name for file in Path(dir).iterdir() if 'oaibox' in file.name]
    source.sort(key=lambda x: datetime.strptime(x, 'oaibox.telemetry_%m-%d-%y.json'))

    for file in source:
        print(f'Processing {file}')
        if re.search(r'oaibox\.ue-telemetry.*\.json', file):
            print('Not processing ue-telemetry at this time')
        elif re.search(r'oaibox\.telemetry.*\.json', file):
            telemetry = pl.read_json(dir + file, infer_schema_length=1000).set_sorted('timestamp')
            sort_telemetry_into_trials(runs, telemetry, dir, parse_gnb_telemetry, file)

def main():
    # parse('/Users/kmcomer/Documents/5G Experiment Data/Phys-layer-unparsed/')
    for path in Path('/Users/kmcomer/Documents/5G Experiment Data/Phys-layer-unparsed/').iterdir():
        if path.is_dir():
            combine_rntis(path)
    # plot('/Users/kmcomer/Documents/5G Experiment Data/Phys-layer-unparsed/') # TODO integrate with main.py for filtering based on trial params

if __name__ == '__main__':
    main()
