import pandas as pd
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import polars as pl
import seaborn as sns
import os

SAVE_LOCALLY = False # whether to save in working directory (lower stakes for debugging)


def parse_gnb_telemetry(trial_data, savepath, separate=True):
    trial_data = trial_data.with_row_index('segment').drop('_id')

    if separate:
        common = trial_data.drop('ues')
        common.write_csv(f'{savepath}/common.csv')

    unpacked = trial_data.explode('ues').unnest('ues')

    for rnti, group in unpacked.group_by('rnti'):
        if separate:
            # Remove all duplicate fields (except 'segment', by which we'll merge) and the rnti column since it's in the filename
            ue_data = group.drop('rnti', set(common.columns) - {'segment'})
            # Save to separate file
            ue_data.write_csv(f'{savepath}/ue_{rnti[0]}.csv')
        else:
            group.write_csv(f'{savepath}/ue_{rnti[0]}.csv')


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
    trial_data = trial_data.join_asof(intervals, on='timestamp', check_sortedness=False)
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


def gather_metrics_by_rnti(savepath, metrics=None):
    ue_dfs = {}
    for file in savepath.iterdir():
        filename = file.name
        if 'ue' in filename and 'common' not in filename:
            rnti = (filename.split('_')[1]).split('.')[0]
            ue_dfs[rnti] = read_data_from_csvs(str(savepath)+'/common.csv', str(file), metrics)
    
    return ue_dfs


def combine_uedfs_by_rnti(ue_dfs, first, second, trial_path, drop_timestamp=True):
    last_pt_first_rnti = ue_dfs[first]['segment'][-1]
    first_pt_second_rnti = ue_dfs[second]['segment'][0]
    try:
        assert(last_pt_first_rnti < first_pt_second_rnti)
    except AssertionError:
        print(f'Exception. Trying to merge non-consecutive physical layer data: {first}\'s first point is {first_pt_second_rnti}. {second}\'s last point is {last_pt_first_rnti}')
        raise

    new_name = first+'-'+second
    print(f'Merging {new_name} (from {first} and {second})')
    ue_dfs[new_name] = pl.concat([ue_dfs[first],ue_dfs[second]], how="vertical_relaxed")
    if drop_timestamp:
        ue_dfs[new_name].drop('timestamp')
    ue_dfs[new_name].write_csv(f'{trial_path}/ue_{new_name}.csv')
    os.remove(f'{trial_path}/ue_{first}.csv')
    os.remove(f'{trial_path}/ue_{second}.csv')
    del ue_dfs[first]
    del ue_dfs[second]
    return new_name


def merge_uedfs_single_disconnect(savepath, drop_timestamp=True, iperf=False): # Combination by only looking at whether one UE dropped and another joined at one time. Will not combine if multiple possible pairs (simultaneous disconnection)
    if iperf:
        ue_dfs = {(file.name.split('_')[1]).split('.')[0] : pl.read_csv(savepath / file.name, try_parse_dates=True) for file in savepath.iterdir() if 'ue' in file.name}
    else:
        try:
            ue_dfs = gather_metrics_by_rnti(savepath)
        except FileNotFoundError as exception:
            print(f'physical layer data not found on savepath {savepath}. Exception: {exception}')
            return
    
    update = True
    while update:
        update = False
        order = sorted(ue_dfs, key=lambda rnti: ue_dfs[rnti]['segment'][-1]) # sort by last element of segment
        last_overall = ue_dfs[order[-1]]['segment'][-1]
        for rnti in order:
            if rnti not in ue_dfs: # may have been merged and no longer exist
                rnti = [name for name in ue_dfs if rnti in name][0] # there should only be one match
            first_pt = ue_dfs[rnti]['segment'][0]
            # print(f'{rnti, first_pt, ue_dfs[rnti]['segment'][-1]}')
            if first_pt != 0:
                possible_pairs = [ue_df for ue_df in ue_dfs if ue_dfs[ue_df]['segment'][-1] < first_pt]
                if len(possible_pairs) == 1:
                    update=True
                    rnti = combine_uedfs_by_rnti(ue_dfs, possible_pairs[0], rnti, savepath, drop_timestamp=drop_timestamp)
            last_pt = ue_dfs[rnti]['segment'][-1]
            if last_pt != last_overall:
                possible_pairs = [ue_df for ue_df in ue_dfs if ue_dfs[ue_df]['segment'][0] > last_pt]
                if len(possible_pairs) == 1:
                    update=True
                    rnti = combine_uedfs_by_rnti(ue_dfs, rnti, possible_pairs[0], savepath, drop_timestamp=drop_timestamp)


def get_runs_list(path, name, cols, start_col, end_col):
    runs_brief = pd.DataFrame()
    for file in Path(path).iterdir():
        filename = file.name
        if name in filename:
            try:
                runs = pd.read_csv(file)#, columns=['Run ID', 'Created At', 'Finished At'])
                runs_brief = runs.loc[:, cols]
                runs_brief[start_col] = pd.to_datetime(runs_brief[start_col].str.strip(),utc=True)
                runs_brief[end_col] = pd.to_datetime(runs_brief[end_col].str.strip(),utc=True)
                runs_brief.name = filename
            except ValueError:
                print(f'ValueError getting runs list for {path.name,filename}')
    
    if runs_brief.empty:
        print([file.name for file in Path(path).iterdir()])
        raise FileNotFoundError('No file containing list of runs in directory')
    
    return runs_brief


def parse(telemetry_dir, trial_dir, sort_telemetry_func, runs):
    source = [file.name for file in Path(telemetry_dir).iterdir() if 'oaibox' in file.name]
    source.sort(key=lambda x: datetime.strptime(x, 'oaibox.telemetry_%m-%d-%y.json'))

    for file in source:
        # print(f'Processing {file}')
        if re.search(r'oaibox\.ue-telemetry.*\.json', file):
            print('Not processing ue-telemetry at this time')
        elif re.search(r'oaibox\.telemetry.*\.json', file):
            try:
                telemetry = pl.read_ndjson(telemetry_dir + file, infer_schema_length=1000).set_sorted('timestamp') # read_json if saved from GUI, read_ndjson if saved from cli
            except pl.exceptions.ComputeError:
                telemetry = pl.read_json(telemetry_dir + file, infer_schema_length=1000).set_sorted('timestamp')
            sort_telemetry_func(runs, telemetry, trial_dir, parse_gnb_telemetry, file)


def sort_telemetry_into_iperf(runs, telemetry_df, run_dir, parser, file):
    telemetry_df=telemetry_df.with_columns(timestamp=pl.from_epoch(telemetry_df['timestamp'], time_unit="ms").dt.replace_time_zone(time_zone="UTC"))
    # print(f'telemetry_df: {telemetry_df['timestamp']}')

    for _, trial in runs.iterrows():
        try:
            telemetry_df = telemetry_df.filter(pl.col('timestamp') >= trial['start'])
            trial_df = telemetry_df.filter(pl.col('timestamp') <= trial['end'])

            if telemetry_df.is_empty():
                return

            # print(f'telemetry_df: {trial_df['timestamp']}, trial start: {trial['start']}, trial end: {trial['end']}')
            if not trial_df.is_empty():
                savepath = f"{run_dir}/{trial['device']}_{runs.name}"
                print(f'saving phys metrics for {savepath}\t{trial_df.shape}')
                Path(savepath).mkdir(parents=True, exist_ok=True)
                parser(trial_df, savepath, False)
        
        except TypeError as exception:
            print(f'\033[91mException: {exception}. Data for trial {trial['Run ID']} cannot be parsed.\033[0m')
            continue


def main():
    dir = '/Users/kmcomer/Documents/5G Experiment Data/'

    # # Parse phys layer data from iperf trials
    # for f in Path(dir).iterdir():
    #     if 'iperf' in f.name and 'zip' not in f.name:
    #         print(f'{f.name}')
    #         for t in f.iterdir():
    #             print(f'Processing {t.name}')
    #             if '0_' in t.name:
    #                 # Merge disconnected/segmented data from same device
    #                 for s in t.iterdir():
    #                     if s.is_dir():
    #                         merge_uedfs_single_disconnect(s, drop_timestamp=False, iperf=True)
                    
    #                 # Parse iperf data from gNB telemetry
    #                 runs = get_runs_list(t, 'UL.csv', ['start', 'device', 'end'], 'start', 'end')
    #                 parse(dir, t, sort_telemetry_into_iperf, runs)
    #                 runs = get_runs_list(t, 'DL.csv', ['start', 'device', 'end'], 'start', 'end')
    #                 parse(dir, t, sort_telemetry_into_iperf, runs)
                    
    
    # # Parse phys layer data from FL experiments
    # runs = get_runs_list(dir, 'Runs', ['Run ID', 'Created At', 'Finished At'], 'Created At', 'Finished At')
    # parse(dir, dir, sort_telemetry_into_trials, runs)

    # Combine RNTIs by name (manually) -- only if certain about data belonging to same device
    # trial_path = Path('/Users/kmcomer/Documents/5G Experiment Data/Phys-layer-unparsed/7222719212451226563_6N_20MHz_7-2_MIMO2x2_Dirichlet')
    # ue_dfs = gather_metrics_by_rnti(trial_path)
    # combine_uedfs_by_rnti(ue_dfs, 'abca', '105c', trial_path) # should return error (105c precedes abca)

    # # Combine RNTIs that must belong to the same device (one disconnect & reconnect only)
    # for path in Path('/Users/kmcomer/Documents/5G Experiment Data/Phys-layer-unparsed').iterdir():
    #     if path.is_dir():
    #         if 'iperf' in str(path):
    #             pass
    #         else:
    #             merge_uedfs_single_disconnect(path / 'phys_layer')

if __name__ == '__main__':
    main()