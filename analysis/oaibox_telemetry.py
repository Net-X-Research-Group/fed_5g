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
from data_loading import parse_experiment_name
from helpers import sort_experiments_by_sweep, format_sweep_label

SAVE_LOCALLY = False # whether to save in working directory (lower stakes for debugging)

THROUGHPUT_METRICS = [
    'ul_throughput_bps',
    'dl_throughput_bps',
    'ul_throughput_mbps',
    'dl_throughput_mbps',
]

DEFAULT_NON_ZERO_METRICS = set(THROUGHPUT_METRICS)


def _normalize_non_zero_metrics(non_zero_metrics):
    if non_zero_metrics is None:
        return DEFAULT_NON_ZERO_METRICS
    return set(non_zero_metrics)


def _normalize_min_thresholds(min_thresholds):
    if min_thresholds is None:
        return {}
    return dict(min_thresholds)


def _metric_values(ue_df, metric, non_zero_metrics=None, min_thresholds=None):
    if metric not in ue_df.columns:
        return []

    ignore_zero = metric in _normalize_non_zero_metrics(non_zero_metrics)
    threshold = _normalize_min_thresholds(min_thresholds).get(metric)
    values = []
    for value in ue_df[metric].to_list():
        if value is None or pd.isna(value):
            continue

        numeric_value = pd.to_numeric(value, errors='coerce')
        if pd.isna(numeric_value):
            continue

        if ignore_zero and (numeric_value < 0 or np.isclose(numeric_value, 0.0, atol=1e-12)):
            continue
        if threshold is not None and numeric_value < threshold:
            continue
        values.append(float(numeric_value))
    return values


def _build_distribution_dataframe(ue_dfs, metric, paired_metric=None, non_zero_metrics=None, min_thresholds=None):
    rows = []
    for rnti, ue_df in ue_dfs.items():
        if metric not in ue_df.columns:
            continue

        values = _metric_values(ue_df, metric, non_zero_metrics, min_thresholds=min_thresholds)
        for value in values:
            row = {'device': str(rnti), 'value': value, 'series': metric}
            if paired_metric is not None:
                row['direction'] = 'UL'
            rows.append(row)

        if paired_metric is not None and paired_metric in ue_df.columns:
            paired_values = _metric_values(ue_df, paired_metric, non_zero_metrics, min_thresholds=min_thresholds)
            for value in paired_values:
                rows.append({
                    'device': str(rnti),
                    'value': value,
                    'series': paired_metric,
                    'direction': 'DL',
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)

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

def plot_rntis_by_time(ue_dfs, metric, metric_units, run_id, pts_to_plot, savepath=None, show=True, non_zero_metrics=None, min_thresholds=None):
    fig = plt.figure(figsize=(10, 6))
    for rnti, ue_df in ue_dfs.items():
        values = _metric_values(ue_df, metric, non_zero_metrics, min_thresholds=min_thresholds)
        if not values:
            continue

        if ue_df['segment'][0] <= pts_to_plot:
            series = ue_df.select(['timestamp', metric]).to_pandas()
            if metric in _normalize_non_zero_metrics(non_zero_metrics):
                series[metric] = pd.to_numeric(series[metric], errors='coerce')
                series = series[(series[metric] > 0) & (~np.isclose(series[metric], 0.0, atol=1e-12))]
            metric_threshold = _normalize_min_thresholds(min_thresholds).get(metric)
            if metric_threshold is not None:
                series[metric] = pd.to_numeric(series[metric], errors='coerce')
                series = series[series[metric] >= metric_threshold]
            series = series.dropna(subset=[metric]).head(pts_to_plot)
            if series.empty:
                continue
            plt.scatter(series['timestamp'], series[metric], marker='.', label=rnti)
    
    plt.xlabel('Time (s)')
    plt.ylabel(metric + metric_units)
    plt.title(f'{metric} over time for all UEs, trial {run_id}')
    plt.legend()
    plt.grid(True)

    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_rntis_distribution(
    ue_dfs,
    metric,
    metric_units,
    run_id,
    plot_type='kde',
    paired_metric=None,
    split_violin=True,
    savepath=None,
    show=True,
    non_zero_metrics=None,
    min_thresholds=None,
):
    df = _build_distribution_dataframe(
        ue_dfs,
        metric,
        paired_metric=paired_metric,
        non_zero_metrics=non_zero_metrics,
        min_thresholds=min_thresholds,
    )
    if df.empty:
        return

    kind_map = {
        'violin': 'violin',
        'box': 'box',
        'count': 'count',
        'bar': 'bar',
        # Use bar as a catplot-compatible fallback for previous kde option.
        'kde': 'bar',
    }
    kind = kind_map.get(plot_type, 'violin')

    catplot_kwargs = {
        'data': df,
        'x': 'device',
        'kind': kind,
        'height': 6,
        'aspect': 1.8,
    }

    if paired_metric is not None:
        catplot_kwargs['hue'] = 'direction'

    if kind in {'violin', 'box', 'bar'}:
        catplot_kwargs['y'] = 'value'

    if kind == 'violin':
        catplot_kwargs['inner'] = 'quart'
        catplot_kwargs['cut'] = 0
        if paired_metric is not None:
            catplot_kwargs['split'] = split_violin
        catplot_kwargs['gap'] = 0.1

    try:
        g = sns.catplot(**catplot_kwargs)
    except TypeError:
        # Fallback for seaborn versions that do not support some kwargs like `gap`.
        catplot_kwargs.pop('gap', None)
        g = sns.catplot(**catplot_kwargs)

    ax = g.ax
    if paired_metric is not None:
        ax.set_ylabel(f'{metric} / {paired_metric}{metric_units}')
        ax.set_title(f'{metric} vs {paired_metric} by device, trial {run_id}')
    else:
        ax.set_ylabel(metric + metric_units)
        ax.set_title(f'{metric} distribution for all UEs, trial {run_id}')
    ax.set_xlabel('Device')
    ax.grid(True)

    if savepath:
        g.savefig(savepath, format='svg', dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(g.fig)

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

def _metrics_to_load(metrics):
    if metrics is None:
        return None

    derived = set(THROUGHPUT_METRICS)
    load_metrics = [m for m in metrics if m not in derived]
    if any(m in derived for m in metrics):
        load_metrics.extend(['ulBytes', 'dlBytes'])

    # preserve order while removing duplicates
    deduped = list(dict.fromkeys(load_metrics))
    return deduped


def add_throughput_columns(ue_df):
    required = {'timestamp', 'ulBytes', 'dlBytes'}
    if not required.issubset(set(ue_df.columns)):
        return ue_df

    ue_df = ue_df.sort('timestamp').with_columns(
        delta_t_s=pl.col('timestamp').diff().dt.total_seconds(),
        ul_bytes_delta=pl.col('ulBytes').diff(),
        dl_bytes_delta=pl.col('dlBytes').diff(),
    ).with_columns(
        # Counter resets or duplicate timestamps create invalid/negative throughput; ignore those points.
        ul_throughput_bps=pl.when(
            (pl.col('delta_t_s') > 0) & (pl.col('ul_bytes_delta') >= 0)
        ).then((pl.col('ul_bytes_delta') * 8) / pl.col('delta_t_s')).otherwise(None),
        dl_throughput_bps=pl.when(
            (pl.col('delta_t_s') > 0) & (pl.col('dl_bytes_delta') >= 0)
        ).then((pl.col('dl_bytes_delta') * 8) / pl.col('delta_t_s')).otherwise(None),
    ).with_columns(
        ul_throughput_mbps=pl.col('ul_throughput_bps') / 1_000_000,
        dl_throughput_mbps=pl.col('dl_throughput_bps') / 1_000_000,
    )

    return ue_df


def summarize_throughput_by_device(ue_dfs, run_label, non_zero_metrics=None, min_thresholds=None):
    rows = []
    for rnti, ue_df in ue_dfs.items():
        if not {'ul_throughput_mbps', 'dl_throughput_mbps'}.issubset(set(ue_df.columns)):
            continue

        ul_values = _metric_values(ue_df, 'ul_throughput_mbps', non_zero_metrics, min_thresholds=min_thresholds)
        dl_values = _metric_values(ue_df, 'dl_throughput_mbps', non_zero_metrics, min_thresholds=min_thresholds)

        ul_avg = pd.Series(ul_values).mean() if ul_values else None
        dl_avg = pd.Series(dl_values).mean() if dl_values else None
        ul_std = pd.Series(ul_values).std() if len(ul_values) > 1 else None
        dl_std = pd.Series(dl_values).std() if len(dl_values) > 1 else None
        rows.append({
            'run_id': run_label,
            'device': rnti,
            'avg_ul_throughput_mbps': ul_avg,
            'std_ul_throughput_mbps': ul_std,
            'avg_dl_throughput_mbps': dl_avg,
            'std_dl_throughput_mbps': dl_std,
        })

    return rows


def gather_metrics_by_rnti(savepath, metrics=None):
    try:
        phys_layer = savepath.joinpath('phys_layer')
    except FileNotFoundError as exception:
        print(f'phys_layer directory not found on savepath {savepath}. Exception: {exception}')
        return
    
    ue_dfs = {}
    load_metrics = _metrics_to_load(metrics)
    for file in phys_layer.iterdir():
        filename = file.name
        if 'ue' in filename and 'common' not in filename:
            rnti = (filename.split('_')[1]).split('.')[0]
            ue_df = read_data_from_csvs(str(phys_layer)+'/common.csv', str(file), load_metrics)
            ue_dfs[rnti] = add_throughput_columns(ue_df)
    
    return ue_dfs

def combine_rntis(savepath):
    try:
        ue_dfs = gather_metrics_by_rnti(savepath)
    except FileNotFoundError as exception:
        print(f'phys_layer directory not found on savepath {savepath}. Exception: {exception}')
        return
    
    update = True
    while update:
        update = False
        order = sorted(ue_dfs, key=lambda rnti: ue_dfs[rnti]['segment'][-1]) # sort by last element of segment
        last_overall = ue_dfs[order[-1]]['segment'][-1]
        for rnti in order:
            # print(f'{df, ue_dfs[df]['segment'][0], ue_dfs[df]['segment'][-1]}')
            if rnti not in ue_dfs: # may have been merged and no longer exist
                rnti = [name for name in ue_dfs if rnti in name][0] # there should only be one match
            first_pt = ue_dfs[rnti]['segment'][0]
            if first_pt != 0:
                possible_pairs = [ue_df for ue_df in ue_dfs if ue_dfs[ue_df]['segment'][-1] <= first_pt]
                if len(possible_pairs) == 1:
                    update=True
                    pair = possible_pairs[0]
                    new_name = pair+'-'+rnti
                    ue_dfs[new_name] = pl.concat([ue_dfs[pair],ue_dfs[rnti]], how="vertical_relaxed")
                    ue_dfs[new_name].drop('timestamp').write_csv(f'{savepath}/phys_layer/ue_{new_name}.csv')
                    os.remove(f'{savepath}/phys_layer/ue_{rnti}.csv')
                    os.remove(f'{savepath}/phys_layer/ue_{pair}.csv')
                    del ue_dfs[pair]
                    del ue_dfs[rnti]
                    rnti = new_name
            last_pt = ue_dfs[rnti]['segment'][-1]
            if last_pt != last_overall:
                possible_pairs = [ue_df for ue_df in ue_dfs if ue_dfs[ue_df]['segment'][0] >= last_pt] # TODO fix
                if len(possible_pairs) == 1:
                    update=True
                    pair = possible_pairs[0]
                    new_name = rnti+'-'+pair
                    ue_dfs[new_name] = pl.concat([ue_dfs[rnti],ue_dfs[pair]], how="vertical_relaxed")
                    ue_dfs[new_name].drop('timestamp').write_csv(f'{savepath}/phys_layer/ue_{new_name}.csv')
                    os.remove(f'{savepath}/phys_layer/ue_{rnti}.csv')
                    os.remove(f'{savepath}/phys_layer/ue_{pair}.csv')
                    del ue_dfs[pair]
                    del ue_dfs[rnti]
                # print(f'{rnti} possible pairs: {possible_pairs}')

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

def build_experiment_index(root_dir):
    experiments = []
    for exp in Path(root_dir).iterdir():
        if not exp.is_dir():
            continue
        params = parse_experiment_name(exp.name)
        experiments.append({'path': exp, **params})
    return experiments


def filter_experiments(experiment_paths, filters):
    return [
        exp for exp in experiment_paths
        if all(
            exp.get(k) is None or (exp.get(k) in v if isinstance(v, list) else exp.get(k) == v)
            for k, v in filters.items()
        )
    ]


def plot(
    dir,
    filters=None,
    sweep_param=None,
    metrics=None,
    pts_to_plot=1000,
    plot_mode='distribution',
    distribution_plot_type='violin', #box,violin
    pair_ul_dl=False,
    non_zero_metrics=None,
    min_thresholds=None,
):
    # metrics = ['ulMcs', 'dlMcs', 'rssi', 'rsrp', 'rsrq', 'dlBler', 'ulQm', 'dlQm', 'ulBler', 'phr', 'pcmax', 'sinr', 'pucchSnr', 'cqi', 'puschSnr']
    # segment\tulBler\tphr\tpucchSnr\tueId\tdlMcs\tulMcs\tulQm\tri\tranUeId\tdlBytes\tdlQm\tcqi\tpuschSnr\tinSync\tpmi\tsinr\tpcmax\trsrq\trsrp\trssi\tamfUeId\tdlBler\tulBytes
    metrics = metrics or ['rsrp']
    non_zero_metrics = _normalize_non_zero_metrics(non_zero_metrics)
    min_thresholds = _normalize_min_thresholds(min_thresholds)
    avgs = {metric: {} for metric in metrics}
    stds = {metric: {} for metric in metrics}
    throughput_summary = []

    experiments = build_experiment_index(dir)
    if filters:
        experiments = filter_experiments(experiments, filters)

    if sweep_param:
        experiments = sort_experiments_by_sweep(experiments, sweep_param)

    print(f'Plotting phys layer data for {len(experiments)} experiment(s)')
    if not experiments:
        print('No experiments matched the given filters')
        return

    for exp in experiments:
        path = exp['path']
        ue_dfs = gather_metrics_by_rnti(path, metrics)

        if not ue_dfs:
            continue

        if sweep_param:
            run_label = format_sweep_label(sweep_param, exp[sweep_param], exp)
        else:
            run_label = path.name

        processed_metrics = set()
        for metric in metrics:
            if metric in processed_metrics:
                continue

            paired_metric = None
            if pair_ul_dl and metric.startswith('ul_'):
                candidate = 'dl_' + metric[3:]
                if candidate in metrics:
                    paired_metric = candidate
                    processed_metrics.add(candidate)

            if plot_mode == 'time':
                plot_rntis_by_time(
                    ue_dfs,
                    metric,
                    metric_units='',
                    run_id=run_label,
                    pts_to_plot=pts_to_plot,
                    non_zero_metrics=non_zero_metrics,
                    min_thresholds=min_thresholds,
                )
            else:
                plot_rntis_distribution(
                    ue_dfs,
                    metric,
                    metric_units='',
                    run_id=run_label,
                    plot_type=distribution_plot_type,
                    paired_metric=paired_metric,
                    split_violin=True,
                    non_zero_metrics=non_zero_metrics,
                    min_thresholds=min_thresholds,
                )

            processed_metrics.add(metric)

        throughput_summary.extend(summarize_throughput_by_device(
            ue_dfs,
            run_label,
            non_zero_metrics=non_zero_metrics,
            min_thresholds=min_thresholds,
        ))

    if throughput_summary:
        print('\nThroughput by device (Mbps): mean and std')
        print(pd.DataFrame(throughput_summary).to_string(index=False))

    for metric in avgs:
        print(f'trial, \t\t\t mean \t\t\t std \t\t\t (metric: {metric})')
        for trial in avgs[metric]:
            print(f'{trial} \t {avgs[metric][trial]} \t {stds[metric][trial]}')
    # plot_over_trials(agg_dfs, metric, metric_units='')


def plot_cellular_sweep_phys(
    experiment_paths,
    filters,
    sweep,
    output_dir,
    metrics=None,
    pts_to_plot=1000,
    plot_mode='time',
    distribution_plot_type='violin',
    pair_ul_dl=False,
    non_zero_metrics=None,
    min_thresholds=None,
):
    filtered = filter_experiments(experiment_paths, filters)
    filtered = sort_experiments_by_sweep(filtered, sweep)

    filter_parts = [f"{k}_{str(v).replace('/', '_').replace(' ', '_')}" for k, v in filters.items()]
    filter_dir = '_'.join(filter_parts)
    sweep_output_dir = output_dir / filter_dir / sweep
    sweep_output_dir.mkdir(exist_ok=True, parents=True)

    metrics = metrics or ['rsrp', 'ul_throughput_mbps', 'dl_throughput_mbps']
    non_zero_metrics = _normalize_non_zero_metrics(non_zero_metrics)
    min_thresholds = _normalize_min_thresholds(min_thresholds)
    throughput_summary = []

    print(f'Generating saved phys-layer plots for {len(filtered)} experiment(s)')
    for exp in filtered:
        ue_dfs = gather_metrics_by_rnti(exp['path'], metrics)
        if not ue_dfs:
            continue

        run_label = format_sweep_label(sweep, exp[sweep], exp)
        processed_metrics = set()
        for metric in metrics:
            if metric in processed_metrics:
                continue

            paired_metric = None
            if pair_ul_dl and metric.startswith('ul_'):
                candidate = 'dl_' + metric[3:]
                if candidate in metrics:
                    paired_metric = candidate
                    processed_metrics.add(candidate)

            if plot_mode == 'time':
                save_file = sweep_output_dir / f'{metric}_by_time.svg'
                plot_rntis_by_time(
                    ue_dfs,
                    metric,
                    metric_units='',
                    run_id=run_label,
                    pts_to_plot=pts_to_plot,
                    savepath=save_file,
                    show=False,
                    non_zero_metrics=non_zero_metrics,
                    min_thresholds=min_thresholds,
                )
            else:
                if paired_metric:
                    save_file = sweep_output_dir / f'{metric}_vs_{paired_metric}_{distribution_plot_type}.svg'
                else:
                    save_file = sweep_output_dir / f'{metric}_{distribution_plot_type}.svg'

                plot_rntis_distribution(
                    ue_dfs,
                    metric,
                    metric_units='',
                    run_id=run_label,
                    plot_type=distribution_plot_type,
                    paired_metric=paired_metric,
                    split_violin=True,
                    savepath=save_file,
                    show=False,
                    non_zero_metrics=non_zero_metrics,
                    min_thresholds=min_thresholds,
                )

            processed_metrics.add(metric)

        throughput_summary.extend(summarize_throughput_by_device(
            ue_dfs,
            run_label,
            non_zero_metrics=non_zero_metrics,
            min_thresholds=min_thresholds,
        ))

    if throughput_summary:
        throughput_df = pd.DataFrame(throughput_summary)
        throughput_fp = sweep_output_dir / 'throughput_summary_by_device.csv'
        throughput_df.to_csv(throughput_fp, index=False)
        print(f'Saved throughput summary to {throughput_fp}')

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
    dir = '/Users/kmcomer/Documents/5G Experiment Data/Phys-layer-unparsed/'
    # for f in Path(dir).iterdir():
    #     if 'iperf' in f.name:
    #         print(f'{f.name}')
    #         for t in f.iterdir():
    #             print(f'Processing {t.name}')
    #             if '0_' in t.name:
    #                 runs = get_runs_list(t, 'UL.csv', ['start', 'device', 'end'], 'start', 'end')
    #                 parse(dir, t, sort_telemetry_into_iperf, runs)
    #                 runs = get_runs_list(t, 'DL.csv', ['start', 'device', 'end'], 'start', 'end')
    #                 parse(dir, t, sort_telemetry_into_iperf, runs)

    # runs = get_runs_list(dir, 'Runs', ['Run ID', 'Created At', 'Finished At'], 'Created At', 'Finished At')
    # parse(dir, dir, sort_telemetry_into_trials, runs)
    # for path in Path('/Users/kmcomer/Documents/5G Experiment Data/Phys-layer-unparsed/').iterdir():
    #     if path.is_dir():
    #         if 'iperf' in str(path):
    #             pass
    #         else:
    #             combine_rntis(path)
    data_dir = Path('/Users/kmcomer/Documents/5G Experiment Data/FedAvg/')
    all_experiments = build_experiment_index(data_dir)

    filters = {
        'bandwidth': '40 MHz',
        'rank': '2x2',
        'distribution': 'dirichlet',
        'congestion': False,
        'tdd': '7-2',
        'nodes': '6N'
    }

    plot(
        str(data_dir),
        filters=filters,
        sweep_param='network',
        metrics=['ul_throughput_mbps', 'dl_throughput_mbps'],#['rsrp', 'rssi', 'rsrq', 'ul_throughput_mbps', 'dl_throughput_mbps'],
        plot_mode='distribution',
        distribution_plot_type='violin',
        pair_ul_dl=True,
        non_zero_metrics=['ul_throughput_mbps', 'dl_throughput_mbps'],
        min_thresholds={
        'ul_throughput_mbps': 0.01,
        'dl_throughput_mbps': 0.01,
    },
    )

    plot_cellular_sweep_phys(
        all_experiments,
        filters=filters,
        sweep='network',
        output_dir=Path.cwd() / 'phys_layer_plots',
        metrics=['rsrp', 'rssi', 'rsrq', 'ul_throughput_mbps', 'dl_throughput_mbps'],
        plot_mode='distribution',
        distribution_plot_type='violin',
        pair_ul_dl=True,
        non_zero_metrics=['ul_throughput_mbps', 'dl_throughput_mbps'],
    )

if __name__ == '__main__':
    main()
