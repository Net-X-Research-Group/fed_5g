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


def _normalize_phase_filter(phase_filter):
    if phase_filter is None:
        return None
    if isinstance(phase_filter, str):
        if phase_filter.lower() in {'all', '*'}:
            return None
        return [phase_filter]
    phase_list = list(phase_filter)
    if not phase_list:
        return None
    return phase_list


def _cache_paths(cache_dir, exp_name, rnti):
    exp_cache_dir = Path(cache_dir) / re.sub(r'[^a-zA-Z0-9_.-]', '_', str(exp_name))
    exp_cache_dir.mkdir(parents=True, exist_ok=True)
    round_fp = exp_cache_dir / 'rounds.csv'
    ue_fp = exp_cache_dir / f'ue_{rnti}_round_filtered.csv'
    return round_fp, ue_fp


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
        

def _get_duration_column(df, candidates, default=0.0):
    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce').fillna(default)
            return np.maximum(values, 0.0)
    return pd.Series(default, index=df.index, dtype='float64')


def build_round_windows(agg_metrics_file, max_gap_s=200):
    agg_metrics = pd.read_csv(agg_metrics_file)
    agg_metrics['timestamp'] = pd.to_datetime(agg_metrics['timestamp'], unit='s', utc=True)
    agg_metrics = agg_metrics.sort_values('timestamp').reset_index(drop=True)

    # Identify downtime by large gaps and derive per-round duration.
    gap_s = agg_metrics['timestamp'].diff().dt.total_seconds()
    agg_metrics['round_duration'] = _get_duration_column(
        agg_metrics,
        candidates=['round_duration', 'round_time', 'duration_s'],
        default=np.nan,
    )

    if agg_metrics['round_duration'].isna().all():
        inferred = gap_s.copy()
        inferred.iloc[0] = np.nan
        fallback = np.nanmedian(inferred.to_numpy(dtype=float))
        if np.isnan(fallback):
            fallback = 0.0
        agg_metrics['round_duration'] = inferred.fillna(fallback)
    else:
        fallback = np.nanmedian(agg_metrics['round_duration'].to_numpy(dtype=float))
        if np.isnan(fallback):
            fallback = 0.0
        agg_metrics['round_duration'] = pd.to_numeric(agg_metrics['round_duration'], errors='coerce').fillna(fallback)

    active_mask = gap_s.isna() | (gap_s <= max_gap_s)
    rounds = agg_metrics.loc[active_mask].copy().reset_index(drop=True)

    rounds['round_id'] = np.arange(len(rounds), dtype=int)
    rounds['round_end'] = rounds['timestamp']
    rounds['round_start'] = rounds['round_end'] - pd.to_timedelta(rounds['round_duration'], unit='s')

    # Phase durations (seconds). Missing columns default to 0.
    rounds['downlink_s'] = _get_duration_column(rounds, ['downlink_latency', 'downlink_time', 'dl_time_s'])
    rounds['train_s'] = _get_duration_column(rounds, ['train_time', 'local_train_time', 'training_time'])
    rounds['eval_s'] = _get_duration_column(rounds, ['eval_time', 'evaluation_time'])
    rounds['uplink_s'] = _get_duration_column(rounds, ['uplink_latency', 'uplink_time', 'ul_time_s'])

    # Build phase boundaries from round start.
    rounds['downlink_start'] = rounds['round_start']
    rounds['downlink_end'] = rounds['downlink_start'] + pd.to_timedelta(rounds['downlink_s'], unit='s')
    rounds['training_start'] = rounds['downlink_end']
    rounds['training_end'] = rounds['training_start'] + pd.to_timedelta(rounds['train_s'], unit='s')
    rounds['evaluation_start'] = rounds['training_end']
    rounds['evaluation_end'] = rounds['evaluation_start'] + pd.to_timedelta(rounds['eval_s'], unit='s')
    rounds['uplink_start'] = rounds['evaluation_end']
    rounds['uplink_end'] = rounds['uplink_start'] + pd.to_timedelta(rounds['uplink_s'], unit='s')

    # Idle starts after uplink and ends at round_end.
    rounds['idle_start'] = rounds['uplink_end']
    rounds['idle_end'] = rounds['round_end']

    return rounds


def annotate_telemetry_with_rounds_and_phases(trial_data, rounds):
    if rounds.empty:
        return trial_data

    rounds_pl = pl.from_pandas(rounds[[
        'round_id',
        'round_start',
        'round_end',
        'downlink_end',
        'training_end',
        'evaluation_end',
        'uplink_end',
    ]])

    ts_dtype = trial_data.schema.get('timestamp')
    if ts_dtype is not None:
        rounds_pl = rounds_pl.with_columns(
            pl.col('round_start').cast(ts_dtype),
            pl.col('round_end').cast(ts_dtype),
            pl.col('downlink_end').cast(ts_dtype),
            pl.col('training_end').cast(ts_dtype),
            pl.col('evaluation_end').cast(ts_dtype),
            pl.col('uplink_end').cast(ts_dtype),
        )
    rounds_pl = rounds_pl.set_sorted('round_start')

    annotated = trial_data.join_asof(
        rounds_pl,
        left_on='timestamp',
        right_on='round_start',
        strategy='backward',
        check_sortedness=False,
    )

    annotated = annotated.filter(
        pl.col('round_id').is_not_null() &
        (pl.col('timestamp') <= pl.col('round_end'))
    )

    annotated = annotated.with_columns(
        phase=pl.when(pl.col('timestamp') <= pl.col('downlink_end')).then(pl.lit('downlink'))
        .when(pl.col('timestamp') <= pl.col('training_end')).then(pl.lit('training'))
        .when(pl.col('timestamp') <= pl.col('evaluation_end')).then(pl.lit('evaluation'))
        .when(pl.col('timestamp') <= pl.col('uplink_end')).then(pl.lit('uplink'))
        .otherwise(pl.lit('idle')),
        round_elapsed_s=(pl.col('timestamp') - pl.col('round_start')).dt.total_seconds(),
        round_duration_s=(pl.col('round_end') - pl.col('round_start')).dt.total_seconds(),
    )

    annotated = annotated.with_columns(
        round_t=pl.when(pl.col('round_duration_s') > 0)
        .then(pl.col('round_elapsed_s') / pl.col('round_duration_s'))
        .otherwise(None),
    )

    return annotated


def compute_round_average_profile(annotated_trial_data, metric, n_points=100, phase_filter=None, round_ids=None):
    if metric not in annotated_trial_data.columns:
        return pd.DataFrame()

    df = annotated_trial_data.to_pandas()
    phase_list = _normalize_phase_filter(phase_filter)
    if phase_list is not None:
        df = df[df['phase'].isin(phase_list)]
    if round_ids is not None:
        df = df[df['round_id'].isin(round_ids)]

    df = df.dropna(subset=['round_id', 'round_t', metric])
    if df.empty:
        return pd.DataFrame()

    grid = np.linspace(0.0, 1.0, n_points)
    aligned = []

    for round_id, grp in df.groupby('round_id'):
        grp = grp.sort_values('round_t')
        x = grp['round_t'].to_numpy(dtype=float)
        y = pd.to_numeric(grp[metric], errors='coerce').to_numpy(dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if len(x) < 2:
            continue

        keep = np.r_[True, np.diff(x) > 0]
        x = x[keep]
        y = y[keep]
        if len(x) < 2:
            continue

        interp = np.interp(grid, x, y)
        aligned.append(interp)

    if not aligned:
        return pd.DataFrame()

    aligned = np.vstack(aligned)
    return pd.DataFrame({
        'round_t': grid,
        'mean': np.nanmean(aligned, axis=0),
        'std': np.nanstd(aligned, axis=0),
    })


def plot_round_average_across_devices(
    ue_dfs,
    metric,
    round_ids=None,
    n_points=100,
    phase_filter=None,
    show_error_bars=False,
    errorbar_step=10,
    include_effective_sum=False,
    effective_on_secondary_axis=True,
    run_label=None,
    savepath=None,
    show=True,
):
    phase_list = _normalize_phase_filter(phase_filter)
    phase_suffix = '' if phase_list is None else f" ({'/'.join(phase_list)})"
    effective_allowed = metric in THROUGHPUT_METRICS

    fig, ax = plt.subplots(figsize=(11, 5))
    plotted = 0
    effective_sum = None
    round_t_grid = None
    round_counts = {}

    for device_id, ue_df in ue_dfs.items():
        if 'round_id' not in ue_df.columns:
            continue

        df_dev = ue_df.to_pandas()
        phase_dev = _normalize_phase_filter(phase_filter)
        if phase_dev is not None and 'phase' in df_dev.columns:
            df_dev = df_dev[df_dev['phase'].isin(phase_dev)]
        if round_ids is not None and 'round_id' in df_dev.columns:
            df_dev = df_dev[df_dev['round_id'].isin(round_ids)]
        if 'round_id' in df_dev.columns:
            round_counts[str(device_id)] = int(df_dev['round_id'].nunique())

        profile = compute_round_average_profile(
            ue_df,
            metric,
            n_points=n_points,
            phase_filter=phase_filter,
            round_ids=round_ids,
        )
        if profile.empty:
            continue

        ax.plot(profile['round_t'], profile['mean'], label=str(device_id), linewidth=2)
        if show_error_bars:
            step = max(1, int(errorbar_step))
            idx = np.arange(0, len(profile), step)
            ax.errorbar(
                profile['round_t'].to_numpy()[idx],
                profile['mean'].to_numpy()[idx],
                yerr=profile['std'].to_numpy()[idx],
                fmt='none',
                alpha=0.35,
                capsize=2,
            )

        if include_effective_sum and effective_allowed:
            if effective_sum is None:
                effective_sum = np.zeros(len(profile), dtype=float)
                round_t_grid = profile['round_t'].to_numpy(dtype=float)
            effective_sum += profile['mean'].to_numpy(dtype=float)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    handles, labels = ax.get_legend_handles_labels()
    if include_effective_sum and effective_allowed and effective_sum is not None:
        if effective_on_secondary_axis:
            ax2 = ax.twinx()
            eff_line = ax2.plot(
                round_t_grid,
                effective_sum,
                color='black',
                linestyle='--',
                linewidth=2,
                label='effective total',
            )
            ax2.set_ylabel(f'effective {metric}')
            handles += eff_line
            labels += ['effective total']
        else:
            eff_line = ax.plot(
                round_t_grid,
                effective_sum,
                color='black',
                linestyle='--',
                linewidth=2,
                label='effective total',
            )
            handles += eff_line
            labels += ['effective total']

    ax.set_xlabel('Normalized Round Time')
    ax.set_ylabel(metric)
    run_suffix = f' [{run_label}]' if run_label else ''
    ax.set_title(f'Round-averaged profile by device: {metric}{phase_suffix}{run_suffix}')
    ax.grid(True)
    ax.legend(handles, labels, ncol=2, fontsize=8)

    if round_counts:
        counts_text = ', '.join(f'{dev}:{cnt}' for dev, cnt in sorted(round_counts.items()))
        summary_text = f'devices={plotted} | rounds per device: {counts_text}'
        ax.text(
            0.01,
            0.01,
            summary_text,
            transform=ax.transAxes,
            fontsize=8,
            va='bottom',
            ha='left',
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none'),
        )

    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_ul_dl_round_average_across_devices(
    ue_dfs,
    ul_metric,
    dl_metric,
    n_points=100,
    phase_filter=None,
    layout='same_axes',
    run_label=None,
    savepath=None,
    show=True,
):
    phase_list = _normalize_phase_filter(phase_filter)
    phase_suffix = '' if phase_list is None else f" ({'/'.join(phase_list)})"
    layout = (layout or 'same_axes').lower()

    # Collect profiles once for UL/DL.
    ul_profiles = {}
    dl_profiles = {}
    for device_id, ue_df in ue_dfs.items():
        if 'round_id' not in ue_df.columns:
            continue
        ul_profile = compute_round_average_profile(
            ue_df,
            ul_metric,
            n_points=n_points,
            phase_filter=phase_filter,
            round_ids=None,
        )
        dl_profile = compute_round_average_profile(
            ue_df,
            dl_metric,
            n_points=n_points,
            phase_filter=phase_filter,
            round_ids=None,
        )
        if not ul_profile.empty:
            ul_profiles[str(device_id)] = ul_profile
        if not dl_profile.empty:
            dl_profiles[str(device_id)] = dl_profile

    if not ul_profiles and not dl_profiles:
        return

    run_suffix = f' [{run_label}]' if run_label else ''

    if layout == 'subplots':
        fig, (ax_ul, ax_dl) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for device_id, prof in ul_profiles.items():
            ax_ul.plot(prof['round_t'], prof['mean'], linewidth=2, label=device_id)
        for device_id, prof in dl_profiles.items():
            ax_dl.plot(prof['round_t'], prof['mean'], linewidth=2, label=device_id)

        ax_ul.set_ylabel(ul_metric)
        ax_ul.set_title(f'UL round-averaged profile by device{phase_suffix}{run_suffix}')
        ax_ul.grid(True)
        ax_ul.legend(ncol=2, fontsize=8)

        ax_dl.set_xlabel('Normalized Round Time')
        ax_dl.set_ylabel(dl_metric)
        ax_dl.set_title(f'DL round-averaged profile by device{phase_suffix}{run_suffix}')
        ax_dl.grid(True)
        ax_dl.legend(ncol=2, fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(11, 5))
        colors = sns.color_palette('tab10', n_colors=max(len(ul_profiles), len(dl_profiles), 1))
        device_order = sorted(set(list(ul_profiles.keys()) + list(dl_profiles.keys())))
        color_map = {dev: colors[i % len(colors)] for i, dev in enumerate(device_order)}

        for device_id, prof in ul_profiles.items():
            ax.plot(
                prof['round_t'],
                prof['mean'],
                linewidth=2,
                linestyle='-',
                color=color_map[device_id],
                label=f'{device_id} UL',
            )
        for device_id, prof in dl_profiles.items():
            ax.plot(
                prof['round_t'],
                prof['mean'],
                linewidth=2,
                linestyle='--',
                color=color_map[device_id],
                label=f'{device_id} DL',
            )

        ax.set_xlabel('Normalized Round Time')
        ax.set_ylabel(f'{ul_metric} / {dl_metric}')
        ax.set_title(f'UL/DL round-averaged profile by device{phase_suffix}{run_suffix}')
        ax.grid(True)
        ax.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, format='svg', dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_round_examples_and_average(annotated_trial_data, metric, round_ids=None, n_points=100, phase_filter=None, device_label=None):
    df = annotated_trial_data.to_pandas()
    phase_list = _normalize_phase_filter(phase_filter)
    if phase_list is not None:
        df = df[df['phase'].isin(phase_list)]
    if round_ids is not None:
        df = df[df['round_id'].isin(round_ids)]

    if device_label is None:
        device_label = 'device'
    phase_suffix = '' if phase_list is None else f" ({'/'.join(phase_list)})"

    if not df.empty and metric in df.columns:
        plt.figure(figsize=(11, 5))
        for rid, grp in df.groupby('round_id'):
            grp = grp.sort_values('round_elapsed_s')
            plt.plot(grp['round_elapsed_s'], grp[metric], alpha=0.5, label=f'round {rid}')
        plt.xlabel('Round Time (s)')
        plt.ylabel(metric)
        plt.title(f'Per-round traces: {metric} [{device_label}]{phase_suffix}')
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)
        plt.show()

    profile = compute_round_average_profile(
        annotated_trial_data,
        metric,
        n_points=n_points,
        phase_filter=phase_filter,
        round_ids=round_ids,
    )
    if not profile.empty:
        plt.figure(figsize=(11, 5))
        plt.plot(profile['round_t'], profile['mean'], color='black', label='mean')
        plt.fill_between(
            profile['round_t'],
            profile['mean'] - profile['std'],
            profile['mean'] + profile['std'],
            alpha=0.2,
            color='gray',
            label='mean ± std',
        )
        plt.xlabel('Normalized Round Time')
        plt.ylabel(metric)
        plt.title(f'Round-averaged profile: {metric} [{device_label}]{phase_suffix}')
        plt.grid(True)
        plt.legend()
        plt.show()


def filter_out_inactivity(agg_metrics_file, trial_data, max_gap_s=200, return_rounds=False, annotate_phases=False):
    trial_data = trial_data.sort('timestamp')
    rounds = build_round_windows(agg_metrics_file, max_gap_s=max_gap_s)
    total_rows = len(pd.read_csv(agg_metrics_file))
    # print(f'Detected {total_rows-len(rounds)} period(s) of downtime')

    if rounds.empty:
        if return_rounds:
            return trial_data, rounds
        return trial_data

    intervals = pl.from_pandas(rounds[['round_start', 'round_end']])
    ts_dtype = trial_data.schema.get('timestamp')
    if ts_dtype is not None:
        intervals = intervals.with_columns(
            pl.col('round_start').cast(ts_dtype),
            pl.col('round_end').cast(ts_dtype),
        )
    intervals = intervals.set_sorted('round_start')

    original_size = len(trial_data)
    filtered = trial_data.join_asof(
        intervals,
        left_on='timestamp',
        right_on='round_start',
        strategy='backward',
        check_sortedness=False,
    )
    filtered = filtered.filter(
        pl.col('round_start').is_not_null() &
        (pl.col('timestamp') <= pl.col('round_end'))
    ).drop('round_start', 'round_end')

    # print(f'Filtered out {original_size-len(filtered)} data points collected during downtime')

    if annotate_phases:
        filtered = annotate_telemetry_with_rounds_and_phases(filtered, rounds)

    if return_rounds:
        return filtered, rounds
    return filtered


def _apply_round_filter_to_ue_dfs(exp_path, ue_dfs, max_gap_s=200, annotate_phases=False, cache_dir=None, use_cache=True):
    agg_metrics_file = Path(exp_path) / 'train_agg_metrics.csv'
    if not agg_metrics_file.exists():
        return ue_dfs, pd.DataFrame()

    filtered_ue_dfs = {}
    rounds = pd.DataFrame()
    exp_name = Path(exp_path).name

    for idx, (rnti, ue_df) in enumerate(ue_dfs.items()):
        round_fp = None
        ue_fp = None
        if cache_dir is not None:
            round_fp, ue_fp = _cache_paths(cache_dir, exp_name, rnti)

        if use_cache and ue_fp is not None and ue_fp.exists():
            cached_df = pl.read_csv(str(ue_fp), try_parse_dates=True)
            if 'round_id' in cached_df.columns:
                cached_df = cached_df.with_columns(pl.col('round_id').cast(pl.Int64))
            filtered_ue_dfs[rnti] = cached_df
            if rounds.empty and round_fp is not None and round_fp.exists():
                rounds = pd.read_csv(round_fp)
                for dt_col in [
                    'timestamp', 'round_start', 'round_end',
                    'downlink_start', 'downlink_end',
                    'training_start', 'training_end',
                    'evaluation_start', 'evaluation_end',
                    'uplink_start', 'uplink_end',
                    'idle_start', 'idle_end',
                ]:
                    if dt_col in rounds.columns:
                        rounds[dt_col] = pd.to_datetime(rounds[dt_col], utc=True, errors='coerce')
            continue

        if idx == 0:
            filtered_df, rounds = filter_out_inactivity(
                str(agg_metrics_file),
                ue_df,
                max_gap_s=max_gap_s,
                return_rounds=True,
                annotate_phases=annotate_phases,
            )
        else:
            filtered_df = filter_out_inactivity(
                str(agg_metrics_file),
                ue_df,
                max_gap_s=max_gap_s,
                return_rounds=False,
                annotate_phases=annotate_phases,
            )

        if ue_fp is not None:
            filtered_df.write_csv(str(ue_fp))
        if round_fp is not None and not round_fp.exists() and not rounds.empty:
            rounds.to_csv(round_fp, index=False)

        filtered_ue_dfs[rnti] = filtered_df

    return filtered_ue_dfs, rounds



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

def plot_rntis_by_time(ue_dfs, metric, metric_units, run_id, pts_to_plot, pts_offset=0, savepath=None, show=True, non_zero_metrics=None, min_thresholds=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    rows = []
    pts_offset = max(0, int(pts_offset))

    for rnti, ue_df in ue_dfs.items():
        values = _metric_values(ue_df, metric, non_zero_metrics, min_thresholds=min_thresholds)
        if not values:
            continue

        series = ue_df.select(['timestamp', metric]).to_pandas()
        if metric in _normalize_non_zero_metrics(non_zero_metrics):
            series[metric] = pd.to_numeric(series[metric], errors='coerce')
            series = series[(series[metric] > 0) & (~np.isclose(series[metric], 0.0, atol=1e-12))]
        metric_threshold = _normalize_min_thresholds(min_thresholds).get(metric)
        if metric_threshold is not None:
            series[metric] = pd.to_numeric(series[metric], errors='coerce')
            series = series[series[metric] >= metric_threshold]

        series = series.dropna(subset=[metric]).iloc[pts_offset: pts_offset + pts_to_plot]
        if series.empty:
            continue

        for _, row in series.iterrows():
            rows.append({'timestamp': row['timestamp'], 'value': row[metric], 'device': str(rnti)})

    if not rows:
        plt.close(fig)
        return

    plot_df = pd.DataFrame(rows)
    sns.scatterplot(
        data=plot_df,
        x='timestamp',
        y='value',
        hue='device',
        # jitter=False,
        # dodge=False,
        alpha=0.8,
        size=3,
        ax=ax,
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(metric + metric_units)
    ax.set_title(f'{metric} over time for all UEs, trial {run_id}')
    ax.legend()
    ax.grid(True)

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

    ue_df = ue_df.with_columns(
        # Some exports store counters as strings (and occasionally include commas).
        ul_bytes_num=pl.col('ulBytes').cast(pl.Utf8).str.replace_all(',', '').cast(pl.Float64, strict=False),
        dl_bytes_num=pl.col('dlBytes').cast(pl.Utf8).str.replace_all(',', '').cast(pl.Float64, strict=False),
    ).sort('timestamp').with_columns(
        delta_t_s=pl.col('timestamp').diff().dt.total_seconds(),
        ul_bytes_delta=pl.col('ul_bytes_num').diff(),
        dl_bytes_delta=pl.col('dl_bytes_num').diff(),
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
    ).drop('ul_bytes_num', 'dl_bytes_num')

    return ue_df
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


def _extract_run_id(exp):
    if exp.get('run_id') is not None:
        return str(exp.get('run_id'))

    name = exp['path'].name
    match = re.search(r'(^|_)(\d{6,})(_|$)', name)
    if match:
        return match.group(2)
    return None


def _format_run_label(exp, sweep_param=None):
    if sweep_param:
        base = format_sweep_label(sweep_param, exp[sweep_param], exp)
    else:
        base = exp['path'].name

    run_id = _extract_run_id(exp)
    nodes = exp.get('nodes')
    suffix_parts = []

    if nodes is not None and str(nodes) not in str(base):
        suffix_parts.append(str(nodes))

    if run_id is not None and run_id not in str(base):
        suffix_parts.append(f'RunID:{run_id}')

    if suffix_parts:
        return f"{base}, {', '.join(suffix_parts)}"
    return str(base)


def _compute_metric_mean_std(ue_dfs, metric, non_zero_metrics=None, min_thresholds=None):
    values = []
    for _, ue_df in ue_dfs.items():
        values.extend(_metric_values(
            ue_df,
            metric,
            non_zero_metrics=non_zero_metrics,
            min_thresholds=min_thresholds,
        ))

    if not values:
        return None, None

    series = pd.Series(values)
    mean = series.mean()
    std = series.std() if len(series) > 1 else None
    return mean, std

def _run_phys_layer_plotting(experiments, metrics, **kwargs):
    defaults = dict(
        plot_mode='distribution',
        distribution_plot_type='violin',
        pair_ul_dl=False,
        non_zero_metrics=None,
        min_thresholds=None,
        pts_to_plot=1000,
        pts_offset=5000,
        save_dir=None,
        filter_rounds_in_memory=False,
        annotate_round_phases=False,
        round_gap_s=200,
        round_ids_to_plot=None,
        round_phase_to_plot=None,
        round_profile_device=None,
        round_profile_all_devices=False,
        round_profile_points=100,
        round_profile_error_bars=False,
        round_profile_errorbar_step=10,
        round_profile_ul_dl_combined=False,
        round_profile_ul_dl_layout='same_axes',
        round_profile_include_effective=False,
        round_profile_effective_secondary_axis=True,
        save_round_profiles=False,
        round_filter_cache_dir=None,
        use_round_filter_cache=True,
    )

    params = {**defaults, **kwargs}

    non_zero_metrics = _normalize_non_zero_metrics(params['non_zero_metrics'])
    min_thresholds = _normalize_min_thresholds(params['min_thresholds'])
    avgs = {metric: {} for metric in metrics}
    stds = {metric: {} for metric in metrics}
    throughput_summary = []

    if params['save_dir'] is None:
        print(f'Plotting phys layer data for {len(experiments)} experiment(s)')
    else:
        print(f'Generating saved phys-layer plots for {len(experiments)} experiment(s)')

    for exp in experiments:
        path = exp['path']
        combine_rntis(path)
        ue_dfs = gather_metrics_by_rnti(path, metrics)
        if not ue_dfs:
            continue

        rounds = pd.DataFrame()
        if params['filter_rounds_in_memory']:
            ue_dfs, rounds = _apply_round_filter_to_ue_dfs(
                path,
                ue_dfs,
                max_gap_s=params['round_gap_s'],
                annotate_phases=params['annotate_round_phases'],
                cache_dir=params['round_filter_cache_dir'],
                use_cache=params['use_round_filter_cache'],
            )

        run_label = exp.get('run_label', path.name)
        processed_metrics = set()
        plotted_metrics = []
        skipped_metrics = []

        for metric in metrics:
            if metric in processed_metrics:
                continue

            paired_metric = None
            if params['pair_ul_dl'] and metric.startswith('ul_'):
                candidate = 'dl_' + metric[3:]
                if candidate in metrics:
                    paired_metric = candidate
                    processed_metrics.add(candidate)

            metric_mean, metric_std = _compute_metric_mean_std(
                ue_dfs,
                metric,
                non_zero_metrics=non_zero_metrics,
                min_thresholds=min_thresholds,
            )
            avgs[metric][run_label] = metric_mean
            stds[metric][run_label] = metric_std

            pair_mean = None
            pair_std = None
            if paired_metric is not None:
                pair_mean, pair_std = _compute_metric_mean_std(
                    ue_dfs,
                    paired_metric,
                    non_zero_metrics=non_zero_metrics,
                    min_thresholds=min_thresholds,
                )
                avgs[paired_metric][run_label] = pair_mean
                stds[paired_metric][run_label] = pair_std

            if paired_metric is not None:
                if metric_mean is None and pair_mean is None:
                    skipped_metrics.append(f'{metric} vs {paired_metric} (no values after filtering)')
                    processed_metrics.add(metric)
                    continue
            else:
                if metric_mean is None:
                    skipped_metrics.append(f'{metric} (no values after filtering)')
                    processed_metrics.add(metric)
                    continue

            save_file = None
            if params['save_dir'] is not None:
                if params['plot_mode'] == 'time':
                    save_file = params['save_dir'] / f'{metric}_by_time.svg'
                elif paired_metric:
                    save_file = params['save_dir'] / f'{metric}_vs_{paired_metric}_{params['distribution_plot_type']}.svg'
                else:
                    save_file = params['save_dir'] / f'{metric}_{params['distribution_plot_type']}.svg'

            if params['plot_mode'] == 'time':
                plot_rntis_by_time(
                    ue_dfs,
                    metric,
                    metric_units='',
                    run_id=run_label,
                    pts_to_plot=params['pts_to_plot'],
                    pts_offset=params['pts_offset'],
                    savepath=save_file,
                    show=(params['save_dir'] is None),
                    non_zero_metrics=non_zero_metrics,
                    min_thresholds=min_thresholds,
                )
            else:
                plot_rntis_distribution(
                    ue_dfs,
                    metric,
                    metric_units='',
                    run_id=run_label,
                    plot_type=params['distribution_plot_type'],
                    paired_metric=paired_metric,
                    split_violin=True,
                    savepath=save_file,
                    show=(params['save_dir'] is None),
                    non_zero_metrics=non_zero_metrics,
                    min_thresholds=min_thresholds,
                )

            if paired_metric is not None:
                plotted_metrics.append(f'{metric} vs {paired_metric}')
            else:
                plotted_metrics.append(metric)

            processed_metrics.add(metric)

            if params['round_ids_to_plot'] is not None:
                phase_list = _normalize_phase_filter(params['round_phase_to_plot'])
                phase_targets = phase_list if phase_list is not None else [None]

                if params['round_profile_all_devices']:
                    for phase_target in phase_targets:
                        if (
                            params['round_profile_ul_dl_combined'] and
                            paired_metric is not None and
                            metric in THROUGHPUT_METRICS and
                            paired_metric in THROUGHPUT_METRICS
                        ):
                            combined_savepath = None
                            if params['save_round_profiles']:
                                phase_name = 'all' if phase_target is None else str(phase_target)
                                safe_phase = re.sub(r'[^a-zA-Z0-9_.-]', '_', phase_name)
                                if params['save_dir'] is not None:
                                    round_profile_dir = params['save_dir'] / 'round_profiles'
                                else:
                                    round_profile_dir = Path.cwd() / 'round_profiles'
                                round_profile_dir.mkdir(parents=True, exist_ok=True)
                                safe_layout = re.sub(r'[^a-zA-Z0-9_.-]', '_', str(params['round_profile_ul_dl_layout']))
                                combined_savepath = round_profile_dir / f'{run_label}_{metric}_vs_{paired_metric}_{safe_layout}_{safe_phase}.svg'

                            plot_ul_dl_round_average_across_devices(
                                ue_dfs,
                                metric,
                                paired_metric,
                                n_points=params['round_profile_points'],
                                phase_filter=phase_target,
                                layout=params['round_profile_ul_dl_layout'],
                                run_label=run_label,
                                savepath=combined_savepath,
                                show=(params['save_dir'] is None),
                            )
                            continue

                        overlay_savepath = None
                        if params['save_round_profiles']:
                            phase_name = 'all' if phase_target is None else str(phase_target)
                            safe_phase = re.sub(r'[^a-zA-Z0-9_.-]', '_', phase_name)
                            if params['save_dir'] is not None:
                                round_profile_dir = params['save_dir'] / 'round_profiles'
                            else:
                                round_profile_dir = Path.cwd() / 'round_profiles'
                            round_profile_dir.mkdir(parents=True, exist_ok=True)
                            overlay_savepath = round_profile_dir / f'{run_label}_{metric}_all_devices_{safe_phase}.svg'

                        plot_round_average_across_devices(
                            ue_dfs,
                            metric,
                            round_ids=None,
                            n_points=params['round_profile_points'],
                            phase_filter=phase_target,
                            show_error_bars=params['round_profile_error_bars'],
                            errorbar_step=params['round_profile_errorbar_step'],
                            include_effective_sum=params['round_profile_include_effective'],
                            effective_on_secondary_axis=params['round_profile_effective_secondary_axis'],
                            run_label=run_label,
                            savepath=overlay_savepath,
                            show=(params['save_dir'] is None),
                        )
                else:
                    target_device = params['round_profile_device']
                    if target_device is None:
                        target_device = next(iter(ue_dfs.keys()), None)
                    devices_to_plot = [target_device] if target_device is not None else []

                    for device_id in devices_to_plot:
                        if device_id in ue_dfs and 'round_id' in ue_dfs[device_id].columns:
                            for phase_target in phase_targets:
                                plot_round_examples_and_average(
                                    ue_dfs[device_id],
                                    metric,
                                    round_ids=params['round_ids_to_plot'],
                                    n_points=params['round_profile_points'],
                                    phase_filter=phase_target,
                                    device_label=str(device_id),
                                )

        throughput_summary.extend(summarize_throughput_by_device(
            ue_dfs,
            run_label,
            non_zero_metrics=non_zero_metrics,
            min_thresholds=min_thresholds,
        ))

        print(f'Run {run_label}: plotted metrics -> {plotted_metrics if plotted_metrics else "none"}')
        if skipped_metrics:
            print(f'Run {run_label}: skipped metrics -> {skipped_metrics}')

    if throughput_summary:
        print('\nThroughput by device (Mbps): mean and std')
        throughput_df = pd.DataFrame(throughput_summary)
        print(throughput_df.to_string(index=False))
        if params['save_dir'] is not None:
            throughput_fp = params['save_dir'] / 'throughput_summary_by_device.csv'
            throughput_df.to_csv(throughput_fp, index=False)
            print(f'Saved throughput summary to {throughput_fp}')

    for metric in avgs:
        print(f'trial, \t\t\t mean \t\t\t std \t\t\t (metric: {metric})')
        for trial in avgs[metric]:
            print(f'{trial} \t {avgs[metric][trial]} \t {stds[metric][trial]}')

        

def main():
    data_dir = Path('/Users/kmcomer/Documents/5G Experiment Data/FedAvg/')
    output_dir = Path.cwd() / 'phys_layer_plots'

    # does NOT include all available metrics = ['ulMcs', 'dlMcs', 'rssi', 'rsrp', 'rsrq', 'dlBler', 'ulQm', 'dlQm', 'ulBler', 'phr', 'pcmax', 'sinr', 'pucchSnr', 'cqi', 'puschSnr']
    filters = { # one ue_[rnti].csv per device
        'bandwidth': '40 MHz',
        'rank': '2x2',
        'distribution': 'dirichlet',
        'congestion': False,
        'tdd': '7-2',
        'nodes': '6N'
    }
    sweep = 'network'
    metrics=['rssi', 'ul_throughput_mbps', 'dl_throughput_mbps']

    # filters = { # many ue_[rnti].csv per device
    #     'bandwidth': '80 MHz',
    #     'rank': '2x2',
    #     'distribution': 'dirichlet',
    #     'congestion': False,
    #     'tdd': '5-4',
    #     'nodes': '6N'
    # }

    experiments = build_experiment_index(data_dir)
    experiments = filter_experiments(experiments, filters)
    experiments = sort_experiments_by_sweep(experiments, sweep)

    if not experiments:
        print('No experiments matched the given filters')
        return
    
    filter_parts = [f"{k}_{str(v).replace('/', '_').replace(' ', '_')}" for k, v in filters.items()]
    filter_dir = '_'.join(filter_parts)
    sweep_output_dir = output_dir / filter_dir / sweep
    sweep_output_dir.mkdir(exist_ok=True, parents=True)

    for exp in experiments:
        exp['run_label'] = _format_run_label(exp, sweep_param=sweep)

    _run_phys_layer_plotting(
        experiments,
        metrics,
        plot_mode='distribution',
        distribution_plot_type='violin',
        pair_ul_dl=False,
        non_zero_metrics=['ul_throughput_mbps', 'dl_throughput_mbps'],
        min_thresholds={'ul_throughput_mbps': 0.01, 'dl_throughput_mbps': 0.01},
        filter_rounds_in_memory=True,
        annotate_round_phases=True,
        round_ids_to_plot=[2, 3, 4],
        round_phase_to_plot=['all'],  # or 'all' / None
        round_profile_all_devices=True,
        round_profile_points=100,
        round_profile_ul_dl_combined=True,
        round_profile_ul_dl_layout='same_axes',  # use 'subplots' for stacked UL/DL panels
        round_profile_include_effective=True,
        round_profile_effective_secondary_axis=True,
        pts_to_plot=100,
        pts_offset=0,
        round_filter_cache_dir=Path.cwd(),
        save_path=Path.cwd() / 'phys_layer_plots'
    )


if __name__ == '__main__':
    main()