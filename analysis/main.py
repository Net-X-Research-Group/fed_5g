import pandas as pd
from pathlib import Path
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats
import itertools
from data_loading import *
from helpers import *
import pingouin as pg
from matplotlib.patches import Patch
from scipy.stats import spearmanr

# Configure matplotlib for LaTeX fonts
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}'
})

column_name_map = {
    'train_loss': 'Training Loss',
    'train_time': 'Training Time (s)',
    'eval_loss': 'Evaluation Loss',
    'eval_acc': 'Evaluation Accuracy',
    'eval_time': 'Evaluation Time (s)',
    'round_duration': 'Round Duration (s)',
}

TIME_METRICS = ['train_time', 'eval_time', 'round_duration']


def plot_agg_metric(agg_metrics, output_dir, filter_str, sweep_param):
    """Create one figure per metric, with a line for each experiment"""
    sns.set_style("whitegrid")
    agg_metrics = sort_experiments_by_sweep(agg_metrics, sweep_param)

    metric_cols = ['eval_time', 'train_time', 'eval_acc', 'eval_loss', 'train_loss', 'round_duration']


    figs = {}
    for metric in metric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        display_name = column_name_map.get(metric, metric.replace('_', ' ').title())

        if metric in TIME_METRICS:
            # Frequency plot for time metrics
            for exp in agg_metrics:
                label = format_sweep_label(sweep_param, exp[sweep_param], exp)
                sns.histplot(exp['metrics'][metric], label=label, ax=ax)
            ax.set_xlabel(display_name, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
        else:
            # Line plot for other metrics
            for exp in agg_metrics:
                df = exp['metrics']
                label = format_sweep_label(sweep_param, exp[sweep_param], exp)
                sns.lineplot(x=df['server_round'], y=df[metric],
                             label=label, linewidth=2, ax=ax)
            ax.set_xlabel('Server Round', fontsize=12)
            ax.set_ylabel(display_name, fontsize=12)

        ax.legend()
        save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str)
        figs[metric] = fig

    return figs


def plot_agg_metric_vs_time(agg_metrics, output_dir, filter_str, sweep_param):
    sns.set_style("whitegrid")
    agg_metrics = sort_experiments_by_sweep(agg_metrics, sweep_param)

    metric_cols = [col for col in agg_metrics[0]['metrics'].columns
                   if col not in ['server_round', 'timestamp']]

    figs = {}
    for metric in metric_cols:
        if metric in TIME_METRICS:
            continue

        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_dist = fig.add_subplot(gs[1])

        display_name = column_name_map.get(metric, metric.replace('_', ' ').title())

        for exp in agg_metrics:
            df = exp['metrics'].copy()

            # Compute elapsed_time as cumulative sum of cleaned durations
            df['elapsed_time'] = df['round_duration'].cumsum()

            label = format_sweep_label(sweep_param, exp[sweep_param], exp)
            sns.lineplot(x=df['elapsed_time'], y=df[metric],
                         label=label, linewidth=2, ax=ax_main)
            print(f'Total elapsed time: {df["elapsed_time"].iloc[-1]} for {label}')

        # Distribution subplot: histogram of round durations
        for exp in agg_metrics:
            df = exp['metrics'].copy()

            label = format_sweep_label(sweep_param, exp[sweep_param], exp)
            sns.kdeplot(df['round_duration'].dropna(),
                        label=label, linewidth=2, ax=ax_dist)

        ax_main.set_xlabel('')
        ax_main.set_ylabel(display_name, fontsize=12)
        ax_main.legend(title='Bandwidth' if sweep_param == 'bandwidth' else sweep_param.title(), loc='best')
        ax_main.grid(True, alpha=0.3)

        ax_dist.set_xlabel('Round Duration (s)', fontsize=10)
        ax_dist.set_ylabel('Density', fontsize=10)
        ax_dist.tick_params(labelsize=9)
        ax_dist.grid(True, alpha=0.3, axis='y')
        #ax_dist.legend()

        save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str, suffix="_vs_time")
        figs[metric] = fig


    return figs

def plot_individual(metrics: list, output_dir: Path, filter_str: str, sweep_param: str):
    combined_df = prepare_individual_metrics(metrics, sweep_param)
    sweep_values = combined_df[sweep_param].unique()
    colors, markers = get_sweep_colors_markers(sweep_values)

    if sweep_param == 'bandwidth':
        sweep_values = sorted(sweep_values, key=lambda x: int(x.replace(' MHz', '')))
    elif sweep_param == 'tdd':
        sweep_values = sorted(sweep_values, key=lambda x: tuple(map(int, x.split('-'))))
    else:
        sweep_values = sorted(sweep_values)


    cids = sorted(combined_df['cid'].unique())


    # Boxplots for time metrics
    for metric in TIME_METRICS:
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, sweep_val in enumerate(sweep_values):
            subset = combined_df[combined_df[sweep_param] == sweep_val]
            data = [subset[subset['cid']==c][metric].dropna() for c in cids]
            pos = np.arange(len(cids)) + i * 0.25  # Space groups by 0.25 units
            label=format_sweep_label(sweep_param, sweep_val)
            ax.boxplot(data, positions=pos, widths=0.2, patch_artist=True,
                       flierprops=dict(marker=markers[i], markerfacecolor=colors[i],
                                       markeredgecolor=colors[i], markersize=6, alpha=0.6),
                        boxprops = dict(facecolor=colors[i], alpha=0.7),
                        medianprops = dict(color='black', linewidth=1.5))

        # Create manual legend
        legend_elements = [Patch(facecolor=colors[i], alpha=0.7,
                                label=format_sweep_label(sweep_param, val))
                          for i, val in enumerate(sweep_values)]
        ax.legend(handles=legend_elements)
        ax.set(xticks=np.arange(len(cids)), xticklabels=cids,
               xlabel='Client ID', ylabel=column_name_map[metric])
        save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str, suffix="_by_client")

        # KDE for round_duration
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, sweep_val in enumerate(sweep_values):
            subset = combined_df[(combined_df[sweep_param] == sweep_val) & (combined_df['round_duration'] > 0)][
                'round_duration']
            label=format_sweep_label(sweep_param, sweep_val)
            sns.kdeplot(data=subset, ax=ax, color=colors[i], label=label)
            ax.axvline(subset.median(), color=colors[i], linestyle='--', alpha=0.5)

        ax.set(xlabel=column_name_map['round_duration'], ylabel='Density')
        ax.legend()
        save_and_close_figure(fig, output_dir, 'round_duration', sweep_param, filter_str, suffix="_distribution")

def plot_latency_metrics(latency_metrics: pd.DataFrame, output_dir: Path, filter_str: str, sweep_param: str):
    """Plot latency metrics with CSV stats and figures for UL/DL latencies"""
    latency_metrics = latency_metrics.dropna()
    # Sort sweep values
    sweep_values = latency_metrics[sweep_param].unique()
    if sweep_param == 'bandwidth':
        sweep_values = sorted(sweep_values, key=lambda x: int(x.replace(' MHz', '')))
    elif sweep_param == 'tdd':
        sweep_values = sorted(sweep_values, key=lambda x: tuple(map(int, x.split('-'))))
    else:
        sweep_values = sorted(sweep_values)

    colors, markers = get_sweep_colors_markers(sweep_values)

    # Figure 1: Individual figures for each sweep value showing UL/DL for each CID
    for i, sweep_val in enumerate(sweep_values):
        subset = latency_metrics[latency_metrics[sweep_param] == sweep_val]
        cids = sorted(subset['cid'].unique())

        if len(subset) == 0:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Downlink latency by CID
        dl_data = [subset[subset['cid'] == cid]['downlink_latency'].values for cid in cids]
        #bp1 = ax1.boxplot(dl_data, tick_labels=cids, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
        sns.kdeplot(data=dl_data, ax=ax1, fill=True, alpha=0.5)
        #for patch in bp1['boxes']:
        #    patch.set_facecolor(colors[i])
        #    patch.set_alpha(0.7)
        ax1.set_xlabel('Downlink Latency (s)')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Downlink Latency - {subset["display_label"].iloc[0]}')
        ax1.grid(True, alpha=0.3)

        # Uplink latency by CID
        ul_data = [subset[subset['cid'] == cid]['uplink_latency'].values for cid in cids]
        #bp2 = ax2.boxplot(ul_data, tick_labels=cids, patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
        sns.kdeplot(data=ul_data, ax=ax2, fill=True, alpha=0.5)
        #for patch in bp2['boxes']:
        #    patch.set_facecolor(colors[i])
        #    patch.set_alpha(0.7)
        #ax2.set_xlabel('Client ID')
        ax2.set_xlabel(f'Uplink Latency (s)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Uplink Latency - {subset["display_label"].iloc[0]}')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        safe_sweep_val = str(sweep_val).replace('/', '_').replace(' ', '_')
        filename = f"latency_by_cid_{sweep_param}_{safe_sweep_val}.svg"
        fig.savefig(output_dir / filename, format='svg', bbox_inches='tight')
        plt.close(fig)

    # Figure 2: Box plots for all sweep values showing UL/DL distributions side by side
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for box plots
    all_data = []
    all_labels = []
    positions = []
    colors_list = []

    pos = 1
    for i, sweep_val in enumerate(sweep_values):
        subset = latency_metrics[latency_metrics[sweep_param] == sweep_val]
        if len(subset) > 0:
            # Add downlink data
            all_data.append(subset['downlink_latency'].values)
            all_labels.append(f"DL")
            positions.append(pos)
            colors_list.append(colors[i])

            # Add uplink data
            all_data.append(subset['uplink_latency'].values)
            all_labels.append('UL')
            positions.append(pos + 0.5)
            colors_list.append(colors[i])

            pos += 1.5  # Space between different sweep values

    # Create box plot
    bp = ax.boxplot(all_data, positions=positions,
                    patch_artist=True, widths=0.4,
                    medianprops=dict(color='black', linewidth=1.5),
                    showfliers=False)

    # Color the boxes
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_list[i])
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_ylabel('Latency (s)')
    ax.set_title(f'Latency Distribution by {sweep_param.title()}')
    ax.grid(True, alpha=0.3, axis='y')

    # Get the display_label for sweep_values for legend
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7,
                            label=latency_metrics[latency_metrics[sweep_param] == val]['display_label'].iloc[0])
                      for i, val in enumerate(sweep_values)]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()

    filename = f"latency_boxplot_{sweep_param}_sweep.svg"
    fig.savefig(output_dir / filename, format='svg', bbox_inches='tight')
    plt.close(fig)

def plot_cellular_sweep(experiment_paths: list, filters: dict, sweep: str, output_dir: Path):
    filtered = [exp for exp in experiment_paths
                if all(exp.get(k) is None or (exp.get(k) in v if isinstance(v, list) else exp.get(k) == v)
                    for k, v in filters.items())]

    filter_parts = [f"{k}_{str(v).replace('/', '_').replace(' ', '_')}" for k, v in filters.items()]
    filter_dir = "_".join(filter_parts)
    sweep_output_dir = output_dir / filter_dir / sweep
    sweep_output_dir.mkdir(exist_ok=True, parents=True)

    agg_metrics = []
    individual_metrics = []
    latency_metrics = []

    for exp in filtered:
        metrics = load(exp)
        common_data = {**exp, 'execution_time': metrics.get('execution_time', None),
                       'start_time': metrics.get('start_time', None), 'sweep': sweep}
        agg_metrics.append({**common_data, 'metrics': metrics['server_agg_metric']})
        individual_metrics.append({**common_data, 'metrics': metrics['individual_metrics']})
        latency_metrics.append({**common_data, 'metrics': metrics['latency']})

    # Filtering agg_metrics and create round_duration before plotting
    for exp in agg_metrics:
        if 'round_duration' not in exp['metrics'].columns:
            df = exp['metrics'].copy()
            if 'round_duration' not in df.columns:
                df['round_duration'] = df['timestamp'].diff()

            # Replace outliers with median for time calculation
            original_count = len(df['round_duration'].dropna())
            outliers_mask = (df['round_duration'] > 200) | (pd.isna(df['round_duration']))
            outliers_count = outliers_mask.sum()

            median_duration = df['round_duration'].median()
            df['round_duration'] = df['round_duration'].apply(
                lambda x: median_duration if (pd.isna(x) or x > 200) else x
            )

            # Record filtering operation
            if outliers_count.any():
                print(f'Percent removed {outliers_count} out of {original_count} outliers from round_duration, replaced with median={median_duration} experiment={format_sweep_label(sweep, exp[sweep], exp)}')
            exp['metrics'] = df


    # Combine all latency data and filter all outliers before plotting
    combined_latency = []
    for exp in latency_metrics:
        df = exp['metrics'].copy()
        display_label = format_sweep_label(sweep, exp[sweep], exp)
        df[sweep] = exp[sweep]
        df['display_label'] = display_label
        combined_latency.append(df)

    if not combined_latency:
        print("No latency data found")
        return

    latency_metrics = pd.concat(combined_latency, ignore_index=True)
    mask = (latency_metrics['downlink_latency'] > 100) & latency_metrics['downlink_latency'].notna()
    num_replaced = mask.sum()
    latency_metrics['downlink_latency'] = np.where(mask, np.nan, latency_metrics['downlink_latency'])
    print(f"Replaced {num_replaced} outliers in downlink_latency with NaN ({num_replaced/len(latency_metrics['downlink_latency'])})%")


    mask = (latency_metrics['uplink_latency'] > 100) & latency_metrics['uplink_latency'].notna()
    num_replaced = mask.sum()
    latency_metrics['uplink_latency'] = np.where(mask, np.nan, latency_metrics['uplink_latency'])
    print(f"Replaced {num_replaced} outliers in uplink_latency with NaN ({num_replaced/len(latency_metrics['uplink_latency'])})%")


    plot_agg_metric(agg_metrics, sweep_output_dir, "", sweep)
    plot_agg_metric_vs_time(agg_metrics, sweep_output_dir, "", sweep)
    plot_individual(individual_metrics, sweep_output_dir, "", sweep)
    plot_latency_metrics(latency_metrics, sweep_output_dir, "", sweep)

    tost_test(individual_metrics, sweep, sweep_output_dir)
    uplink_statistics(latency_metrics, individual_metrics, sweep_output_dir, sweep)


def uplink_statistics(latency_metrics, individual_metrics, output_dir, sweep_param):
    """Compute uplink latency statistics and determine correlation with round duration"""
    individual_metrics = prepare_individual_metrics(individual_metrics, sweep_param)
    merged = pd.merge(latency_metrics, individual_metrics,
                      on=['cid', 'server_round', sweep_param],
                      suffixes=('_latency', '_individual'))

    # Aggregate to round level
    round_agg = merged.groupby([sweep_param, 'server_round']).agg({
        'uplink_latency': 'max',    # Slowest uplink in round
        'downlink_latency': 'max',  # Slowest uplink in round
        'round_duration': 'first'  # Round duration (same for all clients)
    }).reset_index()

    # Correlation per node count
    for nodes, grp in round_agg.groupby(sweep_param):
        rho, p = spearmanr(grp['downlink_latency'], grp['round_duration'])
        print(f"\n{nodes}: ρ={rho:.3f}, p={p:.4f}, n={len(grp)}")

    # Does uplink latency significant effect round duration between sweeps?
    groups = [grp['round_duration'].values for _, grp in round_agg.groupby(sweep_param)]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 0)
    pd.set_option('display.max_colwidth', None)

    round_duration_by_config = merged.groupby(sweep_param)[['uplink_latency', 'downlink_latency', 'round_duration']].agg(['mean', 'std'])
    print(round_duration_by_config)

    # Percent latency of round duration
    merged['uplink_pct_round'] = 100 * merged['uplink_latency'] / merged['round_duration']
    merged['downlink_pct_round'] = 100 * merged['downlink_latency'] / merged['round_duration']
    pct_stats = merged.groupby(sweep_param)[['uplink_pct_round', 'downlink_pct_round']].agg(['mean', 'std'])
    print(pct_stats)




def sweep_anova(metrics, sweep_param, output_dir):
    """Perform ANOVA on individual metrics across different sweep values"""

    df = prepare_individual_metrics(metrics, sweep_param)

    anova_results = []

    for metric in TIME_METRICS:
        for cid in sorted(df['cid'].unique()):
            cid_data = df[df['cid'] == cid]

            groups = {name: group[metric].dropna().values
                      for name, group in cid_data.groupby(sweep_param)}

            if len(groups) < 2:
                continue

            group_items = list(groups.values())
            group_names = list(groups.keys())

            f_stat, p_val = stats.f_oneway(*group_items)

            anova_results.append({'cid': cid,
                                  'metric': metric,
                                  'sweep_param': sweep_param,
                                  'f_statistic': f_stat,
                                  'p_value': p_val,
                                  'num_groups': len(groups)})

            """# Plot each distribution for visual inspection
            plt.figure(figsize=(8, 4))
            for name, data in groups.items():
                sns.kdeplot(data, label=f"{sweep_param}={name}", fill=True)
            plt.title(f'Distribution of {metric} for CID {cid}\nANOVA p={p_val:.9f}')
            plt.xlabel(column_name_map.get(metric, metric))
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.show()"""


            # Pairwise t-tests if p < 0.005
            if p_val < 0.005:
                combs = list(set(itertools.combinations(group_names, 2)))
                for i, j in combs:
                    g1 = groups.get(i)
                    g2 = groups.get(j)

                    t_stat, pair_p_val = stats.ttest_ind(g1, g2)

                    """# Plot each distribution for visual inspection
                    plt.figure(figsize=(8, 4))
                    sns.kdeplot(g1, label=f"{sweep_param}={i}", fill=True)
                    sns.kdeplot(g2, label=f"{sweep_param}={j}", fill=True)
                    plt.title(f'Distribution of {metric} for CID {cid}\n{sweep_param}={i} vs {sweep_param}={j} | p={pair_p_val:.9f}')
                    plt.xlabel(column_name_map.get(metric, metric))
                    plt.ylabel('Density')
                    plt.legend()
                    plt.tight_layout()
                    plt.show()"""

                    print('Pairwise t-test:', cid, metric, i, j, pair_p_val)


    pd.DataFrame(anova_results).to_csv(
        output_dir / f'anova_{sweep_param}.csv', index=False)


    return anova_results

def tost_test(metrics, sweep_param, output_dir):
    df = prepare_individual_metrics(metrics, sweep_param)
    results = []
    for metric in TIME_METRICS:
        client_means = df.groupby('cid')[metric].mean()
        cv_between_clients = client_means.std() / client_means.mean()
        print(f'CV between clients for {metric}: {cv_between_clients}')
        print(f'Metric {metric} range across clients: {client_means.min()} - {client_means.max()}')

        for cid in df['cid'].unique():
            cid_data = df[df['cid'] == cid]
            cid_mean = cid_data[metric].mean()



            delta = 0.05 * cid_mean

            tdd_groups = {name: group[metric].dropna().values
                      for name, group in cid_data.groupby(sweep_param)}

            n_pairs = 0
            n_equiv = 0

            for tdd1, tdd2 in itertools.combinations(tdd_groups.keys(), 2):
                tost_result = pg.tost(
                    tdd_groups[tdd1], tdd_groups[tdd2], delta
                )
                p_val = tost_result['pval'].values[0]
                equiv = p_val < 0.05


                n_pairs += 1
                if equiv:
                    n_equiv += 1

            results.append({
                'metric': metric,
                'CID': cid,
                'mean_metric': cid_mean,
                'bound_5pct': delta,
                'equiv_pairs': n_equiv,
                'pct_equiv': 100 * n_equiv / n_pairs if n_pairs > 0 else 0,

            })

    equiv_df = pd.DataFrame(results)
    equiv_df.to_csv(
        output_dir / f'tost_equivalence_{sweep_param}.csv', index=False)

def grid_plot(experiments, output_dir):
    # Only include rank=2x2, network=wwan, nodes=6
    filters = {'rank': '2x2', 'network': 'wwan', 'nodes': '6N', 'congestion': False}
    filtered = [exp for exp in experiments if all(exp.get(key) == val for key, val in filters.items())]

    agg_metrics = []


    for exp in filtered:
        metrics = load(exp)
        common_data = {**exp, 'execution_time': metrics.get('execution_time', None),
                       'start_time': metrics.get('start_time', None)}
        agg_metrics.append({**common_data, 'metrics': metrics['server_agg_metric']})


    # Bandwidth across, TDD down, conbindation grid plot


    # Filtering agg_metrics and create round_duration before plotting
    for exp in agg_metrics:
        if 'round_duration' not in exp['metrics'].columns:
            df = exp['metrics'].copy()
            if 'round_duration' not in df.columns:
                df['round_duration'] = df['timestamp'].diff()

            # Replace outliers with median for time calculation
            original_count = len(df['round_duration'].dropna())
            outliers_mask = (df['round_duration'] > 200) | (pd.isna(df['round_duration']))
            outliers_count = outliers_mask.sum()

            median_duration = df['round_duration'].median()
            df['round_duration'] = df['round_duration'].apply(
                lambda x: median_duration if (pd.isna(x) or x > 200) else x
            )

            # Record filtering operation
            if outliers_count.any():
                print(
                    f'Percent removed {outliers_count} out of {original_count} outliers from round_duration, replaced with median={median_duration}')
            exp['metrics'] = df

    # ============ BUILD MAPPING ============ #
    data_map = {}
    for exp in agg_metrics:
        exp['tdd'] = exp['tdd'].replace('-', ':')
        key = (exp['tdd'], exp['bandwidth'])
        data_map[key] = exp

    keys = sorted(data_map.keys())
    n_plots = len(keys)

    n_cols = 5
    n_rows = int(np.ceil(n_plots / n_cols))

    print(f"Creating centered {n_rows}×{n_cols} grid for {n_plots} plots")

    # ============ CREATE GRID WITH GRIDSPEC ============ #
    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # ============ PLOT WITH CENTER ALIGNMENT ============ #
    idx = 0
    for (tdd, bw), exp in data_map.items():
        row = idx // n_cols
        col = idx % n_cols

        # Center align the last row if incomplete
        plots_in_row = min(n_cols, n_plots - row * n_cols)
        if plots_in_row < n_cols:
            offset = (n_cols - plots_in_row) / 2.0
            ax = plt.subplot2grid((n_rows, n_cols * 2), (row, int(col * 2 + offset * 2)), colspan=2)
        else:
            ax = plt.subplot2grid((n_rows, n_cols), (row, col))

        df = exp['metrics']
        sns.kdeplot(data=df['round_duration'].dropna(), ax=ax, fill=True, alpha=0.5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'{tdd} Split at {bw}', fontsize=15)
        ax.grid(True, alpha=0.3)

        idx += 1

    #fig.supxlabel('Round Duration (s)', fontsize=12)
    #fig.supylabel('Density', fontsize=12)
    #fig.suptitle('Round Duration Distribution by TDD × Bandwidth', fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/round_duration_grid.svg', format='svg', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    directory = Path("/Users/roberthayek/Documents/git_repos/fed_5g/IMC")
    output_dir = Path.cwd()
    print(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    all_experiments = []
    for exp in directory.iterdir():
        if not exp.is_dir():
            continue
        params = parse_experiment_name(exp.name)
        all_experiments.append({'path': exp, **params})

    """plot_cellular_sweep(all_experiments, {'tdd': '7-2', 'rank': '2x2',
                                          'distribution': 'dirichlet', 'congestion': False, 'network': 'wwan', 'nodes': '6N'},
                        sweep='bandwidth', output_dir=output_dir)"""


    # NETWORK COMPARISON: 7-2 TDD, 40 MHz, 2x2, iid, no congestion
    plot_cellular_sweep(all_experiments, {'bandwidth': '40 MHz', 'rank': '2x2',
                                          'distribution': 'dirichlet', 'congestion': False, 'tdd': '7-2', 'nodes': '6N'},
                        sweep='network', output_dir=output_dir)


    #grid_plot(all_experiments, output_dir)



