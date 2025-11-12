import seaborn as sns
import matplotlib.pyplot as plt

def sort_experiments_by_sweep(experiments, sweep_param):
    """Sort experiments by sweep parameter value"""
    if sweep_param == 'bandwidth':
        return sorted(experiments, key=lambda x: int(x['bandwidth'].replace(' MHz', '')))
    elif sweep_param == 'tdd':
        return sorted(experiments, key=lambda x: tuple(map(int, x['tdd'].split('-'))))
    else:
        return sorted(experiments, key=lambda x: x[sweep_param])


def format_sweep_label(sweep_param, sweep_value):
    network_map = {
        'wwan': '5G',
        'wlan': 'Wifi',
        'lan': 'Ethernet'
    }

    """Generate display label for sweep parameter value"""
    if sweep_param == 'network':
        return network_map.get(sweep_value, sweep_value)
    elif sweep_param == 'tdd':
        return sweep_value.replace('-', ':')
    elif sweep_param == 'bandwidth':
        return sweep_value
    else:
        return f"{sweep_param.title()} {sweep_value}"


def save_and_close_figure(fig, output_dir, metric, sweep_param, filter_str, suffix=""):
    """Save figure with consistent naming and close it"""
    if filter_str == "":
        filename = f"{metric}{suffix}_{sweep_param}_sweep.png"
    else:
        filename = f"{metric}{suffix}_{filter_str}.png"
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def get_sweep_colors_markers(sweep_values):
    """Get consistent colors and markers for sweep configurations"""
    n_sweeps = len(sweep_values)
    colors = sns.color_palette("Set1", n_colors=n_sweeps)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h'][:n_sweeps]
    return colors, markers