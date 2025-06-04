import matplotlib.pyplot as plt

def setup_plotting():
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.size': 18,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.format': 'png',
        'savefig.bbox': 'tight'
    })