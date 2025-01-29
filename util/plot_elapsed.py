import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
def main():
    with open('elapsed.json', 'r') as f:
        data = json.load(f)
    elapsed = list(data.values())
    times = []
    for item in elapsed:
        hms = [3600, 60, 1]
        seconds = sum([a * b for a, b in zip(hms, map(int, item.split(':')))])
        times.append(seconds)
    times_df = pd.DataFrame(times, columns=['time'])
    plt.figure()
    plt.scatter(times_df.index, times_df.time)
    mean = np.nanmean(times_df.time)
    plt.axhline(y=mean, linestyle='--', label=f'Mean: {mean:.1f}s')
    plt.xlabel('Trial')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylabel('Seconds')
    plt.title('Time (s) vs. Trial No.')
    plt.tight_layout()
    plt.legend()
    plt.savefig('run_durations.png')
    plt.show()
    

if __name__ == '__main__':
    main()
