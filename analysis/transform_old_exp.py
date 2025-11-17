from pathlib import Path
import pandas as pd
import json

experiment_path = Path("/Users/roberthayek/Documents/git_repos/fed_5g/IMC/17974351302993584288_6N_IID_wlan")

# Add headers to latency files
latency_files = experiment_path.glob("latency_*_CID*.csv")
for latency_file in latency_files:
    print(f"Processing {latency_file}")
    # Skip if headers already set properly
    df = pd.read_csv(latency_file)
    if list(df.columns) == ['downlink_latency', 'uplink_latency']:
        print("Headers already set properly, skipping.")
        continue
    # Read without headers
    df = pd.read_csv(latency_file, header=None)

    # Drop first column
    df.drop(columns=[0], inplace=True)
    # Rename columns
    df.rename(columns={1: 'downlink_latency', 2: 'uplink_latency'}, inplace=True)
    # Save back to CSV
    df.to_csv(latency_file, index=False)


# Create train_agg_metrics.csv
train_agg_metrics_file = experiment_path / "train_agg_metrics.csv"
agg_metrics = experiment_path / "agg_metrics.json"
estimated_round_time = experiment_path / "round_time.csv"
if not train_agg_metrics_file.exists():
    print('Creating train_agg_metrics.csv')

    final_df = pd.DataFrame(columns=['server_round', 'train_loss', 'train_time', 'eval_loss', 'eval_acc', 'eval_time', 'round_duration'])

    with open(agg_metrics, 'r') as f:
        data = json.load(f)

    round_times = pd.read_csv(estimated_round_time)
    round_times['avg'] = round_times.mean(axis=1)

    final_df['round_duration'] = round_times['avg'] # Add round_times['avg'] to final_df['round_duration']
    final_df['server_round'] = pd.Series(data.keys()) # Add json keys to final_df as server round

    for round_num, metrics in data.items():
        server_round = int(round_num) - 1
        final_df.at[server_round, 'train_loss'] = metrics.get('train_loss')
        final_df.at[server_round, 'train_time'] = metrics.get('training_time')
        final_df.at[server_round, 'eval_loss'] = metrics.get('val_loss')
        final_df.at[server_round, 'eval_acc'] = metrics.get('val_acc')
        final_df.at[server_round, 'eval_time'] = metrics.get('val_test_time')

    final_df.to_csv(train_agg_metrics_file, index=False)













