import re
import pandas as pd
from datetime import datetime

def extract_training_losses(log_file_path):
    """
    Extract training loss values from a log file and export to CSV.
    
    Parameters:
    log_file_path (str): Path to the log file
    
    Returns:
    pandas.DataFrame: DataFrame containing timestamp and training loss values
    """
    # Initialize lists to store data
    timestamps = []
    training_losses = []
    validation_losses = []
    validation_accuracies = []
    
    # Regular expression pattern to match the loss information
    pattern = r"Training loss: ([\d.]+), Validation loss: ([\d.]+), Validation accuracy: ([\d.]+)"
    
    # Read the log file
    with open(log_file_path, 'r') as file:
        for line in file:
            if "Finished training" in line:
                # Extract timestamp
                timestamp_str = line.split(' - ')[0]
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                
                # Extract metrics using regex
                match = re.search(pattern, line)
                if match:
                    training_loss = float(match.group(1))
                    validation_loss = float(match.group(2))
                    validation_accuracy = float(match.group(3))
                    
                    # Append to lists
                    timestamps.append(timestamp)
                    training_losses.append(training_loss)
                    validation_losses.append(validation_loss)
                    validation_accuracies.append(validation_accuracy)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'training_loss': training_losses,
        'validation_loss': validation_losses,
        'validation_accuracy': validation_accuracies
    })
    
    return df

def main():
    # Replace with your log file path
    log_file_path = 'nohup.out'
    
    # Extract data
    df = extract_training_losses(log_file_path)
    
    # Export to CSV
    output_file = 'training_metrics.csv'
    df.to_csv(output_file, index=False)
    print(f"Data exported to {output_file}")
    print("\nData preview:")
    print(df)

if __name__ == "__main__":
    main()