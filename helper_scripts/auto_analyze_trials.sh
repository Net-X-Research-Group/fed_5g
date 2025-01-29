#!/bin/bash

source /home/rhayek/fed_5g/venv/bin/activate

# Check if the top-level directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <top-level-directory>"
    exit 1
fi

TOP_LEVEL_DIR="$1"

# Path to your Python script
ANALYZER_SCRIPT="/home/rhayek/fed_5g/util/wireshark_comms_analyzer.py"
COMMS_PLOTTER_SCRIPT="/home/rhayek/fed_5g/util/comms_metrics_agg.py"
ML_PLOTTER_SCRIPT="/home/rhayek/fed_5g/util/plotter.py"
TIME_EXTRACTOR="/home/rhayek/fed_5g/util/elapsed_extract.py"

# Loop through all subdirectories in the current directory/top level.
cd "$TOP_LEVEL_DIR" || { echo "Failed to change directory to $dir"; exit 1; }

for dir in */; do
    if [ -d "$dir" ]; then
        echo "Executing in directory: $dir"
        
        # Change to subdirectory
        cd "$dir" || { echo "Failed to change directory to $dir"; exit 1; }
        pwd
        # Run the Python script
        echo "Running wireshark analyzer..."
        python "$ANALYZER_SCRIPT" -p output.pcapng -c ../config.yml
        echo "Extracting elapsed times..."
        python "$TIME_EXTRACTOR"
        echo "Running comms metrics plotter..."
        python "$COMMS_PLOTTER_SCRIPT"
        echo "Running ML metrics plotter..."
        python "$ML_PLOTTER_SCRIPT" -f agg_metrics.json
        
        # Change back to parent directory
        cd ..
    fi
done
