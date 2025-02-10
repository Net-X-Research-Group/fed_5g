#!/bin/bash

source /home/rhayek/fed_5g/venv/bin/activate

# Check if the top-level directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <top-level-directory>"
    exit 1
fi

TOP_LEVEL_DIR="$1"

ANALYZER_SCRIPT="/home/rhayek/fed_5g/util/wireshark_comms_analyzer.py"
COMMS_PLOTTER_SCRIPT="/home/rhayek/fed_5g/util/comms_metrics_agg.py"
TIME_EXTRACTOR="/home/rhayek/fed_5g/util/elapsed_extract.py"
METRICS_ANALYZER="$HOME/fed_5g/util/metrics_analyzer.py"

# Loop through all subdirectories in the current directory/top level.
cd "$TOP_LEVEL_DIR" || { echo "Failed to change directory to $dir"; exit 1; }
echo "Extracting elapsed times..."
python "$TIME_EXTRACTOR"
for dir in */; do
    if [ -d "$dir" ]; then
        echo "Executing in directory: $dir"
        
        # Change to subdirectory
        cd "$dir" || { echo "Failed to change directory to $dir"; exit 1; }
        pwd
        # Run the Python script
        echo "Running wireshark analyzer..."
        python "$ANALYZER_SCRIPT" -p output.pcapng -c ../config.yml

        echo "Running comms metrics plotter..."
        python "$COMMS_PLOTTER_SCRIPT"

        echo "Running metrics_analyzer..."
        python "$METRICS_ANALYZER"

        # Change back to parent directory
        cd ..
    fi
done
