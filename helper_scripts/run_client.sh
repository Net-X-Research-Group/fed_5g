#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <distribution> <cid> <server_ip>"
    exit 1
fi

DISTRIBUTION=$1
CID=$2
SERVER_IP=$3

source "${HOME}"/fl_venv/bin/activate

python "${HOME}"/fed_5g/util/ue_phy_metrics.py &
PYTHON_PID=$!
echo "Python process started with PID: $PYTHON_PID"
trap "kill $PYTHON_PID 2>/dev/null" EXIT INT TERM

flower-supernode --insecure --superlink="${SERVER_IP}:9092" --node-config="dataset='~/node_datasets/cifar10_${DISTRIBUTION}' cid=${CID}"
