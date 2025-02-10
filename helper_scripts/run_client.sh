#!/bin/bash
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <distribution> <cid> <num_nodes> <server_ip>"
    exit 1
fi

DISTRIBUTION=$1
CID=$2
NUM_NODES=$3
SERVER_IP=$4

source fl_venv/bin/activate
flower-supernode --insecure --superlink="${SERVER_IP}:9092" --node-config="dataset='~/${NUM_NODES}node_datasets/${DISTRIBUTION}' cid=${CID}"
