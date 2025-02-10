#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <distribution> <cid> <num_nodes>"
    exit 1
fi

DISTRIBUTION=$1
CID=$2
NUM_NODES=$3
source fl_venv/bin/activate
flower-supernode --insecure --superlink="129.105.6.252:9092" --node-config="dataset='~/${NUM_NODES}node_datasets/${DISTRIBUTION}' cid=${CID}"
