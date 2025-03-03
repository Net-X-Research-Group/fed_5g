#!/bin/bash
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <distribution> <cid> <server_ip>"
    exit 1
fi

DISTRIBUTION=$1
CID=$2
SERVER_IP=$3

source "${HOME}"/fl_venv/bin/activate
flower-supernode --insecure --superlink="${SERVER_IP}:9092" --node-config="dataset='~/node_datasets/cifar10_${DISTRIBUTION}' cid=${CID}"
