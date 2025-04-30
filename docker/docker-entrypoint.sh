#!/bin/bash


DISTRIBUTION=${DISTRIBUTION:-"iid"}
CLIENT_ID=${CLIENT_ID:-1}
SERVER_IP=${SERVER_IP:-"localhost"}

echo "Starting UE_PHY Logging Fork"
python /app/util/ue_phy_metrics.py > /app/host_home/ue_log.txt 2>&1 &
PYTHON_PID=$!
echo "Python process started with PID: $PYTHON_PID"
trap "kill $PYTHON_PID 2>/dev/null" EXIT INT TERM

echo "Starting Flower client with distribution=$DISTRIBUTION, client_id=$CLIENT_ID, server=$SERVER_IP"
flower-supernode --insecure --superlink="${SERVER_IP}:9092" --node-config="dataset='/app/node_datasets/cifar10_${DISTRIBUTION}' cid=${CLIENT_ID}"
