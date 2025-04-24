#!/bin/bash


DISTRIBUTION=${DISTRIBUTION:-"iid"}
CLIENT_ID=${CLIENT_ID:-1}
SERVER_IP=${SERVER_IP:-"localhost"}

echo "Starting Flower client with distribution=$DISTRIBUTION, client_id=$CLIENT_ID, server=$SERVER_IP"
flower-supernode --insecure --superlink="${SERVER_IP}:9092" --node-config="dataset='/app/node_datasets/cifar10_${DISTRIBUTION}' cid=${CLIENT_ID}"
