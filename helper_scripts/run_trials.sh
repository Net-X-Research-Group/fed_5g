#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <NUM OF NODES> <DISTRIBUTION> <DEPLOYMENT>"
    exit 1
fi

N=$1
DIST=$2
DEPLOYMENT=$3

ALLOWED_DIST=("iid" "dirichlet")
ALLOWED_DEPLOYMENTS=("local-deployment" "lan-deployment" "wlan-deployment" "wwan-deployment")

if [[ ! " ${ALLOWED_DIST[*]} " =~ " ${DIST} " ]]; then
    echo "Error: Distribution must be one of: ${ALLOWED_DIST[*]}"
    exit 1
fi

if [[ ! " ${ALLOWED_DEPLOYMENTS[*]} " =~ " ${DEPLOYMENT} " ]]; then
    echo "Error: Deployment must be one of: ${ALLOWED_DEPLOYMENTS[*]}"
    exit 1
fi

	./dataset_distributor.sh "${N}" "${DIST}"
	sleep 2
	flwr run "$HOME"/fed_5g/flwr-application "${DEPLOYMENT}" --stream
	sleep 2
	#kill -9 $(pgrep tshark)