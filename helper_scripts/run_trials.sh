#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <NUM OF NODES> <DISTRIBUTION> <DEPLOYMENT>"
    exit 1
fi

N=$1
DIST=$2

	./dataset_distributor.sh "${N}" "${DIST}"
	sleep 2
	flwr run "$HOME"/fed_5g/flwr-application "${DEPLOYMENT}" --stream
	sleep 2
	#kill -9 $(pgrep tshark)
  ./transfer_latency_measurements.sh "${N}"

flwr ls "$HOME"/fed_5g/flwr-application --format json > "${HOME}"/trials.json