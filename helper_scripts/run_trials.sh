#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <NUM OF NODES>"
    exit 1
fi

N=$1

for i in $(seq 1 20);
do
	./dataset_distributor.sh ${N}
	cp cifar10_"${N}"_partitions_iid.png cifar10_"${N}"_partitions_iid_trial_$i.png
	sleep 2
	flwr run "$HOME"/fed_5g/flwr-application --stream
	sleep 2
	kill -9 $(pgrep tshark)
done
