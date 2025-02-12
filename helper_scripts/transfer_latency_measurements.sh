#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_nodes>"
    exit 1
fi

NUM_CLIENTS=$1

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(17 18 19 20 21 22)

for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    scp "${LOGIN}":latency.csv latency_cid_${CID}.csv
    ssh "${LOGIN}" rm latency.csv
done