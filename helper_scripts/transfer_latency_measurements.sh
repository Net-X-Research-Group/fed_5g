#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <num_nodes>"
    exit 1
fi

NUM_CLIENTS=$1

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(17 18 19 20 21 22)

for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    FILE_NAME=$(ssh "${LOGIN}" 'ls latency_*.csv')
    RUN_ID=$(echo "${FILE_NAME}" | sed -E 's/latency_(.*)\.csv/\1/')
    DIR_PATH="${HOME}"/"${RUN_ID}"
    NEW_FILE_NAME="${FILE_NAME%.csv}_CID${CID}.csv"
    scp "${LOGIN}:${FILE_NAME}" "${DIR_PATH}"/"${NEW_FILE_NAME}"
    ssh "${LOGIN}" rm latency_*
  done