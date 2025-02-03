#!/bin/bash

PYTHON=python3
FED_5G_DIRECTORY="/Users/kmcomer"

PARTITIONER_SCRIPT="$FED_5G_DIRECTORY/fed_5g/util/dataset_partitioner.py"

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(18 19 20 21)

STRATEGY="iid" # iid, shard, dirichlet
DATASET="cifar10" # mnist, cifar10, cifar100

NUM_CLIENTS=3

# Activate virtual environment
source $FED_5G_DIRECTORY/fed_5g/venv/bin/activate

# Generate datasets
$PYTHON $PARTITIONER_SCRIPT -d $DATASET -n $NUM_CLIENTS -p $STRATEGY

# Distribute
for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    rsync -rP ~/datasets/$STRATEGY/{$DATASET}_{$STRATEGY}_part_{$CID}_test/ {$DEVICE_NAME_PREFIX}{$CID}@{$IP_PREFIX}{${IP_SUFFIXES[$CID-1]}}:~/{$NUM_CLIENTS}node_datasets/
done