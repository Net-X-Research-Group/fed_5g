#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <num_nodes> <strategy>"
    exit 1
fi


PYTHON=python
FED_5G_DIRECTORY="$HOME"

NUM_CLIENTS=$1 # command line argument

STRATEGY=$2 # iid, shard, dirichlet
DATASET="cifar10" # mnist, cifar10, cifar100

PARTITIONER_SCRIPT="$FED_5G_DIRECTORY/fed_5g/util/dataset_partitioner.py"

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(17 18 19 20 21 22)

# Activate virtual environment
source "$FED_5G_DIRECTORY"/fed_5g/venv/bin/activate

# Generate datasets
$PYTHON "$PARTITIONER_SCRIPT" -d $DATASET -n "$NUM_CLIENTS" -p $STRATEGY

# Distribute
for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    FOLDER="~/node_datasets/"
    ssh "$LOGIN" mkdir -p "$FOLDER"
    echo "Copying part ${CID} to $DEVICE_NAME_PREFIX${CID}"
    scp -r ~/datasets/"${STRATEGY}"/${DATASET}_"${STRATEGY}"_part_${CID}/ "$LOGIN":"$FOLDER"
done
