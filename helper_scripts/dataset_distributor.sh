#!/bin/bash

PYTHON=python3                                                      # TODO update for specific machine
FED_5G_DIRECTORY="~/"

NUM_CLIENTS=$1 # command line argument

STRATEGY="iid" # iid, shard, dirichlet
DATASET="cifar10" # mnist, cifar10, cifar100

PARTITIONER_SCRIPT="$FED_5G_DIRECTORY/fed_5g/util/dataset_partitioner.py"

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(17 18 19 20 21 22)

# Activate virtual environment
source $FED_5G_DIRECTORY/fed_5g/venv/bin/activate

# Generate datasets
$PYTHON $PARTITIONER_SCRIPT -d $DATASET -n $NUM_CLIENTS -p $STRATEGY

# Distribute
for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    FOLDER="~/${NUM_CLIENTS}node_datasets/"
    ssh $LOGIN mkdir -p $FOLDER                                     # TODO add SSH public key to .ssh/authorized_keys on each Pi
    echo "Copying part ${CID} to $DEVICE_NAME_PREFIX${CID}"
    scp -r ~/datasets/$STRATEGY/${DATASET}_${STRATEGY}_part_${CID}_test/ $LOGIN:$FOLDER
done
