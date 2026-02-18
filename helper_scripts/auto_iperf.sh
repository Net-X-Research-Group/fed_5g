#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <num_nodes> <bandwidth> <tdd configuration>"
    exit 1
fi

NUM_CLIENTS=$1 # command line argument
BW=$2
TDD=$3

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(17 18 19 20 21 22)

for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    ssh "$LOGIN" sudo nmcli r wwan off
done

for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    ssh "$LOGIN" sudo nmcli r wwan on
    ssh "$LOGIN" sudo nmcli connection up OAI
    ssh "$LOGIN" sudo route add -net 172.31.0.0/24 wwan0
    wwan=ip addr | grep wwan0 | grep inet | grep 1.*/
    # need user to start downlink test and then continue (ex. hit enter)
    ssh "$LOGIN" iperf -s -i 1 -B $wwan -t 120 > iperf/$BW_$TDD_$LOGIN_DL.txt
    # print summary stats (final line of iperf/$BW_$TDD_$LOGIN_DL.txt)
    ssh "$LOGIN" iperf -t 120 -i 1 -fm -b 1000M -c 172.31.0.135 -B $wwan > iperf/$BW_$TDD_$LOGIN_UL.txt
    # print summary stats (final line of iperf/$BW_$TDD_$LOGIN_UL.txt)
    ssh "$LOGIN" sudo nmcli r wwan off
done

