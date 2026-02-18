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

# Create output directory for iperf results
mkdir -p iperf

for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    ssh "$LOGIN" sudo nmcli r wwan off
done

echo "device,rate(Mbps)" > "iperf/${BW}_${TDD}_UL.csv"
echo "device,rate(Mbps)" > "iperf/${BW}_${TDD}_DL.csv"

for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    echo
    echo $LOGIN
    ssh "$LOGIN" sudo nmcli r wwan on
    ssh "$LOGIN" sudo nmcli connection up OAI
    ssh "$LOGIN" sudo route add -net 172.31.0.0/24 wwan0
    wwan=$(ssh "$LOGIN" ip addr show wwan0 | grep "inet " | awk '{print $2}' | cut -d'/' -f1)
    echo ip address: ${wwan}
    
    read -p "Start downlink test on receiver. Set protocol to TCP, bandwidth to 100 Gbps, and modify destination IP. Press Enter once ready to begin..."
    ssh "$LOGIN" iperf -s -i 1 -fm -B "$wwan" -t 120 > "iperf/${BW}_${TDD}_pi0${CID}_DL.txt"
    DL_RATE=$(tail -1 "iperf/${BW}_${TDD}_pi0${CID}_DL.txt" | grep -oE '[0-9.]+\s+Mbits/sec' | awk '{print $1}')
    echo "DL Rate: ${DL_RATE} Mbits/sec"
    echo "commnetpi0${CID},${DL_RATE}" >> "iperf/${BW}_${TDD}_DL.csv"
    
    ssh "$LOGIN" iperf -t 120 -i 1 -fm -b 1000M -c 172.31.0.135 -B "$wwan" > "iperf/${BW}_${TDD}_pi0${CID}_UL.txt"
    UL_RATE=$(tail -1 "iperf/${BW}_${TDD}_pi0${CID}_UL.txt" | grep -oE '[0-9.]+\s+Mbits/sec' | awk '{print $1}')
    echo "UL Rate: ${UL_RATE} Mbits/sec"
    echo "commnetpi0${CID},${UL_RATE}" >> "iperf/${BW}_${TDD}_UL.csv"
    ssh "$LOGIN" sudo nmcli r wwan off
done

