#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <num_nodes> <bandwidth> <tdd configuration>"
    exit 1
fi

NUM_CLIENTS=$1
BW=$2
TDD=$3

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(17 18 19 20 21 22)

TIME=30 #iperf test length (seconds)

mkdir -p iperf
mkdir -p "iperf/${BW}_${TDD}/"

# echo "disconnect all devices from 5G"
# for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
#     LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
#     ssh "$LOGIN" sudo nmcli connection down OAI
#     ssh "$LOGIN" sudo nmcli r wwan off
# done

echo "start,ns,device,interval(s),transfer(MB),bandwidth(Mbps),burst_lat_avg(ms),burst_lat_min(ms),burst_lat_max(ms),burst_lat_stdev(ms),cnt,size,inP,NetPwr,Reads=Dist,end,ns" > "iperf/${BW}_${TDD}/UL.csv"
echo "start,ns,device,interval(s),transfer(MB),bandwidth(Mbps),burst_lat_avg(ms),burst_lat_min(ms),burst_lat_max(ms),burst_lat_stdev(ms),cnt,size,inP,NetPwr,Reads=Dist,end,ns" > "iperf/${BW}_${TDD}/DL.csv"

# i could also automate oaibox setup and teardown through running oaibox offline...

for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
    echo
    echo $LOGIN

    # connect device to 5G
    ssh "$LOGIN" 'sudo nmcli r wwan on && sudo nmcli connection up OAI && sudo route add -net 172.31.0.0/24 wwan0'
    wwan=$(ssh "$LOGIN" ip addr show wwan0 | grep "inet " | awk '{print $2}' | cut -d'/' -f1)
    echo ${wwan}
    sleep 3

    # downlink test
    ssh "$LOGIN" "iperf -s -i 1 -fm -B ${wwan} -t ${TIME}" > "iperf/${BW}_${TDD}/pi0${CID}_DL.txt" &
    sleep 2
    start=$(gdate -u -Ins)
    echo start ${start}
    ssh user@10.105.46.208 docker exec oai-ext-dn iperf -t ${TIME} -i 1 -fm -B 172.31.0.135 -p 5001 -c ${wwan} --trip-times #--txdelay-time 1
    end=$(gdate -u -Ins)
    # scp "$LOGIN":"${BW}_${TDD}_pi0${CID}_DL.txt" "iperf/${BW}_${TDD}/pi0${CID}_DL.txt"
    # ssh  > "iperf/${BW}_${TDD}/pi0${CID}_DL.txt"
    echo end ${end}
    
    # Extract all values from the final summary row
    FINAL_LINE=$(tail -1 "iperf/${BW}_${TDD}/pi0${CID}_DL.txt")
    END_TIME=$(echo "$FINAL_LINE" | awk '{print $3}' | cut -d'-' -f2)
    TRANSFER=$(echo "$FINAL_LINE" | awk '{print $5, $6}' | sed 's/ //')
    BANDWIDTH=$(echo "$FINAL_LINE" | awk '{print $7}')
    BURST_LAT_AVG=$(echo "$FINAL_LINE" | awk '{print $9}' | cut -d'/' -f1)
    BURST_LAT_MIN=$(echo "$FINAL_LINE" | awk '{print $9}' | cut -d'/' -f2)
    BURST_LAT_MAX=$(echo "$FINAL_LINE" | awk '{print $9}' | cut -d'/' -f3)
    BURST_LAT_STDEV=$(echo "$FINAL_LINE" | awk '{print $9}' | cut -d'/' -f4)
    CNT_SIZE=$(echo "$FINAL_LINE" | sed -n 's/.*(\([^)]*\)).*/\1/p' | tr '/' ',')
    INP=$(echo "$FINAL_LINE" | awk '{print $12, $13}' | sed 's/ //')
    NETPWR=$(echo "$FINAL_LINE" | awk '{print $14}')
    READS_DIST=$(echo "$FINAL_LINE" | awk '{print $15}')
    
    echo "DL Rate: ${BANDWIDTH} Mbits/sec"
    echo "${start},pi0${CID},${END_TIME},${TRANSFER},${BANDWIDTH},${BURST_LAT_AVG},${BURST_LAT_MIN},${BURST_LAT_MAX},${BURST_LAT_STDEV},${CNT_SIZE},${INP},${NETPWR},${READS_DIST},${end}" >> "iperf/${BW}_${TDD}/DL.csv"
    sleep 3

    # uplink test
    ssh user@10.105.46.208 docker exec oai-ext-dn iperf -s -i 1 -B 172.31.0.135 -p 5001 -t ${TIME} -fm > "iperf/${BW}_${TDD}/pi0${CID}_UL.txt" &
    sleep 2
    start=$(gdate -u -Ins)
    echo start ${start}
    ssh "$LOGIN" iperf -t ${TIME} -i 1 -fm -c 172.31.0.135 -B "$wwan" --trip-times --txdelay-time 3
    end=$(gdate -u -Ins)
    echo end ${end}
    
    # Extract all values from the final summary row
    FINAL_LINE=$(tail -1 "iperf/${BW}_${TDD}/pi0${CID}_UL.txt")
    END_TIME=$(echo "$FINAL_LINE" | awk '{print $3}' | cut -d'-' -f2)
    TRANSFER=$(echo "$FINAL_LINE" | awk '{print $5, $6}' | sed 's/ //')
    BANDWIDTH=$(echo "$FINAL_LINE" | awk '{print $7}')
    BURST_LAT_AVG=$(echo "$FINAL_LINE" | awk '{print $9}' | cut -d'/' -f1)
    BURST_LAT_MIN=$(echo "$FINAL_LINE" | awk '{print $9}' | cut -d'/' -f2)
    BURST_LAT_MAX=$(echo "$FINAL_LINE" | awk '{print $9}' | cut -d'/' -f3)
    BURST_LAT_STDEV=$(echo "$FINAL_LINE" | awk '{print $9}' | cut -d'/' -f4)
    CNT_SIZE=$(echo "$FINAL_LINE" | sed -n 's/.*(\([^)]*\)).*/\1/p' | tr '/' ',')
    INP=$(echo "$FINAL_LINE" | awk '{print $12, $13}' | sed 's/ //')
    NETPWR=$(echo "$FINAL_LINE" | awk '{print $14}')
    READS_DIST=$(echo "$FINAL_LINE" | awk '{print $15}')
    
    echo "UL Rate: ${BANDWIDTH} Mbits/sec"
    echo "${start},pi0${CID},${END_TIME},${TRANSFER},${BANDWIDTH},${BURST_LAT_AVG},${BURST_LAT_MIN},${BURST_LAT_MAX},${BURST_LAT_STDEV},${CNT_SIZE},${INP},${NETPWR},${READS_DIST},${end}" >> "iperf/${BW}_${TDD}/UL.csv"
    sleep 3
    
    # disconnect device from 5G
    ssh "$LOGIN" 'sudo nmcli connection down OAI && sudo nmcli r wwan off'
    sleep 1
done

# allow 1 specific client in case of missing data
# put txt files in folder