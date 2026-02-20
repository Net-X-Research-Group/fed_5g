#!/bin/bash

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <num_nodes|specific_node> <bandwidth> <tdd configuration> [single_client_id]"
    echo "Examples:"
    echo "  $0 6 40 7-2              # Run for 6 clients"
    echo "  $0 6 40 7-2 3            # Run only for client 3"
    exit 1
fi

NUM_CLIENTS=$1
BW=$2
TDD=$3
SINGLE_CLIENT=${4:-0}  # Optional 4th argument, default to 0 (run all)

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(17 18 19 20 21 22)

TIME=30 #iperf test length (seconds)

mkdir -p iperf
mkdir -p "iperf/${BW}_${TDD}/"

# Check if CSV files exist and prompt user
if [ -f "iperf/${BW}_${TDD}/UL.csv" ] || [ -f "iperf/${BW}_${TDD}/DL.csv" ]; then
    echo "CSV files already exist for ${BW}_${TDD}"
    read -p "Do you want to (o)verwrite or (a)ppend? [o/a]: " choice
    case $choice in
        [Aa]* )
            APPEND_MODE=true
            echo "Data will be appended to existing CSV files"
            ;;
        [Oo]* )
            APPEND_MODE=false
            echo "Existing CSV files will be overwritten"
            echo "start,ns,device,interval(s),transfer(MB),bandwidth(Mbps),burst_lat_avg(ms),burst_lat_min(ms),burst_lat_max(ms),burst_lat_stdev(ms),cnt,size,inP,NetPwr,Reads=Dist,end,ns" > "iperf/${BW}_${TDD}/UL.csv"
            echo "start,ns,device,interval(s),transfer(MB),bandwidth(Mbps),burst_lat_avg(ms),burst_lat_min(ms),burst_lat_max(ms),burst_lat_stdev(ms),cnt,size,inP,NetPwr,Reads=Dist,end,ns" > "iperf/${BW}_${TDD}/DL.csv"
            ;;
        * )
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
else
    APPEND_MODE=false
    echo "start,ns,device,interval(s),transfer(MB),bandwidth(Mbps),burst_lat_avg(ms),burst_lat_min(ms),burst_lat_max(ms),burst_lat_stdev(ms),cnt,size,inP,NetPwr,Reads=Dist,end,ns" > "iperf/${BW}_${TDD}/UL.csv"
    echo "start,ns,device,interval(s),transfer(MB),bandwidth(Mbps),burst_lat_avg(ms),burst_lat_min(ms),burst_lat_max(ms),burst_lat_stdev(ms),cnt,size,inP,NetPwr,Reads=Dist,end,ns" > "iperf/${BW}_${TDD}/DL.csv"
fi

# Determine which clients to test
if [ "$SINGLE_CLIENT" -gt 0 ]; then
    echo "Running test for single client: pi0${SINGLE_CLIENT}"
    START_CLIENT=$SINGLE_CLIENT
    END_CLIENT=$SINGLE_CLIENT
else
    echo "Running test for clients 1 through ${NUM_CLIENTS}"
    START_CLIENT=1
    END_CLIENT=$NUM_CLIENTS
fi

# # Disconnect all devices from 5G (optional, only if running all clients)
# if [ "$SINGLE_CLIENT" -eq 0 ]; then
#     echo "Disconnecting all devices from 5G"
#     for ((CID=1;CID<=NUM_CLIENTS;CID++)); do
#         LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}
#         ssh "$LOGIN" sudo nmcli connection down OAI
#         ssh "$LOGIN" sudo nmcli r wwan off
#     done
# fi

# i could also automate oaibox setup and teardown through running oaibox offline...

for ((CID=START_CLIENT;CID<=END_CLIENT;CID++)); do
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
    start=$(date -u -Ins)
    echo start ${start}
    ssh user@10.105.46.208 docker exec oai-ext-dn iperf -t ${TIME} -i 1 -fm -B 172.31.0.135 -p 5001 -c ${wwan} --trip-times #--txdelay-time 1
    end=$(date -u -Ins)
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
    start=$(date -u -Ins)
    echo start ${start}
    ssh "$LOGIN" iperf -t ${TIME} -i 1 -fm -c 172.31.0.135 -B "$wwan" --trip-times #--txdelay-time 3
    end=$(date -u -Ins)
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

# need to parse phys layer metrics for collected data