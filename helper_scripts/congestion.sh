#!/bin/bash
STOP="$HOME/5gnr_testbed_scripts/stop_telit_modem_qmicli.sh"
START="$HOME/5gnr_testbed_scripts/start_telit_modem_qmicli.sh"
CLIENT="commnetpi05@129.105.6.21"
HOST="commnetpi06@129.105.6.22"
IPERF_PORT=5023
TEST_DURATION=7200

graceful_quit() {
    ssh $CLIENT "pkill -f iperf3"
    pkill -f iperf3
    exit 0
}

check_connection() {
    local target="$1"  # "local" or "remote"
    local ping_output
    local packet_loss
    
    echo "Checking ${target} connection..."
    
    if [ "$target" = "local" ]; then
        # Run ping locally
        ping_output=$(ping 8.8.8.8 -I wwan0 -c 3)
    else
        # Run ping remotely
        ssh $CLIENT "ping 8.8.8.8 -I wwan0 -c 3" > /tmp/remote_ping.txt
        ping_output=$(<"/tmp/remote_ping.txt")
    fi
    
    # Extract packet loss using regex
    if [[ $ping_output =~ ([0-9]+)%\ packet\ loss ]]; then
        packet_loss=${BASH_REMATCH[1]}
        echo "${target^} packet loss: $packet_loss%"
        return $packet_loss
    else
        echo "Failed to check ${target} connection"
        return 100  # Return 100 to indicate error
    fi
}

restart_modem() {
    local target="$1"  # "local" or "remote"
    
    echo "Restarting ${target} modem..."
    
    if [ "$target" = "local" ]; then
        $STOP
        sleep 2
        $START &
    else
        ssh $CLIENT "
            $STOP
            sleep 2
            $START &
        "
    fi
}

# Get wwan0 IP address for either local or remote machine
get_wwan0_ip() {
    local target="$1"  # "local" or "remote"
    local wwan0_ip
    
    echo "Getting wwan0 IP address on ${target} machine..."
    
    if [ "$target" = "local" ]; then
        # Get IP locally
        wwan0_ip=$(ip addr show wwan0 | grep -oP 'inet \K[\d.]+' || ifconfig wwan0 | grep -oP 'inet addr:\K[\d.]+')
    else
        # Get IP remotely
        wwan0_ip=$(ssh $CLIENT "ip addr show wwan0 | grep -oP 'inet \K[\d.]+' || ifconfig wwan0 | grep -oP 'inet addr:\K[\d.]+'")
    fi
    
    if [[ -z "$wwan0_ip" ]]; then
        echo "Error: Could not determine wwan0 IP address on ${target} machine" >&2
        return 1
    fi
    
    echo "Found ${target} wwan0 IP: $wwan0_ip"
    echo "$wwan0_ip"
    return 0
}

# Combined function to check if iperf3 is running
check_iperf() {
    local target="$1"  # "local" or "remote"
    local is_running
    
    echo "Checking if iperf3 is running on ${target} device..."
    
    if [ "$target" = "local" ]; then
        # Get count locally, trim whitespace
        is_running=$(pgrep -c iperf3 2>/dev/null | tr -d ' \t\n\r' || echo "0")
    else
        # Get count remotely, trim whitespace
        is_running=$(ssh $CLIENT "pgrep -c iperf3 2>/dev/null | tr -d ' \t\n\r'" || echo "0")
    fi
    
    # Debug the actual value
    echo "DEBUG: Raw count value: '$is_running'"
    
    # First check if it's empty or exactly "0"
    if [ -z "$is_running" ] || [ "$is_running" = "0" ]; then
        echo "iperf3 is not running on ${target} device (case 1)"
        return 1
    # Then check if it contains a 0 (e.g., "0 0" case)
    elif echo "$is_running" | grep -q "^0"; then
        echo "iperf3 is not running on ${target} device (case 2)"
        return 1
    # If it's any other value, we assume it's a non-zero count
    else
        echo "iperf3 is running on ${target} device (processes: $is_running)"
        return 0
    fi
}

# Start iperf3 server binding to wwan0 interface
start_iperf_server() {
    get_wwan0_ip "local"
    local wwan0_ip=$?
    
    if [ $? -ne 0 ] || [ -z "$wwan0_ip" ]; then
        echo "Failed to get wwan0 IP, cannot start server"
        return 1  # Return error code to indicate failure
    else
        echo "Starting iperf3 server bound to $wwan0_ip"
        iperf3 -s -p $IPERF_PORT -B $wwan0_ip -V &
        sleep 2
        return 0
    fi
}

# Run iperf3 test
run_iperf_test() {
    echo "Running iperf3 test..."
    
    # Get wwan0 IP address on the CLIENT machine
    local wwan0_ip=$(get_wwan0_ip "remote")
    
    if [ $? -ne 0 ] || [ -z "$wwan0_ip" ]; then
        echo "Error: Could not determine wwan0 IP address on client"
        return 1
    fi
    
    # Run iperf3 with specific IP binding
    ssh $CLIENT "iperf3 -c $HOST -B $wwan0_ip -p $IPERF_PORT -V -t $TEST_DURATION --bidir" || echo "iperf3 test failed"
}

# Beginning of sequential execution
pkill -f iperf3
trap 'graceful_quit' SIGINT SIGTERM

# Main loop
while true; do
    check_connection "local"
    local_loss=$?
    
    check_connection "remote"
    remote_loss=$?
    
    # Handle local connection issues
    if [ $local_loss -lt 100 ]; then
        echo "Local network has $local_loss% packet loss"
    else
        echo "Local connection failed. Restarting local modem..."
        restart_modem "local"
    fi
    
    # Handle remote connection issues
    if [ $remote_loss -lt 100 ]; then
        echo "Remote network has $remote_loss% packet loss"
    else
        echo "Remote connection failed. Restarting remote modem..."
        restart_modem "remote"
    fi
    

    # Check if local iperf3 server is running, start if needed
    if ! check_iperf "local"; then
        echo "Starting local iperf3 server..."
        start_iperf_server
        if [ $? -ne 0 ]; then
            echo "Failed to start iperf3 server due to wwan0 issues. Restarting modem..."
            restart_modem "local"
            continue  # Skip to next iteration of the loop
        fi
    fi
    
    # Only try to run client if the server is running
    if check_iperf "local"; then
        # Check if iperf3 client is running on remote device
        if ! check_iperf "remote"; then
            echo "Starting iperf3 test..."
            run_iperf_test
        fi
    else
        echo "Local iperf3 server is not running, skipping client test"
    fi
    
    echo "Waiting before next check cycle..."
    sleep 30
done