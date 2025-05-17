#!/bin/bash

DOCKER_CLIENT="docker-fl-client-1"
STOP="$HOME/5gnr_testbed_scripts/stop_telit_modem_qmicli.sh"
START="$HOME/5gnr_testbed_scripts/start_telit_modem_qmicli.sh"
MAX_FAILED_ATTEMPTS=4
FAILED_ATTEMPTS=0
MAX_LOG_AGE_SECONDS=500

# Initialize variables to track log updates
LAST_LOGGED_LINE=""
LAST_UPDATE_TIME=$(date +%s)

while true; do
	LAST_LOG_LINE=$(docker logs -n 1 docker-fl-client-1 2>&1)
	DATE=$(date '+%Y-%m-%d %H:%M:%S')
	echo "$DATE: $LAST_LOG_LINE"

	# Check if the log has changed
	if [ "$LAST_LOG_LINE" != "$LAST_LOGGED_LINE" ]; then
		# Log has changed, update the timestamp
		LAST_LOGGED_LINE="$LAST_LOG_LINE"
		LAST_UPDATE_TIME=$(date +%s)
		echo "Log updated, resetting timeout counter"
	else
		# Log hasn't changed, check if we've exceeded the timeout
		CURRENT_TIME=$(date +%s)
		TIME_DIFF=$((CURRENT_TIME - LAST_UPDATE_TIME))
		echo "Log unchanged for ${TIME_DIFF} seconds"
		
		if [ $TIME_DIFF -gt $MAX_LOG_AGE_SECONDS ]; then
			echo "Log is stale (${TIME_DIFF}s since last update). Restarting Docker containers..."
			docker compose down
			"$STOP"
			sudo ifconfig wwan0 down
			docker compose pull
			docker compose up -d
			FAILED_ATTEMPTS=0
			LAST_UPDATE_TIME=$(date +%s)  # Reset the timer
			sleep 15
			continue
		fi
	fi

	if echo "$LAST_LOG_LINE" | grep "Connection attempt failed"; then
		RETRY_TIME=$(echo "$LAST_LOG_LINE" | grep -oP "retrying in \K[0-9.]+")
		WAIT_TIME=$(echo "$RETRY_TIME" | awk '{print int($1) + 1}') # ceiling of retry time
		echo "Connection failed. Waiting for $WAIT_TIME seconds before checking again"
		sleep "$WAIT_TIME"

		FAILED_ATTEMPTS=$((FAILED_ATTEMPTS + 1))
		if [ "$FAILED_ATTEMPTS" -ge "$MAX_FAILED_ATTEMPTS" ]; then
			echo "Maximum failed attempts reached. Restarting modem..."
			"$STOP"
			"$START" &

			FAILED_ATTEMPTS=0
			sleep 15

#			# Check if start script executed properly"
#			echo "Checking for 'udp broadcast discover'"
#			sleep 15 # Give some time for the start script to initialize
#			LAST_LOG_LINE=$(docker logs -n 1 docker-fl-client-1 2>&1)
#
#			if echo "$LAST_LOG_LINE" | grep -q "udp broadcast discover"; then
#				echo "'udp broadcast discover' detected. Rerunning start/stop scripts"
#				"$STOP"
#				"$START" &
#
#				sleep 15
#			fi
		fi
	elif echo "$LAST_LOG_LINE" | grep -q "Disconnect and shut down"; then
		docker compose down
		docker compose pull
		docker compose up -d
		FAILED_ATTEMPTS=0
		sleep 15
	# elif echo "$LAST_LOG_LINE" | grep -q "Connection successful"; then
	# 	echo "Connection successful. Resetting failed attempts counter."
	# 	FAILED_ATTEMPTS=0
	else
		echo "No connection issues detected. Monitoring continues..."
	fi

	# Wait a short time before checking the logs again
	sleep 15
done
