for i in $(seq 1 10);
do
	flwr run . --stream
	sleep 2
	kill -9 $(pgrep tshark)
done
