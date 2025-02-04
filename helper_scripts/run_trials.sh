for i in $(seq 1 10);
do
	./dataset_distributor.sh 5  # N - number of clients
	#flwr run . --stream
	sleep 2
	kill -9 $(pgrep tshark)
done
