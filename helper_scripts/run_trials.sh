for i in $(seq 1 20);
do
	pushd "$HOME"/fed_5g/helper_scripts || exit
	./dataset_distributor.sh 3  # N - number of clients
	cp cifar10_3_partitions_iid.png cifar10_3_partitions_iid_trial_$i.png
	popd || exit
	sleep 2
	flwr run . --stream
	sleep 2
	kill -9 $(pgrep tshark)
done
