#!/bin/bash

ssh-keygen -t rsa
cd ~/.ssh/ || exit
for ((CID=1;CID<=6;CID++)); do
	LOGIN=commnetpi0$CID@129.105.6.$((CID+16))
	scp id_rsa.pub $LOGIN:
	ssh $LOGIN "cat id_rsa.pub >> .ssh/authorized_keys"
done
