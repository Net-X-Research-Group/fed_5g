#!/bin/bash

DEVICE_NAME_PREFIX="commnetpi0"
IP_PREFIX="129.105.6."
IP_SUFFIXES=(17 18 19 20 21 22)

for ((CID=1;CID<=6;CID++)); do
    # Create the login string using the current suffix value
    LOGIN=$DEVICE_NAME_PREFIX$CID@$IP_PREFIX${IP_SUFFIXES[$CID-1]}

    echo "Connecting to ${LOGIN}..."

    ssh "${LOGIN}" << 'EOF'
        echo "Installing Docker on $(hostname)..."

        sudo apt update && sudo apt upgrade -y

        sudo apt-get install ca-certificates curl -y
        sudo install -m 0755 -d /etc/apt/keyrings
        sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
        sudo chmod a+r /etc/apt/keyrings/docker.asc

        # Add the repository to Apt sources:
        echo \
        "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian \
        $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        sudo apt-get update
        sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
EOF
done