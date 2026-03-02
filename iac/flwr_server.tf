terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "7.20.0"
    }
  }
}

provider "google" {
  project = "mcc-rch-sas-cbrs"
  region  = "us-central1"
  zone    = "us-central1-a"
}

variable "fed_5g_repo" {
  type        = string
  description = "GitHub Repository for fed5g repository"
}

variable "docker_server_path" {
  type        = string
  description = "Path to the docker compose files."
}

resource "google_compute_network" "flwr_vpc" {
  name                    = "flwr-network"
}


resource "google_compute_firewall" "flwr_firewall" {
  name    = "flwr-firewall"
  network = google_compute_network.flwr_vpc.name

  allow {
    protocol = "tcp"
    ports    = ["9092", "9093", "22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["flwr-server"]
}

resource "google_compute_instance" "flwr_compute" {
  name         = "flwr-fl-server"
  machine_type = "n1-standard-2"
  zone         = "us-central1-a"

  tags = ["flwr-server", "http-server", "https-server"]

  boot_disk {
    initialize_params {
      image        = "projects/ubuntu-os-cloud/global/images/ubuntu-minimal-2404-noble-amd64-v20260219"
      size         = "100"
      architecture = "X86_64"
      type = "pd-balanced"
    }
  }

  network_interface {
    network = google_compute_network.flwr_vpc.name
    access_config {
      network_tier = "PREMIUM"
    }
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e

    exec > >(tee /var/log/startup-script.log)
    exec 2>&1

    # Install Docker properly
    apt-get update
    apt-get install -y ca-certificates curl
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    chmod a+r /etc/apt/keyrings/docker.asc

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update
    apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

    usermod -aG docker ubuntu

    mkdir -p /home/ubuntu/flwr_output
    chown -R ubuntu:ubuntu /home/ubuntu

    su - ubuntu -c 'cd /home/ubuntu && git clone ${var.fed_5g_repo} fed5g'
    su - ubuntu -c 'cd /home/ubuntu/fed5g/${var.docker_server_path} && docker compose -f docker-compose.cellular.yml up -d'

    echo "Setup Completed"

  EOF
}

output "server_external_ip" {
  value       = google_compute_instance.flwr_compute.network_interface[0].access_config[0].nat_ip
  description = "External IP of FlowerAI server"
}

output "ssh_command" {
  value       = "gcloud compute ssh ubuntu@flwr-fl-server --zone=us-central1-a"
  description = "Command to SSH to server as ubuntu user"
}

output "check_deployment_logs" {
  value       = "gcloud compute ssh ubuntu@flwr-fl-server --zone=us-central1-a --command='tail -f /var/log/startup-script.log'"
  description = "Watch deployment progress in real-time"
}

output "check_containers" {
  value       = "gcloud compute ssh ubuntu@flwr-fl-server --zone=us-central1-a --command='docker ps'"
  description = "Check if containers are running"
}

output "download_results" {
  value       = "gcloud compute scp --recurse ubuntu@flwr-fl-server:/home/ubuntu/flwr_output ./results --zone=us-central1-a"
  description = "Download FL experiment results after completion"
}
