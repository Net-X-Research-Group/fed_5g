terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "7.20.0"
    }
  }
}

provider "google" {
  project = "federated_5g"
  region  = "us-central1"
  zone    = "us-central1-a"
}

resource "google_compute_network" "flwr_vpc" {
  name = "flwr-network"
}

resource "google_compute_instance" "flwr_compute" {
  name         = "flwr-fl-server"
  machine_type = "n1-standard-2"
  zone         = "us-central1-a"

  boot_disk {
    initialize_params {
      image = "ubuntu/ubuntu-2204-lts"
      size = "100"
      architecture = "X86_64"
    }
  }

  network_interface {
    network = google_compute_network.flwr_vpc.name
    access_config {
      # Ephemeral public IP
    }
  }

}

resource "google_compute_firewall" "flwr_firewall" {
  name = "flwr-firewall"
  network = google_compute_network.flwr_vpc.name

  allow {
    protocol = "tcp"
    ports = ["9092", "9093", "22"]
  }

  source_ranges = ["0.0.0.0/0"]

}



