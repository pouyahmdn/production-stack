#!/bin/bash

bash "$SCRIPT_DIR/install-drivers.sh"
source ~/.bashrc
bash "$SCRIPT_DIR/install-minikube-cluster.sh"