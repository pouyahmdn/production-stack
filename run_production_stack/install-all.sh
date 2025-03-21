#!/bin/bash
set -e
# --- Debug and Environment Setup ---
echo "Current PATH: $PATH"
echo "Operating System: $(uname -a)"

# Get the script directory to reference local scripts reliably.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/install-drivers.sh"
source ~/.bashrc
bash "$SCRIPT_DIR/install-minikube-cluster.sh"