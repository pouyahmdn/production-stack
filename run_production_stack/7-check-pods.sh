#!/bin/bash

# Get all pod names that match the prefix
pods=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep '^vllm-llama3-deployment-vllm')

# Check if any pods were found
if [ -z "$pods" ]; then
  echo "No pods found matching the prefix 'vllm-llama3-deployment-vllm'"
  exit 1
fi

# Loop through each pod and print its logs
for pod in $pods; do
  echo "====== Logs for pod: $pod ======"
  kubectl logs "$pod" | tail -n 100 | grep "Avg prompt throughput" | tail -n 5
  echo ""
done
