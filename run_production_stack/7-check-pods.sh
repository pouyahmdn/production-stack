#!/bin/bash

# Get all pod names that match the prefix
pods=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep '^vllm-llama3-deployment-vllm')

# Check if any pods were found
if [ -z "$pods" ]; then
  echo "No pods found matching the prefix 'vllm-llama3-deployment-vllm'"
  exit 1
fi

# Determine if a search argument was provided
if [ $# -eq 0 ]; then
  # No argument: show last 100 lines of logs for each pod
  for pod in $pods; do
    echo "====== Last 100 lines for pod: $pod ======"
    kubectl logs "$pod" --tail=100
    echo ""
  done
else
  # Argument provided: grep logs and show last 5 matching lines
  query=$1
  for pod in $pods; do
    echo "====== Last 5 matches for '$query' in pod: $pod ======"
    kubectl logs "$pod" --tail=100 | grep "$query" | tail -n 5
    echo ""
  done
fi