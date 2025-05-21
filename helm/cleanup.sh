#!/bin/bash

# Get routing logic from environment variable or command line argument
ROUTING_LOGIC=${ROUTING_LOGIC:-llq}  # Default to llq if not provided
if [ "$1" != "" ]; then
  ROUTING_LOGIC=$1
fi

set -ex

helm uninstall vllm-${ROUTING_LOGIC} --namespace ${ROUTING_LOGIC}

# Use like this
# ./cleanup.sh llq