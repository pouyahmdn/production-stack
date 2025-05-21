#!/bin/bash
set -e

# Get routing logic from environment variable or command line argument
ROUTING_LOGIC=${ROUTING_LOGIC:-llq}  # Default to llq if not provided
if [ "$1" != "" ]; then
  ROUTING_LOGIC=$1
fi
# Get number of replicas from command line argument or default to 4
REPLICAS=${2:-4}
# Get environment from command line argument or default to test
ENVIRONMENT=${3:-test}


helm upgrade --install vllm-${ROUTING_LOGIC} . -f values-benchmark-router.yaml \
    --namespace ${ROUTING_LOGIC} --create-namespace \
    --set routerSpec.routingLogic=${ROUTING_LOGIC} \
    --set environment=${ENVIRONMENT} \
    --set "routerSpec.labels.environment=${ENVIRONMENT}" \
    --set wandb.apiKey=${WANDB_API_KEY} \
    --set "servingEngineSpec.modelSpec[0].hf_token=${HF_TOKEN}" \
    --set "servingEngineSpec.modelSpec[0].labels.environment=${ENVIRONMENT}" \
    --set "servingEngineSpec.modelSpec[0].replicaCount=${REPLICAS}" \
    ${DRY_RUN:+--dry-run}

# Use like this
# ./deploy_router_benchmark.sh llq 4 test  