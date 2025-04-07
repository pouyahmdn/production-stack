#!/bin/bash
. 2-set_api_tokens.sh

# Get the script directory to reference local scripts reliably.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

helm install vllm "$SCRIPT_DIR/../helm" -f $1 --set servingEngineSpec.modelSpec[0].hf_token="$HFAPI_TOKEN"