. 2-set_api_tokens.sh
helm repo add vllm https://vllm-project.github.io/production-stack
helm install vllm vllm/vllm-stack -f $1 --set servingEngineSpec.modelSpec[0].hf_token="$HFAPI_TOKEN"