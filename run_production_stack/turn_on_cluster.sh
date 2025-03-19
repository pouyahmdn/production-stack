helm repo add vllm https://vllm-project.github.io/production-stack
helm install vllm vllm/vllm-stack -f llama3-1gpu.yaml --set servingEngineSpec.modelSpec.hf_token=$HFAPI_TOKEN
watch -n 1 'kubectl get pods'
curl -o- http://localhost:30080/v1/models
curl -X POST http://localhost:30080/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "prompt": "Who are you?",
    "max_tokens": 50
  }'