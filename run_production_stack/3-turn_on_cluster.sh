helm repo add vllm https://vllm-project.github.io/production-stack
helm install vllm vllm/vllm-stack -f $1 --set servingEngineSpec.modelSpec[0].hf_token="$HFAPI_TOKEN"
watch -n 1 'kubectl get pods'
screen -dmS port kubectl port-forward svc/vllm-router-service 30080:80
echo "Port forwarding started. Use 'pkill -9 -f \"kubectl port-forward\"' to stop."
curl -o- http://localhost:30080/v1/models
echo "curl -X POST http://localhost:30080/v1/completions   -H \"Content-Type: application/json\"   -d '"'{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "prompt": "Who are you?",
    "max_tokens": 50
  }'