helm repo add vllm https://vllm-project.github.io/production-stack
helm install vllm vllm/vllm-stack -f $1 --set servingEngineSpec.modelSpec[0].hf_token="$HFAPI_TOKEN"
watch -n 1 'kubectl get pods'
screen -dmS port bash -c "kubectl port-forward svc/vllm-router-service 30080:80"
echo "Port forwarding started. Use 'screen -r port' to stop."
curl -o- http://localhost:30080/v1/models
echo "curl -X POST http://localhost:30080/v1/completions   -H \"Content-Type: application/json\"   -d '"'{
    "model": "'$1'",
    "prompt": "Who are you?",
    "max_tokens": 50
  }'