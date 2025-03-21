pkill -9 -f "kubectl port-forward"
helm uninstall vllm
minikube stop
minikube delete
docker remove $(docker ps -q)