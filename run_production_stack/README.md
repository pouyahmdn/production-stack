# Setup

## 1. Install necessary packages and drivers and prepare (ONLY ONCE)

First run:

```bash
bash 0-install-docker.sh
```

After the script is finished, then run:

```bash
bash 1-install-all.sh
bash 4-prep_python.sh
```

Add hugging face token to `2-set_api_tokens.sh`:
```bash
vim 2-set_api_tokens.sh
```

Then edit `limits.conf`:
```bash
sudo vim /etc/security/limits.conf
```
and change ulimit cap by adding:
```bash
* hard nofile 524288
* soft nofile 524288
```
Now, logout and login again and verify ulimit cap:
```bash
ulimit -a -S | grep "open files"
ulimit -a -H | grep "open files"
```

## 2. Start cluster

In case you are resuming a stopped VM, first relaunch the minikube kubernetes cluster:

```bash
bash 6-reload_minikube.sh
```

Create a new screen session and start a cluster with one of the configs:

```bash
screen -R cluster
bash 3-turn_on_cluster.sh config/llama3-4gpu.yaml
```

You can query the progress with:

```bash
watch -n 1 'kubectl get pods'
```

Setting up the cluster can take 5-10 minutes. After the cluster is running, start port forwarding and detach (CTRL + A + D): 

```bash
kubectl port-forward svc/vllm-router-service 30080:80
```

After startup, query available models with this command:

```bash
curl -o- http://localhost:30080/v1/models
```

Then launch an inference request like this:

```bash
curl -X POST http://localhost:30080/v1/completions -H "Content-Type: application/json" -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "prompt": "Who are you?",
    "max_tokens": 50,
    "stream": "true"
  }'
```

where you replace `meta-llama/Meta-Llama-3-8B-Instruct` with the served model name returned from the previous `curl` query.

## 3. Running benchmark

Run the following commands:

```bash
cd ../benchmarks/
screen -R benchmark
source ~/prodstack/bin/activate
bash full_test.sh MODEL_TYPE http://localhost:30080/v1 TEST_NAME
```

where you replace `MODEL_TYPE` with the served model name returned from the previous `curl` query, and replace `TEST_NAME` with a name for this benchmark experiment; data is saved as `[TEST_TYPE]_output_[QPS].csv`.

## 4. Turn off cluster

Stop cluster by going back to the cluster screen session:

```bash
screen -r cluster
```

Stop port forwarding by CTRL + C, and then run:

```bash
bash 5-turn_off_cluster.sh
```

## Useful commands:

You can see GPU usage with `nvidia-smi`:

```bash
watch -n 1 'nvidia-smi'
```

## 5. Build Router (Only use if you change vllm-router source code)

```bash
eval $(minikube docker-env)
docker build -t pouyah/vllm_router_custom:latest -f docker/Dockerfile .
```

Optionally push the image to docker hub (will take several minutes):

```bash
docker push pouyah/vllm_router_custom:latest
```

### TODO

1. Observe queuing stats + engine stats more rapidly. Can we access metrics from vllm instances?
2. Experiment with other knobs.
3. Try larger instance sizes.