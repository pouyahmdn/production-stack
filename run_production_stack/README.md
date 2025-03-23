# Setup

## 1. Install necessary packages and drivers (ONLY ONCE)

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

## 2. Start cluster

In case you are resuming a stopped VM, first relaunch the minikube kubernetes cluster:

```bash
bash 6-reload_minikube.sh
```

Create a new screen session and start a cluster with one of the configs:

```bash
screen -S cluster
bash 3-turn_on_cluster.sh config/llama3-4gpu.yaml
```

After port forwarding starts, detach (CTRL + A + D). Setting up the cluster can take 5-10 minutes. You can query the progress with:

```bash
watch -n 1 'kubectl get pods'
```

After startup, query available models with this command:

```bash
curl -o- http://localhost:30080/v1/models
```

Then launch an inference request like this:

```bash
curl -X POST http://localhost:30080/v1/completions -H "Content-Type: application/json" -d '{
    "model": "MODEL_TYPE",
    "prompt": "Who are you?",
    "max_tokens": 50
  }'
```

where you replace `MODEL_TYPE` with the served model name returned from the previous `curl` query.

## 3. Running benchmark

Run the following commands:

```bash
cd ../../benchmarks/multi-round-qa/
screen -r benchmark
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