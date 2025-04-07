#!/bin/bash

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <model> <base url> <save file key>"
    exit 1
fi

# Get the script directory to reference local scripts reliably.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

. "$SCRIPT_DIR/../run_production_stack/2-set_api_tokens.sh"

MODEL=$1
BASE_URL=$2
KEY=$3

python3 cleanup_sharegpt.py --model $MODEL --share_gpt_path ShareGPT.json

python3 ./multi-round-qa.py \
        --user-lag 0 \
        --num-rounds 2 \
        --qps 2 \
        --sharegpt \
        --answer-len 512 \
        --ignore-eos \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --output /tmp/warmup.csv \
        --log-interval 30 \
        --time 200

# CONFIGURATION
QPS_VALUES=(12.0 6.0 3.0 1.5)

run_benchmark() {
    # $1: qps
    # $2: output file

    # Real run
    python3 ./multi-round-qa.py \
        --user-lag 10 \
        --num-rounds 1 \
        --qps "$1" \
        --sharegpt \
        --answer-len 4096 \
        --input-inflate-rate 0.05 \
        --output-inflate-rate 0.05 \
        --input-inflate-mult 10 \
        --output-inflate-mult 10 \
        --ignore-eos \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --output "$2" \
        --log-interval 30 \
        --time 300

    sleep 10
}

# Run benchmarks for the determined QPS values
for qps in "${QPS_VALUES[@]}"; do
    output_file="${KEY}_output_${qps}.csv"
    run_benchmark "$qps" "$output_file"
done
