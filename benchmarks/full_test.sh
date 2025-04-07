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

if [ -f "$SCRIPT_DIR/ShareGPT.json" ]; then
  echo "File exists. Continuing..."
else
  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
  python3 cleanup_sharegpt.py --model $MODEL --share_gpt_path ShareGPT_V3_unfiltered_cleaned_split.json
  mv ShareGPT_V3_unfiltered_cleaned_split.json ShareGPT.json
fi

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
QPS_VALUES=(32.0 16.0 8.0 4.0 2.0 1.0)

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
        --ignore-eos \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --output "$2" \
        --log-interval 30 \
        --time 600

    sleep 10
}

# Run benchmarks for the determined QPS values
for qps in "${QPS_VALUES[@]}"; do
    output_file="${KEY}_output_${qps}.csv"
    run_benchmark "$qps" "$output_file"
done
