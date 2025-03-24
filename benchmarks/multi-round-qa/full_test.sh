#!/bin/bash

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <model> <base url> <save file key>"
    exit 1
fi

# Get the script directory to reference local scripts reliably.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

. "$SCRIPT_DIR/../../run_production_stack/2-set-api_tokens.sh"

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
        --num-users 1 \
        --num-rounds 2 \
        --qps 2 \
        --shared-system-prompt 10 \
        --user-history-prompt 10 \
        --answer-len 4096 \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --output /tmp/warmup.csv \
        --log-interval 30 \
        --time 200 \
        --sharegpt

# CONFIGURATION
NUM_USERS=320
NUM_ROUNDS=10
QPS_VALUES=(32.0 24.0 16.0 12.0 8.0 6.0 3.0 2.0 1.5 1.0)

run_benchmark() {
    # $1: qps
    # $2: output file

    # Real run
    python3 ./multi-round-qa.py \
        --num-users $NUM_USERS \
        --num-rounds $NUM_ROUNDS \
        --qps "$1" \
        --shared-system-prompt 10 \
        --user-history-prompt 10 \
        --answer-len 4096 \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --output "$2" \
        --log-interval 30 \
        --time 180 \
        --sharegpt

    sleep 10
}

# Run benchmarks for the determined QPS values
for qps in "${QPS_VALUES[@]}"; do
    output_file="${KEY}_output_${qps}.csv"
    run_benchmark "$qps" "$output_file"
done
