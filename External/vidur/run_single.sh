
#!/bin/bash
set -ex


MODEL="meta-llama/Meta-Llama-3-8B"
NUM_REPLICAS=4
MAX_INPUT_LEN=4096
MAX_OUTPUT_LEN=4096
TIME=1000
INPUT_INFLATE_RATE=0.05
OUTPUT_INFLATE_RATE=0.05
INPUT_INFLATE_MULT=10
OUTPUT_INFLATE_MULT=10


# Get the script directory to reference local scripts reliably.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

. "$SCRIPT_DIR/set_hf_token.sh"

if [[ "$3" == "retrace" ]]; then
  cd "$SCRIPT_DIR/data/processed_traces"
  python3 cleanup_sharegpt.py --model $MODEL --share_gpt_path ShareGPT.json
  python3 trace_gen.py --qps $2 \
                       --max-input-len $MAX_INPUT_LEN --max-output-len $MAX_OUTPUT_LEN --time $TIME \
                       --input-inflate-rate $INPUT_INFLATE_RATE --output-inflate-rate $OUTPUT_INFLATE_RATE \
                       --input-inflate-mult $INPUT_INFLATE_MULT --output-inflate-mult $OUTPUT_INFLATE_MULT \
                       --output "sharegpt_$2.csv"
  cd "$SCRIPT_DIR"
elif [[ "$3" == "no_retrace" ]]; then
  echo "Skipping creating shareGPT traces, hoping they exist..."
else
  echo "$3 has no meaning for trace generation; should have been 'retrace' or 'no_trace'"
  exit
fi

# --replica_config_memory_margin_fraction 0.255 \
# --metrics_config_write_json_trace \
# --metrics_config_keep_individual_batch_metrics
python -m vidur.main  \
 --replica_config_device a10 \
 --replica_config_memory_margin_fraction 0.2445 \
 --replica_config_model_name $MODEL \
 --cluster_config_num_replicas $NUM_REPLICAS \
 --global_scheduler_config_type $1 \
 --replica_config_tensor_parallel_size 1 \
 --replica_config_num_pipeline_stages 1 \
 --request_generator_config_type trace_replay \
 --trace_request_generator_config_trace_file "$SCRIPT_DIR/data/processed_traces/sharegpt_$2.csv" \
 --trace_request_generator_config_max_tokens 8192 \
 --length_generator_config_type trace \
 --trace_request_length_generator_config_trace_file "$SCRIPT_DIR/data/processed_traces/sharegpt_$2.csv" \
 --trace_request_length_generator_config_max_tokens 8192 \
 --interval_generator_config_type trace \
 --trace_request_interval_generator_config_trace_file "$SCRIPT_DIR/data/processed_traces/sharegpt_$2.csv" \
 --replica_scheduler_config_type vllm \
 --vllm_scheduler_config_batch_size_cap 128 \
 --vllm_scheduler_config_max_tokens_in_batch 8192 \
 --vllm_scheduler_config_block_size 16 \
 --vllm_scheduler_config_watermark_blocks_fraction 0.01 \
 --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 8192 \
 --random_forrest_execution_time_predictor_config_prediction_max_batch_size 128 \
 --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 8192 \
 --metrics_config_write_metrics \
 --no-metrics_config_write_json_trace \
 --no-metrics_config_enable_chrome_trace \
 --no-metrics_config_save_table_to_wandb \
 --no-metrics_config_store_plots \
 --no-metrics_config_store_operation_metrics \
 --no-metrics_config_store_token_completion_metrics \
 --metrics_config_store_request_metrics \
 --no-metrics_config_store_batch_metrics \
 --no-metrics_config_store_utilization_metrics \
 --no-metrics_config_keep_individual_batch_metrics \
 --metrics_config_output_dir "simulator_results/sharegpt_$1_$2"