#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QPS_VALUES=(7.0)
SCH_VALUES=("ai")

for qps in "${QPS_VALUES[@]}"; do
    for sch in "${SCH_VALUES[@]}"; do
      if [[ $# -eq 1 ]]; then
        bash "$SCRIPT_DIR/run_single.sh" $sch $qps $1
      else
        bash "$SCRIPT_DIR/run_single.sh" $sch $qps "no_retrace"
      fi
    done
done