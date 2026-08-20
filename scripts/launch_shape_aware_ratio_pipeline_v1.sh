#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
OUT=${OUT:-$EXP/results/shape_aware_ratio_search_qwen3_8b_v1}
mkdir -p "$OUT/logs"
nohup bash "$EXP/scripts/run_shape_aware_ratio_pipeline_v1.sh" > "$OUT/logs/shape_aware_ratio_pipeline_v1.log" 2>&1 &
