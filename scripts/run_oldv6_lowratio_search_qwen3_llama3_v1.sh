#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
cd "$ROOT"

export PATH="/data/yzy/miniconda3/envs/romeo_sm120/bin:${PATH:-}"
export PYTHONPATH="$ROOT:$ROOT/fake_quant:$ROOT/kernel_quant:$ROOT/kernel_quant/scripts:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/data/yzy/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

SCRIPT="$ROOT/experiments/kernel_quant/layer_latency_split_v1/tools/calibrate_per_linear_v6_lowratio_v1.py"
OUT_ROOT="$ROOT/experiments/kernel_quant/layer_latency_split_v1/results/oldv6_lowratio_search_v2"
mkdir -p "$OUT_ROOT/logs"

pids=()

echo "[LAUNCH] Qwen3-8B low-ratio old-v6 search on GPU ${QWEN_GPU:-1}"
(
  export CUDA_VISIBLE_DEVICES=${QWEN_GPU:-1}
  python -u "$SCRIPT" \
    --model Qwen/Qwen3-8B \
    --dataset wikitext2 \
    --out_dir "$OUT_ROOT/qwen3_8b/split_lambda_0p08_lowratio" \
    --rotation_config /data/yzy/quarot/qwen3-8B_layer_all.csv \
    --nsamples 8 \
    --seqlen 256 \
    --steps 80 \
    --eval_every 5 \
    --capture_rows 1024 \
    --train_rows 128 \
    --val_rows 256 \
    --train_out_channels 256 \
    --val_out_channels 256 \
    --ratio_lambda 0.08 \
    --init_ratio 0.04 \
    --max_ratio 0.16 \
    --init_activation_percentile 99.75 \
    --init_weight_percentile 99.9 \
    --min_percentile 0.98 \
    --recon_tolerance_rel 0.05 \
    --recon_tolerance_abs 1e-5 \
    --seed 0 \
    --eps 1e-8
) > "$OUT_ROOT/logs/qwen3_8b_search.log" 2>&1 &
pids+=("$!")

echo "[LAUNCH] Llama3-8B low-ratio old-v6 search on GPU ${LLAMA_GPU:-2}"
(
  export CUDA_VISIBLE_DEVICES=${LLAMA_GPU:-2}
  python -u "$SCRIPT" \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset wikitext2 \
    --out_dir "$OUT_ROOT/llama3_8b/split_lambda_0p08_lowratio" \
    --rotation_config /data/yzy/quarot/llama3-8B_layer.csv \
    --nsamples 128 \
    --seqlen 2048 \
    --steps 300 \
    --eval_every 100 \
    --capture_rows 4096 \
    --train_rows 3072 \
    --val_rows 1024 \
    --train_out_channels 512 \
    --val_out_channels 512 \
    --ratio_lambda 0.08 \
    --init_ratio 0.08 \
    --max_ratio 0.10 \
    --init_activation_percentile 0.9975 \
    --init_weight_percentile 0.9975 \
    --min_percentile 0.96 \
    --recon_tolerance_rel 0.05 \
    --recon_tolerance_abs 1e-5 \
    --seed 0 \
    --eps 1e-8
) > "$OUT_ROOT/logs/llama3_8b_search.log" 2>&1 &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[DONE] one or more searches failed"
  exit "$failed"
fi

echo "[DONE] all searches finished"
echo "$OUT_ROOT/qwen3_8b/split_lambda_0p08_lowratio/policy.json"
echo "$OUT_ROOT/llama3_8b/split_lambda_0p08_lowratio/policy.json"
