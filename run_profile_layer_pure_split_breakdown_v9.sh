#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/quarot-clean/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-8B}

STATUS=$EXP/status/profile_layer_pure_split_breakdown_v9.status
OUT=$EXP/results/profile_layer_pure_split_breakdown_v9

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/reports" "$EXP/results" "$OUT"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[PYBIN] $PYBIN"
echo "[MODEL] $MODEL"
echo "[OUT] $OUT"

cd "$ROOT"

export HF_HOME=/data/yzy/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTHONPATH="$ROOT:$EXP/tools:${PYTHONPATH:-}"

"$PYBIN" -u "$EXP/tools/profile_layer_pure_split_breakdown_v9.py" \
  --model "$MODEL" \
  --layer_idx 0 \
  --seq_len 128 \
  --batches 16,64 \
  --device cuda:0 \
  --warmup 5 \
  --profile_runs 5 \
  --split_ratio 0.05 \
  --eps 1e-8 \
  --out_dir "$OUT"

echo "[DONE] $(date -Is)"
