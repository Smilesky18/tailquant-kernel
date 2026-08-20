#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
ROMEO=${ROMEO:-/data/yzy/RoMeo}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}

LABEL=${LABEL:?need LABEL}
MODEL=${MODEL:?need MODEL}
POLICY=${POLICY:?need POLICY}
GPU=${GPU:?need GPU}

STATUS=$EXP/status/${LABEL}_full_layer_latency_v28d.status
OUT=$EXP/results/${LABEL}_full_layer_latency_v28d
QFCACHE=$EXP/reports/qfactory_cache_${LABEL}_v28d

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT" "$QFCACHE"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[LABEL] $LABEL"
echo "[MODEL] $MODEL"
echo "[POLICY] $POLICY"
echo "[GPU] $GPU"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[OUT] $OUT"

if [ ! -f "$POLICY" ]; then
  echo "[ERROR] policy file not found: $POLICY"
  exit 2
fi

cd "$ROOT"

export HF_HOME=/data/yzy/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

export QFACTORY_ARCH=120
export QFACTORY_CACHE_DIR="$QFCACHE"
export PYTHONPATH="$ROMEO:$ROOT:$EXP/tools:${PYTHONPATH:-}"

echo "[SMOKE] layer0 batch16"
"$PYBIN" -u "$EXP/tools/bench_full_llama_layer_latency_policy_v28d.py" \
  --model "$MODEL" \
  --policy "$POLICY" \
  --label "${LABEL}_smoke" \
  --out_dir "$OUT/smoke" \
  --device cuda:0 \
  --seq_len 128 \
  --batches 16 \
  --layers 0 \
  --warmup 2 \
  --iters 5 \
  --eps 1e-8

echo "[FULL] all layers batch16,64"
"$PYBIN" -u "$EXP/tools/bench_full_llama_layer_latency_policy_v28d.py" \
  --model "$MODEL" \
  --policy "$POLICY" \
  --label "$LABEL" \
  --out_dir "$OUT" \
  --device cuda:0 \
  --seq_len 128 \
  --batches 16,64 \
  --layers all \
  --warmup 8 \
  --iters 30 \
  --eps 1e-8

echo "[DONE] $(date -Is)"
