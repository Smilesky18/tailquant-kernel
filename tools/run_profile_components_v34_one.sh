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
BATCHES=${BATCHES:-16,64,256}
LAYERS=${LAYERS:-all}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}
COMPONENT_WARMUP=${COMPONENT_WARMUP:-2}
COMPONENT_ITERS=${COMPONENT_ITERS:-5}

OUT=$EXP/results/profile_components_v34_${LABEL}
STATUS=$EXP/status/profile_components_v34_${LABEL}.status
QFCACHE=${QFCACHE:-$EXP/reports/qfactory_cache_profile_components_v34_${LABEL}}

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT" "$QFCACHE"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"
echo "[START] $(date -Is)"
echo "[LABEL] $LABEL"
echo "[MODEL] $MODEL"
echo "[POLICY] $POLICY"
echo "[GPU] $GPU"
echo "[BATCHES] $BATCHES"
echo "[LAYERS] $LAYERS"
echo "[OUT] $OUT"
echo "[QFCACHE] $QFCACHE"

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
export PATH="/data/yzy/miniconda3/envs/romeo_sm120/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

export QFACTORY_ARCH=120
export QFACTORY_CACHE_DIR="$QFCACHE"
export PYTHONPATH="$ROMEO:$ROOT:$EXP/tools:${PYTHONPATH:-}"

"$PYBIN" -u "$EXP/tools/profile_multimodel_split_components_v34.py" \
  --model "$MODEL" \
  --label "$LABEL" \
  --policy "$POLICY" \
  --out_dir "$OUT" \
  --device cuda:0 \
  --seq_len 128 \
  --batches "$BATCHES" \
  --layers "$LAYERS" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --component_warmup "$COMPONENT_WARMUP" \
  --component_iters "$COMPONENT_ITERS" \
  --eps 1e-8 \
  --local_files_only

echo "[DONE] $(date -Is)"
