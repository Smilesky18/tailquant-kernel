#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
ROMEO=${ROMEO:-/data/yzy/RoMeo}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-8B}
POLICY=${POLICY:-/data/yzy/quarot-gpt-2/experiments/kernel_quant/qwen_per_linear_diff_calibration_v6_rotate/lambda_0p08/policy.json}

STATUS=$EXP/status/sparse_correction_roofline_v26.status
OUT=$EXP/results/sparse_correction_roofline_v26

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[ROMEO] $ROMEO"
echo "[PYBIN] $PYBIN"
echo "[MODEL] $MODEL"
echo "[POLICY] $POLICY"
echo "[OUT] $OUT"
echo "[TASK] v26 sparse correction roofline / memory-vs-compute analysis"

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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

export QFACTORY_ARCH=120
export QFACTORY_CACHE_DIR="$EXP/reports/qfactory_cache_v26"
export PYTHONPATH="$ROMEO:$ROOT:$EXP/tools:${PYTHONPATH:-}"

"$PYBIN" -u "$EXP/tools/profile_sparse_correction_roofline_v26.py" \
  --model "$MODEL" \
  --policy "$POLICY" \
  --out_dir "$OUT" \
  --device cuda:0 \
  --seq_len 128 \
  --batches 16,64 \
  --layers all \
  --warmup 5 \
  --iters 30 \
  --eps 1e-8

echo "[DONE] $(date -Is)"
