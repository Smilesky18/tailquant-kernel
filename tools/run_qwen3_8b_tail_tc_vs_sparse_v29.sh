#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
ROMEO=${ROMEO:-/data/yzy/RoMeo}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}
GPU=${GPU:-2}

POLICY=${POLICY:-/data/yzy/quarot-gpt-2/experiments/kernel_quant/qwen_per_linear_diff_calibration_v6_rotate/lambda_0p08/policy.json}
OUT=$EXP/results/qwen3_8b_tail_tc_vs_sparse_v29
STATUS=$EXP/status/qwen3_8b_tail_tc_vs_sparse_v29.status
QFCACHE=$EXP/reports/qfactory_cache_qwen3_8b_tail_tc_vs_sparse_v29

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT" "$QFCACHE"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[POLICY] $POLICY"
echo "[GPU] $GPU"
echo "[OUT] $OUT"

if [ ! -f "$POLICY" ]; then
  echo "[ERROR] policy not found: $POLICY"
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

"$PYBIN" -u "$EXP/tools/bench_qwen3_8b_tail_tc_vs_sparse_v29.py" \
  --model Qwen/Qwen3-8B \
  --policy "$POLICY" \
  --out_dir "$OUT" \
  --device cuda:0 \
  --seq_len 128 \
  --batches 16,64 \
  --layers all \
  --modules all \
  --warmup 8 \
  --iters 20 \
  --sweep_ratios policy,0.001,0.002,0.005,0.01,0.02,0.05,0.08,0.10,0.15 \
  --eps 1e-8

echo "[DONE] $(date -Is)"
