#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
ROMEO=${ROMEO:-/data/yzy/RoMeo}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-8B}
POLICY=${POLICY:-/data/yzy/quarot-gpt-2/experiments/kernel_quant/qwen_per_linear_diff_calibration_v6_rotate/lambda_0p08/policy.json}

STATUS=$EXP/status/full_qwen3_8b_layer_latency_opt_v23.status
OUT=$EXP/results/full_qwen3_8b_layer_latency_opt_v23
BASELINE=$EXP/results/full_qwen3_8b_layer_latency_policy_projected_v22b

QFCACHE=${QFCACHE:-$EXP/reports/qfactory_cache_v22}

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT" "$QFCACHE"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[ROMEO] $ROMEO"
echo "[EXP] $EXP"
echo "[PYBIN] $PYBIN"
echo "[MODEL] $MODEL"
echo "[POLICY] $POLICY"
echo "[BASELINE] $BASELINE"
echo "[OUT] $OUT"
echo "[QFCACHE] $QFCACHE"
echo "[OPT] inplace_sparse_add + mlp_gate_up_shared_prepare"

if [ ! -f "$POLICY" ]; then
  echo "[ERROR] policy file not found: $POLICY"
  exit 2
fi

if [ ! -f "$BASELINE/layer_latency_all_v22.csv" ]; then
  echo "[ERROR] projected baseline result not found: $BASELINE/layer_latency_all_v22.csv"
  exit 3
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
export QFACTORY_CACHE_DIR="$QFCACHE"
export QFACTORY_NO_PIPELINE=${QFACTORY_NO_PIPELINE:-0}

export PYTHONPATH="$ROMEO:$ROOT:$EXP/tools:${PYTHONPATH:-}"

"$PYBIN" -u "$EXP/tools/bench_full_qwen3_8b_layer_latency_opt_v23.py" \
  --model "$MODEL" \
  --policy "$POLICY" \
  --baseline_dir "$BASELINE" \
  --seq_len 128 \
  --batches 16,64 \
  --device cuda:0 \
  --warmup 5 \
  --iters 30 \
  --eps 1e-8 \
  --check_first_n_layers 3 \
  --out_dir "$OUT"

echo "[DONE] $(date -Is)"
