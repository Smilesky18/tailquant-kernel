#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/quarot-clean/bin/python}

STATUS=$EXP/status/standalone_int4_vs_torch_gemm_v12.status
OUT=$EXP/results/standalone_int4_vs_torch_gemm_v12

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[PYBIN] $PYBIN"
echo "[OUT] $OUT"

cd "$ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTHONPATH="$ROOT:$EXP/tools:${PYTHONPATH:-}"

"$PYBIN" -u "$EXP/tools/bench_standalone_int4_vs_torch_gemm_v12.py" \
  --device cuda:0 \
  --warmup 20 \
  --iters 100 \
  --eps 1e-8 \
  --out_dir "$OUT"

echo "[DONE] $(date -Is)"
