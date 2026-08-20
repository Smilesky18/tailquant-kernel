#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
ROMEO=${ROMEO:-/data/yzy/RoMeo}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}

STATUS=$EXP/status/romeo_qfactory_a4w4_backend_v17.status
OUT=$EXP/results/romeo_qfactory_a4w4_backend_v17
QFCACHE=$EXP/reports/qfactory_cache_v17

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT" "$QFCACHE"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[ROMEO] $ROMEO"
echo "[EXP] $EXP"
echo "[PYBIN] $PYBIN"
echo "[OUT] $OUT"
echo "[QFCACHE] $QFCACHE"

cd "$ROOT"

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export QFACTORY_ARCH=120
export QFACTORY_CACHE_DIR="$QFCACHE"
export QFACTORY_NO_PIPELINE=${QFACTORY_NO_PIPELINE:-0}

export PYTHONPATH="$ROMEO:$ROOT:$EXP/tools:${PYTHONPATH:-}"

"$PYBIN" -u "$EXP/tools/bench_romeo_qfactory_a4w4_backend_v17.py" \
  --device cuda:0 \
  --warmup 10 \
  --iters 50 \
  --eps 1e-8 \
  --out_dir "$OUT"

echo "[DONE] $(date -Is)"
