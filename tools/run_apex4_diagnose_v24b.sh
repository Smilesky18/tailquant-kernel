#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
ROMEO=${ROMEO:-/data/yzy/RoMeo}
APEX4=${APEX4:-/data/yzy/APEX4-W4A4}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}

STATUS=$EXP/status/apex4_diagnose_v24b.status
OUT=$EXP/results/apex4_diagnose_v24b
QFCACHE=$EXP/reports/qfactory_cache_v24b

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT" "$QFCACHE"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[ROMEO] $ROMEO"
echo "[APEX4] $APEX4"
echo "[PYBIN] $PYBIN"
echo "[OUT] $OUT"

if [ ! -d "$APEX4/kernels" ]; then
  echo "[ERROR] APEX4 kernels dir not found: $APEX4/kernels"
  exit 2
fi

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=8

export QFACTORY_ARCH=120
export QFACTORY_CACHE_DIR="$QFCACHE"
export PYTHONPATH="$ROMEO:$ROOT:$EXP/tools:$APEX4/kernels:${PYTHONPATH:-}"

export APEX4="$APEX4"
export OUT="$OUT"

cd "$APEX4/kernels"
"$PYBIN" setup.py build_ext --inplace

cd "$ROOT"
"$PYBIN" -u "$EXP/tools/diagnose_apex4_supported_configs_v24b.py"

echo "[DONE] $(date -Is)"
