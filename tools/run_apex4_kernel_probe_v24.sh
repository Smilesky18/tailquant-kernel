#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
ROMEO=${ROMEO:-/data/yzy/RoMeo}
APEX4=${APEX4:-/data/yzy/APEX4-W4A4}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}

STATUS=$EXP/status/apex4_kernel_probe_v24.status
OUT=$EXP/results/apex4_kernel_probe_v24

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[ROMEO] $ROMEO"
echo "[APEX4] $APEX4"
echo "[PYBIN] $PYBIN"
echo "[OUT] $OUT"

cd /data/yzy

if [ ! -d "$APEX4/.git" ]; then
  git clone https://github.com/APEX4-W4A4/APEX4-W4A4 "$APEX4"
else
  git -C "$APEX4" pull --ff-only || true
fi

cd "$APEX4"
git log -1 --oneline || true

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
export QFACTORY_CACHE_DIR="$EXP/reports/qfactory_cache_v24"
export PYTHONPATH="$ROMEO:$ROOT:$EXP/tools:$APEX4/kernels:${PYTHONPATH:-}"

mkdir -p "$QFACTORY_CACHE_DIR"

echo
echo "===== python / torch ====="
"$PYBIN" - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("device:", torch.cuda.get_device_name(0))
PY

echo
echo "===== apex4 repo inventory ====="
find "$APEX4/kernels" -maxdepth 3 -type f | sort | tee "$OUT/apex4_kernel_files.txt"

echo
echo "===== build apex4 kernels ====="
cd "$APEX4/kernels"
"$PYBIN" setup.py build_ext --inplace

echo
echo "===== import smoke ====="
"$PYBIN" - <<'PY'
import sys, os, torch
import csrc
print("csrc ok")
print([x for x in dir(csrc) if "w4a4" in x or "quant" in x or "compress" in x])
PY

echo
echo "===== run qwen-shape probe ====="
cd "$ROOT"
"$PYBIN" -u "$EXP/tools/bench_apex4_kernel_shapes_v24.py" \
  --apex4 "$APEX4" \
  --out_dir "$OUT" \
  --warmup 5 \
  --iters 50

echo "[DONE] $(date -Is)"
