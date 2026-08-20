#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
CUTLASS_PATH=${CUTLASS_PATH:-$ROOT/third_party/cutlass}

STATUS=$EXP/status/cutlass_sm120_w4a4_audit_v27.status
OUT=$EXP/results/cutlass_sm120_w4a4_audit_v27
BUILD=$EXP/reports/cutlass_profiler_sm120_v27_build

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports" "$OUT" "$BUILD"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[CUTLASS_PATH] $CUTLASS_PATH"
echo "[OUT] $OUT"
echo "[BUILD] $BUILD"
echo "[TASK] v27 CUTLASS SM120 S4/W4A4 audit and profiler search"

if [ ! -d "$CUTLASS_PATH" ]; then
  echo "[ERROR] CUTLASS_PATH not found: $CUTLASS_PATH"
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
export MAX_JOBS=16

cd "$CUTLASS_PATH"

echo
echo "===== CUTLASS version ====="
git rev-parse HEAD | tee "$OUT/cutlass_git_head.txt" || true
git log -1 --oneline | tee "$OUT/cutlass_git_log1.txt" || true

echo
echo "===== Search SM120 / Blackwell / S4 symbols ====="
{
  echo "### grep: SM120"
  grep -R "SM120\|sm120\|sm_120\|compute_120" -n include examples tools 2>/dev/null | head -n 300 || true
  echo
  echo "### grep: int4 / s4 / uint4"
  grep -R "int4_t\|uint4_t\|s4\|S4\|int4b\|uint4b" -n include examples tools 2>/dev/null | head -n 500 || true
  echo
  echo "### grep: tcgen05"
  grep -R "tcgen05\|TCGen05\|tmem" -n include examples tools 2>/dev/null | head -n 500 || true
} | tee "$OUT/cutlass_sm120_s4_symbol_audit_v27.txt"

echo
echo "===== Configure CUTLASS profiler for SM120 ====="
cmake -S "$CUTLASS_PATH" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=120 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_LIBRARY_KERNELS="*s4*gemm*;*int4*gemm*;*4bit*gemm*;*sm120*gemm*;*blackwell*gemm*"

echo
echo "===== Build CUTLASS profiler ====="
cmake --build "$BUILD" --target cutlass_profiler -j "${MAX_JOBS:-16}"

PROF="$BUILD/tools/profiler/cutlass_profiler"
if [ ! -x "$PROF" ]; then
  PROF="$BUILD/tools/profiler/cutlass_profiler/cutlass_profiler"
fi

echo "[PROFILER] $PROF"

echo
echo "===== List profiler kernels containing s4/int4/sm120/blackwell ====="
"$PROF" --operation=Gemm --kernels="*s4*" --help > "$OUT/cutlass_profiler_s4_help_v27.txt" 2>&1 || true
"$PROF" --operation=Gemm --kernels="*int4*" --help > "$OUT/cutlass_profiler_int4_help_v27.txt" 2>&1 || true
"$PROF" --operation=Gemm --kernels="*blackwell*" --help > "$OUT/cutlass_profiler_blackwell_help_v27.txt" 2>&1 || true
"$PROF" --operation=Gemm --kernels="*sm120*" --help > "$OUT/cutlass_profiler_sm120_help_v27.txt" 2>&1 || true

echo
echo "===== Run S4 GEMM profiler on Qwen3 shapes if kernels exist ====="
cat > "$OUT/qwen3_w4a4_shapes_v27.csv" <<'EOF'
name,m,n,k
q_or_o_b16,2048,4096,4096
k_or_v_b16,2048,1024,4096
gate_or_up_b16,2048,12288,4096
down_b16,2048,4096,12288
q_or_o_b64,8192,4096,4096
k_or_v_b64,8192,1024,4096
gate_or_up_b64,8192,12288,4096
down_b64,8192,4096,12288
EOF

RESULT_CSV="$OUT/cutlass_profiler_s4_qwen3_shapes_v27.csv"
echo "shape,kernel,time,exit_code" > "$RESULT_CSV"

while IFS=, read -r name m n k; do
  if [ "$name" = "name" ]; then
    continue
  fi

  echo
  echo "[PROFILE_SHAPE] $name m=$m n=$n k=$k"

  LOG="$OUT/profile_${name}_s4_v27.log"

  set +e
  "$PROF" \
    --operation=Gemm \
    --kernels="*s4*" \
    --m="$m" --n="$n" --k="$k" \
    --A=s4:row --B=s4:column --C=s32:row --D=s32:row \
    --accumulator-type=s32 \
    --op_class=tensorop \
    --verification-enabled=false \
    --providers=cutlass \
    --profiling-iterations=100 \
    --warmup-iterations=20 \
    --output="$OUT/profile_${name}_s4_table_v27.csv" \
    > "$LOG" 2>&1
  rc=$?
  set -e

  echo "$name,*s4*,,$rc" >> "$RESULT_CSV"
  echo "[PROFILE_RC] $name rc=$rc"
done < "$OUT/qwen3_w4a4_shapes_v27.csv"

echo
echo "===== Summaries ====="
grep -R "RuntimeError\|Error\|No kernels\|No operation\|Disposition\|Best" -n "$OUT" | tee "$OUT/cutlass_profiler_error_summary_v27.txt" || true

echo "[DONE] $(date -Is)"
