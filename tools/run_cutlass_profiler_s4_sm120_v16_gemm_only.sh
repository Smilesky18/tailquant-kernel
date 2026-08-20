#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
CUTLASS=${CUTLASS:-$ROOT/third_party/cutlass}

OUT=$EXP/results/cutlass_profiler_s4_sm120_v16
BUILD=$EXP/reports/cutlass_profiler_sm120_build_v16
STATUS=$EXP/status/cutlass_profiler_s4_sm120_v16.status

mkdir -p "$OUT" "$BUILD" "$EXP/logs" "$EXP/status"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[CUTLASS] $CUTLASS"
echo "[BUILD] $BUILD"
echo "[OUT] $OUT"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDACXX=${CUDACXX:-$CUDA_HOME/bin/nvcc}
export CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER:-$CUDA_HOME/bin/nvcc}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

rm -rf "$BUILD"
mkdir -p "$BUILD"

cmake -S "$CUTLASS" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CMAKE_CUDA_COMPILER" \
  -DCUTLASS_NVCC_ARCHS=120 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_ENABLE_TOOLS=ON \
  -DCUTLASS_ENABLE_CUBLAS=OFF \
  -DCUTLASS_ENABLE_CUDNN=OFF \
  -DCUTLASS_LIBRARY_KERNELS="*s4*gemm*,*i4*gemm*,*16832gemm*" \
  -DCUTLASS_LIBRARY_OPERATIONS=gemm

cmake --build "$BUILD" --target cutlass_profiler -j8

PROFILER=$BUILD/tools/profiler/cutlass_profiler
echo "[PROFILER] $PROFILER"

"$PROFILER" --help > "$OUT/help.txt" 2>&1 || true

for shape in \
  "2048 4096 4096" \
  "2048 4096 12288" \
  "2048 12288 4096" \
  "8192 4096 4096" \
  "8192 4096 12288" \
  "8192 12288 4096"
do
  set -- $shape
  M=$1
  K=$2
  N=$3

  echo
  echo "[RUN] M=$M N=$N K=$K"

  "$PROFILER" \
    --operation=Gemm \
    --kernels="*" \
    --m="$M" \
    --n="$N" \
    --k="$K" \
    --A=s4:row \
    --B=s4:column \
    --C=s32:row \
    --D=s32:row \
    --accum=s32 \
    --alpha=1 \
    --beta=0 \
    --split_k_slices=1 \
    --warmup-iterations=10 \
    --profiling-iterations=30 \
    --output="$OUT/s4s4s32_M${M}_N${N}_K${K}.csv" \
    > "$OUT/s4s4s32_M${M}_N${N}_K${K}.txt" 2>&1 || true

  grep -E "Operation:|Disposition:|Status:|Runtime:|Math:|No operations|No kernels|Error|Failed|Passed" \
    "$OUT/s4s4s32_M${M}_N${N}_K${K}.txt" | head -n 120 || true
done

grep -RHE "Operation:|Disposition:|Status:|Runtime:|Math:|No operations|No kernels|Error|Failed|Passed" "$OUT"/*.txt \
  > "$OUT/result_grep.txt" || true

find "$OUT" -maxdepth 1 -name "*.csv" -type f -size +0c -print | sort > "$OUT/csv_files.txt"

echo "[DONE] $(date -Is)"
