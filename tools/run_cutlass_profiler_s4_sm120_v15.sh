#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
CUTLASS=${CUTLASS:-$ROOT/third_party/cutlass}

OUT=$EXP/results/cutlass_profiler_s4_sm120_v15
BUILD=$EXP/reports/cutlass_profiler_sm120_build_v15
STATUS=$EXP/status/cutlass_profiler_s4_sm120_v15.status

mkdir -p "$OUT" "$BUILD" "$EXP/logs" "$EXP/status" "$EXP/results" "$EXP/reports"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[CUTLASS] $CUTLASS"
echo "[OUT] $OUT"
echo "[BUILD] $BUILD"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}

if [ ! -d "$CUTLASS" ]; then
  echo "[ERROR] CUTLASS dir not found: $CUTLASS"
  exit 1
fi

echo
echo "===== build cutlass_profiler ====="

cmake -S "$CUTLASS" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_NVCC_ARCHS=120 \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=OFF \
  -DCUTLASS_ENABLE_PROFILER=ON \
  -DCUTLASS_ENABLE_TOOLS=ON

cmake --build "$BUILD" --target cutlass_profiler -j"$(nproc)"

PROFILER=$BUILD/tools/profiler/cutlass_profiler

if [ ! -x "$PROFILER" ]; then
  echo "[ERROR] profiler not found: $PROFILER"
  find "$BUILD" -name "cutlass_profiler" -type f -print
  exit 1
fi

echo "[PROFILER] $PROFILER"

"$PROFILER" --help > "$OUT/help.txt" 2>&1 || true

echo
echo "===== profiler version / help head ====="
head -n 80 "$OUT/help.txt" || true

cat > "$OUT/shapes.tsv" <<'EOF'
204840964096
2048409612288
2048122884096
819240964096
8192409612288
8192122884096
EOF

cat > "$OUT/attempts.tsv" <<'EOF'
labelABCDaccumkernels
s4row_s4col_s32row_accum_s32s4:rows4:columns32:rows32:rows32*s4*
s4row_s4col_s32col_accum_s32s4:rows4:columns32:columns32:columns32*s4*
s4col_s4row_s32col_accum_s32s4:columns4:rows32:columns32:columns32*s4*
s4row_s4col_f16row_accum_s32s4:rows4:columnf16:rowf16:rows32*s4*
s4row_s4col_s32row_accum_s32_nofilters4:rows4:columns32:rows32:rows32*
EOF

run_one() {
  local label="$1"
  local A="$2"
  local B="$3"
  local C="$4"
  local D="$5"
  local ACC="$6"
  local KERNELS="$7"
  local M="$8"
  local K="$9"
  local N="${10}"

  local stem="${label}_M${M}_N${N}_K${K}"
  local txt="$OUT/${stem}.txt"
  local csv="$OUT/${stem}.csv"

  echo
  echo "[RUN] label=$label M=$M N=$N K=$K A=$A B=$B C=$C D=$D accum=$ACC kernels=$KERNELS"

  set +e
  "$PROFILER" \
    --operation=Gemm \
    --kernels="$KERNELS" \
    --m="$M" \
    --n="$N" \
    --k="$K" \
    --A="$A" \
    --B="$B" \
    --C="$C" \
    --D="$D" \
    --accum="$ACC" \
    --alpha=1 \
    --beta=0 \
    --split_k_slices=1 \
    --warmup-iterations=10 \
    --profiling-iterations=30 \
    --output="$csv" \
    > "$txt" 2>&1
  local rc=$?
  set -e

  echo "[RC] $rc label=$label M=$M N=$N K=$K"

  if [ $rc -ne 0 ]; then
    echo "[FAIL_HEAD]"
    head -n 60 "$txt" || true
    echo "[FAIL_TAIL]"
    tail -n 60 "$txt" || true
    return 0
  fi

  echo "[SUCCESS_SUMMARY]"
  grep -E "Operation:|Status:|Disposition:|Verification:|Runtime:|Math:|Bytes:|No operations|No kernels|Passed|Failed|Error" "$txt" | head -n 120 || true
}

echo
echo "===== run S4 profiler attempts ====="

tail -n +2 "$OUT/attempts.tsv" | while IFS=$'\t' read -r label A B C D accum kernels; do
  while IFS=$'\t' read -r M K N; do
    run_one "$label" "$A" "$B" "$C" "$D" "$accum" "$kernels" "$M" "$K" "$N"
  done < "$OUT/shapes.tsv"
done

echo
echo "===== collect result grep ====="

grep -RHE "Operation:|Runtime:|Math:|Disposition:|Verification:|No operations|No kernels|Error|Failed|Passed" "$OUT"/*.txt \
  > "$OUT/result_grep.txt" || true

head -n 300 "$OUT/result_grep.txt" || true

echo
echo "===== csv files ====="
find "$OUT" -maxdepth 1 -name "*.csv" -type f -size +0c -print | sort | tee "$OUT/csv_files.txt"

echo
echo "===== quick csv head ====="
while read -r f; do
  echo
  echo "### $f"
  head -n 5 "$f" || true
done < "$OUT/csv_files.txt" > "$OUT/csv_heads.txt"

cat "$OUT/csv_heads.txt" | head -n 300 || true

echo
echo "[DONE] $(date -Is)"
