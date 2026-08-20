#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/quarot-clean/bin/python}
OUT=$EXP/reports/int4_kernel_audit_v13

mkdir -p "$OUT"

cd "$ROOT"
export PYTHONPATH="$ROOT:$EXP/tools:${PYTHONPATH:-}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[PYBIN] $PYBIN"
echo "[OUT] $OUT"
echo "[TORCH_CUDA_ARCH_LIST] ${TORCH_CUDA_ARCH_LIST:-}"

"$PYBIN" - <<'PY' > "$OUT/ext_info.txt" 2>&1
import os
from pathlib import Path
import torch

import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

device = torch.device("cuda:0")
torch.cuda.set_device(device)

if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    major, minor = torch.cuda.get_device_capability(device)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

BASE = B.BASE
cutlass = BASE.find_cutlass_path("/data/yzy/quarot-gpt-2/third_party/cutlass")
ext = BASE.load_ext(cutlass, verbose=False)

print("[DEVICE]", torch.cuda.get_device_name(device))
print("[CAPABILITY]", torch.cuda.get_device_capability(device))
print("[CUTLASS]", cutlass)
print("[EXT]", ext)
print("[EXT_FILE]", ext.__file__)
print("[EXT_DIR]", Path(ext.__file__).parent)
print("[TORCH_CUDA_ARCH_LIST]", os.environ.get("TORCH_CUDA_ARCH_LIST"))
PY

cat "$OUT/ext_info.txt"

SO=$(grep '^\[EXT_FILE\]' "$OUT/ext_info.txt" | sed 's/^\[EXT_FILE\] //')
EXT_DIR=$(grep '^\[EXT_DIR\]' "$OUT/ext_info.txt" | sed 's/^\[EXT_DIR\] //')

echo "[SO] $SO" | tee "$OUT/summary.txt"
echo "[EXT_DIR] $EXT_DIR" | tee -a "$OUT/summary.txt"

echo
echo "===== build.ninja / compile flags ====="
if [ -f "$EXT_DIR/build.ninja" ]; then
  cp "$EXT_DIR/build.ninja" "$OUT/build.ninja"
  grep -nE "arch|sm_|compute_|gencode|nvcc|cxx|TORCH_CUDA_ARCH_LIST|cutlass" "$EXT_DIR/build.ninja" \
    | tee "$OUT/build_ninja_grep.txt" || true
else
  echo "[WARN] build.ninja not found in $EXT_DIR" | tee "$OUT/build_ninja_grep.txt"
fi

echo
echo "===== extension source files ====="
find "$EXT_DIR" -maxdepth 3 -type f \
  \( -name "*.cu" -o -name "*.cpp" -o -name "*.cuh" -o -name "*.h" -o -name "*.hpp" -o -name "build.ninja" \) \
  | sort | tee "$OUT/source_files.txt"

echo
echo "===== source grep in extension dir and kernel_quant ====="
grep -RInE "cutlass_s4_gemm|pack_a_full_s4|scale_i32_to_fp16|OpClassTensorOp|OpClassSimt|TensorOp|Simt|int4b|int4_t|s4|mma|Mma|GemmShape|ThreadblockShape|WarpShape|InstructionShape|Sm[0-9]+|arch::" \
  "$EXT_DIR" "$ROOT/kernel_quant" \
  > "$OUT/source_grep.txt" 2>&1 || true

head -n 400 "$OUT/source_grep.txt"

echo
echo "===== disassemble / sass grep ====="
if command -v cuobjdump >/dev/null 2>&1; then
  echo "[DUMP_TOOL] cuobjdump" | tee "$OUT/dump_tool.txt"
  cuobjdump --dump-sass "$SO" > "$OUT/sass.txt" 2>&1 || true
elif command -v nvdisasm >/dev/null 2>&1; then
  echo "[DUMP_TOOL] nvdisasm" | tee "$OUT/dump_tool.txt"
  nvdisasm "$SO" > "$OUT/sass.txt" 2>&1 || true
else
  echo "[DUMP_TOOL] strings fallback" | tee "$OUT/dump_tool.txt"
  strings "$SO" > "$OUT/sass.txt" 2>&1 || true
fi

grep -Ein "mma|wgmma|tcgen|hmma|imma|dp4a|ldmatrix|lop3|cutlass|sm_120|sm_90|sm_89" \
  "$OUT/sass.txt" > "$OUT/instruction_hits.txt" || true

echo "[INSTRUCTION_HIT_COUNT]" $(wc -l < "$OUT/instruction_hits.txt") | tee -a "$OUT/summary.txt"
head -n 240 "$OUT/instruction_hits.txt"

echo
echo "===== symbol grep ====="
if command -v nm >/dev/null 2>&1; then
  nm -D "$SO" > "$OUT/nm_symbols.txt" 2>&1 || true
  grep -Ein "cutlass_s4_gemm|pack_a_full_s4|scale_i32_to_fp16|gemm|mma|s4" "$OUT/nm_symbols.txt" \
    > "$OUT/nm_symbol_hits.txt" || true
  head -n 200 "$OUT/nm_symbol_hits.txt"
fi

echo
echo "[DONE] $(date -Is)"
