#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=$ROOT/experiments/kernel_quant/layer_latency_split_v1
TOOLS=$EXP/tools
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}

SCRIPT=$TOOLS/bench_prefill_bf16_romeoquarotdense_split_total_v53.py
POLICY_DIR=$ROOT/experiments/kernel_quant/qwen3_8b_ratio0_percentile_gptq_sweep_v2_parallel8/policies
OUT=${OUT:-$EXP/results/prefill_split_total_v56_qwen3_8b_v2_policies}
LOGDIR=$OUT/logs
STATUS=$OUT/status/qwen3_8b_v2_ratio_split_latency_v56.status
SUMMARY=$OUT/qwen3_8b_v2_ratio_split_latency_v56_summary.csv

MODEL=${MODEL:-Qwen/Qwen3-8B}
ROTATION_CONFIG=${ROTATION_CONFIG:-/data/yzy/quarot/qwen3-8B_layer_all.csv}
SEQ_LEN=${SEQ_LEN:-128}
BATCHES=${BATCHES:-16,64,256}
LAYERS=${LAYERS:-all}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}

mkdir -p "$OUT" "$LOGDIR" "$OUT/status" "$OUT/tmp"

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDACXX=${CUDACXX:-/usr/local/cuda/bin/nvcc}
export PATH="$(dirname "$PYBIN"):$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export QFACTORY_ARCH=${QFACTORY_ARCH:-120}
export NO_USE_FASTER_HADAMARD_TRANSFORM=${NO_USE_FASTER_HADAMARD_TRANSFORM:-1}
export HF_HOME=${HF_HOME:-/data/yzy/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export PYTHONPATH=/data/yzy/RoMeo:$ROOT:$TOOLS${PYTHONPATH:+:$PYTHONPATH}
export TMPDIR=$OUT/tmp
export TORCH_EXTENSIONS_DIR=$OUT/tmp/torch_extensions
export QFACTORY_CACHE_DIR=$OUT/tmp/qfactory_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "running start=$(date -Is)" > "$STATUS"
trap 'rc=$?; if [ "$rc" -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit "$rc"' EXIT

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[OUT] $OUT"
echo "[PYBIN] $PYBIN"
echo "[SCRIPT] $SCRIPT"
echo "[MODEL] $MODEL"
echo "[ROTATION_CONFIG] $ROTATION_CONFIG"
echo "[SEQ_LEN] $SEQ_LEN [BATCHES] $BATCHES [LAYERS] $LAYERS [WARMUP] $WARMUP [ITERS] $ITERS"
echo "[NOTE] v56 runs v53 total-only benchmark with --variants split. It does not run GPTQ; it only swaps the Split policy JSON."

for f in "$PYBIN" "$SCRIPT" "$ROTATION_CONFIG"; do
  if [ ! -e "$f" ]; then
    echo "[ERROR] missing: $f"
    exit 2
  fi
done

"$PYBIN" -m py_compile "$SCRIPT"
df -h "$OUT" || true

cat > "$OUT/policies.tsv" <<EOF_POLICIES
variant	gpu	policy
baseline_like	0	$POLICY_DIR/baseline_like.json
low_cost	1	$POLICY_DIR/low_cost.json
ultra_low_cost	2	$POLICY_DIR/ultra_low_cost.json
near_zero_cost	3	$POLICY_DIR/near_zero_cost.json
zero_single_p995	4	$POLICY_DIR/zero_single_p995.json
zero_single_p9975	5	$POLICY_DIR/zero_single_p9975.json
zero_single_p999	6	$POLICY_DIR/zero_single_p999.json
hard_cap_2p	7	$POLICY_DIR/hard_cap_2p.json
EOF_POLICIES

while IFS=$'\t' read -r variant gpu policy; do
  [ "$variant" = "variant" ] && continue
  if [ ! -e "$policy" ]; then
    echo "[ERROR] missing policy for $variant: $policy"
    exit 3
  fi
done < "$OUT/policies.tsv"

run_one() {
  local variant="$1" gpu="$2" policy="$3"
  local variant_out="$OUT/$variant"
  local log="$LOGDIR/${variant}.log"
  mkdir -p "$variant_out"
  echo "[JOB START] variant=$variant gpu=$gpu policy=$policy time=$(date -Is)" | tee "$LOGDIR/${variant}.status"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" "$SCRIPT" \
    --model "$MODEL" \
    --label "qwen3_8b_${variant}_split_v56" \
    --policy "$policy" \
    --rotation_config "$ROTATION_CONFIG" \
    --seq_len "$SEQ_LEN" \
    --batches "$BATCHES" \
    --layers "$LAYERS" \
    --variants split \
    --device cuda:0 \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --out_dir "$variant_out" \
    --local_files_only \
    > "$log" 2>&1
  local rc=$?
  echo "[JOB END] variant=$variant rc=$rc time=$(date -Is)" | tee -a "$LOGDIR/${variant}.status"
  return "$rc"
}

echo "========== parallel split latency jobs START $(date -Is) =========="
pids=()
while IFS=$'\t' read -r variant gpu policy; do
  [ "$variant" = "variant" ] && continue
  run_one "$variant" "$gpu" "$policy" &
  pid="$!"
  pids+=("$pid")
  echo "[JOB PID] variant=$variant gpu=$gpu pid=$pid"
done < "$OUT/policies.tsv"

overall_rc=0
for p in "${pids[@]}"; do
  wait "$p" || overall_rc=$?
done
echo "========== parallel split latency jobs END $(date -Is) =========="

"$PYBIN" - <<'PY_SUMMARY' "$OUT" "$SUMMARY"
import csv
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = Path(sys.argv[2])
rows = []
for meta_path in sorted(out.glob("*/qwen3_8b_*_split_v56_prefill_layer_total_meta_v53.json")):
    meta = json.load(open(meta_path))
    variant = meta_path.parent.name
    args = meta.get("args", {})
    rows.append({
        "variant": variant,
        "policy": args.get("policy", ""),
        "num_rows": meta.get("num_rows", ""),
        "errors": len(meta.get("errors", [])),
        "sum_split_ms": meta.get("sum_split_ms", ""),
        "layer_csv": meta.get("layer_csv", ""),
        "meta": str(meta_path),
    })
fields = ["variant", "policy", "num_rows", "errors", "sum_split_ms", "layer_csv", "meta"]
summary.parent.mkdir(parents=True, exist_ok=True)
with open(summary, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"[SUMMARY_CSV] {summary}")
for row in rows:
    print("[SUMMARY_ROW] " + json.dumps(row, ensure_ascii=False))
PY_SUMMARY

grep -RInE "Traceback|RuntimeError|CUDA error|out of memory|OOM|AssertionError|\[ERROR\]" "$LOGDIR" | tail -n 300 || true

if [ "$overall_rc" -ne 0 ]; then
  echo "[ERROR] one or more variants failed rc=$overall_rc"
  exit "$overall_rc"
fi
echo "[END] $(date -Is) rc=0"
