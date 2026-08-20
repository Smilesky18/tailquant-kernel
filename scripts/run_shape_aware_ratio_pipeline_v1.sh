#!/usr/bin/env bash
set -u

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
TOOLS=$EXP/tools
OUT=${OUT:-$EXP/results/shape_aware_ratio_search_qwen3_8b_v1}
LOGDIR=$OUT/logs
STATUS=$EXP/status/shape_aware_ratio_search_qwen3_8b_v1.status

PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}
PPL_PYBIN=${PPL_PYBIN:-/data/yzy/miniconda3/envs/quarot-clean/bin/python}
[ -x "$PPL_PYBIN" ] || PPL_PYBIN="$PYBIN"

MODEL=${MODEL:-Qwen/Qwen3-8B}
ROTATION_CONFIG=${ROTATION_CONFIG:-/data/yzy/quarot/qwen3-8B_layer_all.csv}
BASE_POLICY=${BASE_POLICY:-$ROOT/experiments/kernel_quant/qwen_per_linear_diff_calibration_v6_rotate/lambda_0p08/policy.json}

V53_SCRIPT=${V53_SCRIPT:-$TOOLS/bench_prefill_bf16_romeoquarotdense_split_total_v53.py}
PPL_SCRIPT=${PPL_SCRIPT:-$ROOT/kernel_quant/scripts/eval_policy_v6_weightmode_v1.py}

SEQ_LEN=${SEQ_LEN:-128}
BATCHES=${BATCHES:-16,64,256}
LAYERS=${LAYERS:-all}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}

PROFILE_RATIOS=${PROFILE_RATIOS:-0.00125,0.0025,0.005,0.01,0.02,0.04,0.08}
SEARCH_RATIO_GRID=${SEARCH_RATIO_GRID:-0,0.00125,0.0025,0.005,0.01,0.02,0.04,0.08}
SEARCH_LAMBDAS=${SEARCH_LAMBDAS:-0,0.03,0.1,0.3,1,3,10,30}
BATCH_WEIGHTS=${BATCH_WEIGHTS:-16:1,64:1,256:1}

BODY_PERCENTILE=${BODY_PERCENTILE:-99.75}
TAIL_PERCENTILE=${TAIL_PERCENTILE:-100.0}
WEIGHT_PERCENTILE=${WEIGHT_PERCENTILE:-99.75}
PPL_WEIGHT_METHOD=${PPL_WEIGHT_METHOD:-rtn}

RUN_SHAPE_PROFILE=${RUN_SHAPE_PROFILE:-1}
RUN_POLICY_SEARCH=${RUN_POLICY_SEARCH:-1}
RUN_PPL=${RUN_PPL:-1}
RUN_FINAL_LATENCY=${RUN_FINAL_LATENCY:-1}
MAX_PARALLEL=${MAX_PARALLEL:-8}

PPL_N_WINDOWS=${PPL_N_WINDOWS:-128}
PPL_SEQLEN=${PPL_SEQLEN:-2048}
PPL_GPTQ_SEQLEN=${PPL_GPTQ_SEQLEN:-2048}
PPL_NSAMPLES=${PPL_NSAMPLES:-128}

mkdir -p "$OUT" "$LOGDIR" "$OUT/policies" "$OUT/shape_profile_policies" "$OUT/shape_profile_runs" "$OUT/shape_aware_policies" "$OUT/ppl_runs" "$OUT/final_latency_runs" "$(dirname "$STATUS")"

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
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$QFACTORY_CACHE_DIR"

echo "running start=$(date -Is)" > "$STATUS"
trap 'rc=$?; if [ "$rc" -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; echo "[END] $(date -Is) rc=$rc"; exit "$rc"' EXIT

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[OUT] $OUT"
echo "[PYBIN] $PYBIN"
echo "[PPL_PYBIN] $PPL_PYBIN"
echo "[MODEL] $MODEL"
echo "[ROTATION_CONFIG] $ROTATION_CONFIG"
echo "[BASE_POLICY] $BASE_POLICY"
echo "[V53_SCRIPT] $V53_SCRIPT"
echo "[PPL_SCRIPT] $PPL_SCRIPT"
echo "[PROFILE_RATIOS] $PROFILE_RATIOS"
echo "[SEARCH_RATIO_GRID] $SEARCH_RATIO_GRID"
echo "[SEARCH_LAMBDAS] $SEARCH_LAMBDAS"
echo "[PPL_WEIGHT_METHOD] $PPL_WEIGHT_METHOD"
echo "[RUN_SHAPE_PROFILE] $RUN_SHAPE_PROFILE [RUN_POLICY_SEARCH] $RUN_POLICY_SEARCH [RUN_PPL] $RUN_PPL [RUN_FINAL_LATENCY] $RUN_FINAL_LATENCY [MAX_PARALLEL] $MAX_PARALLEL"

for f in "$PYBIN" "$BASE_POLICY" "$ROTATION_CONFIG" "$V53_SCRIPT"; do
  if [ ! -e "$f" ]; then
    echo "[ERROR] missing: $f"
    exit 2
  fi
done

"$PYBIN" -m py_compile \
  "$TOOLS/extract_linear_shapes_from_policy_v1.py" \
  "$TOOLS/make_shape_isolation_policies_v1.py" \
  "$TOOLS/collect_shape_latency_table_v1.py" \
  "$TOOLS/make_shape_aware_policies_v1.py" \
  "$TOOLS/collect_policy_tradeoff_v1.py" \
  "$V53_SCRIPT"

echo
echo "========== stage0: extract shapes =========="
MODULE_SHAPES=$OUT/module_shapes.csv
UNIQUE_SHAPES=$OUT/unique_shapes.csv
"$PYBIN" "$TOOLS/extract_linear_shapes_from_policy_v1.py" \
  --policy "$BASE_POLICY" \
  --out_csv "$MODULE_SHAPES" \
  --out_shape_csv "$UNIQUE_SHAPES"

echo
echo "========== stage1: make shape-isolation policies =========="
SHAPE_POLICY_MANIFEST=$OUT/shape_profile_policy_manifest.tsv
"$PYBIN" "$TOOLS/make_shape_isolation_policies_v1.py" \
  --base_policy "$BASE_POLICY" \
  --shapes_csv "$UNIQUE_SHAPES" \
  --out_dir "$OUT/shape_profile_policies" \
  --manifest "$SHAPE_POLICY_MANIFEST" \
  --ratios "$PROFILE_RATIOS" \
  --body_percentile "$BODY_PERCENTILE" \
  --tail_percentile "$TAIL_PERCENTILE" \
  --weight_percentile "$WEIGHT_PERCENTILE"

run_v53_policy() {
  local gpu="$1"
  local tag="$2"
  local policy="$3"
  local out_dir="$4"
  local log="$5"
  local label="${6:-qwen3_8b_${tag}_shapeaware_v1}"

  mkdir -p "$out_dir"
  echo "[V53 START] tag=$tag gpu=$gpu time=$(date -Is)" | tee "$log"

  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" "$V53_SCRIPT" \
    --model "$MODEL" \
    --label "$label" \
    --policy "$policy" \
    --rotation_config "$ROTATION_CONFIG" \
    --seq_len "$SEQ_LEN" \
    --batches "$BATCHES" \
    --layers "$LAYERS" \
    --variants split \
    --device cuda:0 \
    --warmup "$WARMUP" \
    --iters "$ITERS" \
    --out_dir "$out_dir" \
    --local_files_only \
    >> "$log" 2>&1

  local rc=$?
  echo "[V53 END] tag=$tag rc=$rc time=$(date -Is)" >> "$log"
  return "$rc"
}

if [ "$RUN_SHAPE_PROFILE" = "1" ]; then
  echo
  echo "========== stage2: real-kernel shape/ratio latency profiling START $(date -Is) =========="
  idx=0
  prof_rc=0
  pids=()
  while IFS=$'\t' read -r tag shape_id K N ratio policy target_module_count target_R_sum; do
    [ "$tag" = "tag" ] && continue
    gpu=$((idx % 8))
    run_v53_policy "$gpu" "$tag" "$policy" "$OUT/shape_profile_runs/$tag" "$LOGDIR/shape_profile_${tag}.log" "qwen3_8b_${tag}_shape_profile_v1" &
    pids+=($!)
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "$MAX_PARALLEL" ]; then
      for p in "${pids[@]}"; do
        wait "$p" || prof_rc=$?
      done
      pids=()
    fi
  done < "$SHAPE_POLICY_MANIFEST"

  for p in "${pids[@]}"; do
    wait "$p" || prof_rc=$?
  done
  if [ "$prof_rc" -ne 0 ]; then
    echo "[ERROR] shape profile failed rc=$prof_rc"
    exit "$prof_rc"
  fi
  echo "========== stage2: real-kernel shape/ratio latency profiling END $(date -Is) =========="
fi

echo
echo "========== stage3: collect shape latency table =========="
SHAPE_LATENCY_CSV=$OUT/shape_latency_cost_table.csv
"$PYBIN" "$TOOLS/collect_shape_latency_table_v1.py" \
  --manifest "$SHAPE_POLICY_MANIFEST" \
  --run_root "$OUT/shape_profile_runs" \
  --out_csv "$SHAPE_LATENCY_CSV" \
  --out_summary_json "$OUT/shape_latency_cost_table_summary.json"

if [ "$RUN_POLICY_SEARCH" = "1" ]; then
  echo
  echo "========== stage4: shape-aware policy search =========="
  POLICY_SUMMARY=$OUT/shape_aware_policy_summary.tsv
  "$PYBIN" "$TOOLS/make_shape_aware_policies_v1.py" \
    --base_policy "$BASE_POLICY" \
    --shape_latency_csv "$SHAPE_LATENCY_CSV" \
    --out_dir "$OUT/shape_aware_policies" \
    --summary_tsv "$POLICY_SUMMARY" \
    --lambdas "$SEARCH_LAMBDAS" \
    --ratio_grid "$SEARCH_RATIO_GRID" \
    --batch_weights "$BATCH_WEIGHTS" \
    --body_percentile "$BODY_PERCENTILE" \
    --tail_percentile "$TAIL_PERCENTILE" \
    --weight_percentile "$WEIGHT_PERCENTILE"
fi

run_ppl_policy() {
  local gpu="$1"
  local tag="$2"
  local policy="$3"
  local out_dir="$4"
  local log="$5"

  mkdir -p "$out_dir"
  echo "[PPL START] tag=$tag gpu=$gpu time=$(date -Is)" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PPL_PYBIN" "$PPL_SCRIPT" \
    --policy "$policy" \
    --out_dir "$out_dir" \
    --model "$MODEL" \
    --dataset wikitext2 \
    --cal_dataset wikitext2 \
    --window_start 0 \
    --n_windows "$PPL_N_WINDOWS" \
    --seqlen "$PPL_SEQLEN" \
    --gptq_seqlen "$PPL_GPTQ_SEQLEN" \
    --nsamples "$PPL_NSAMPLES" \
    --seed 0 \
    --rotation_config "$ROTATION_CONFIG" \
    --percdamp 0.01 \
    --weight_method "$PPL_WEIGHT_METHOD" \
    --use_projected_ratio \
    --eps 1e-8 \
    >> "$log" 2>&1
  local rc=$?
  echo "[PPL END] tag=$tag rc=$rc time=$(date -Is)" >> "$log"
  return "$rc"
}

if [ "$RUN_PPL" = "1" ]; then
  echo
  echo "========== stage5: PPL eval for shape-aware policies START $(date -Is) =========="
  if [ ! -e "$PPL_SCRIPT" ]; then
    echo "[ERROR] missing PPL_SCRIPT: $PPL_SCRIPT"
    exit 20
  fi
  "$PPL_PYBIN" -m py_compile "$PPL_SCRIPT"

  idx=0
  ppl_rc=0
  pids=()
  while IFS=$'\t' read -r tag lambda policy module_count nonzero_modules mean_ratio sum_R sum_acc_proxy sum_cost_proxy ratio_hist; do
    [ "$tag" = "tag" ] && continue
    gpu=$((idx % 8))
    run_ppl_policy "$gpu" "$tag" "$policy" "$OUT/ppl_runs/$tag" "$LOGDIR/ppl_${tag}.log" &
    pids+=($!)
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "$MAX_PARALLEL" ]; then
      for p in "${pids[@]}"; do
        wait "$p" || ppl_rc=$?
      done
      pids=()
    fi
  done < "$OUT/shape_aware_policy_summary.tsv"

  for p in "${pids[@]}"; do
    wait "$p" || ppl_rc=$?
  done
  if [ "$ppl_rc" -ne 0 ]; then
    echo "[ERROR] PPL eval failed rc=$ppl_rc"
    exit "$ppl_rc"
  fi
  echo "========== stage5: PPL eval for shape-aware policies END $(date -Is) =========="
fi

if [ "$RUN_FINAL_LATENCY" = "1" ]; then
  echo
  echo "========== stage6: final real-kernel latency for shape-aware policies START $(date -Is) =========="
  idx=0
  lat_rc=0
  pids=()
  while IFS=$'\t' read -r tag lambda policy module_count nonzero_modules mean_ratio sum_R sum_acc_proxy sum_cost_proxy ratio_hist; do
    [ "$tag" = "tag" ] && continue
    gpu=$((idx % 8))
    run_v53_policy "$gpu" "$tag" "$policy" "$OUT/final_latency_runs/$tag" "$LOGDIR/final_latency_${tag}.log" "qwen3_8b_${tag}_shapeaware_policy_v1" &
    pids+=($!)
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "$MAX_PARALLEL" ]; then
      for p in "${pids[@]}"; do
        wait "$p" || lat_rc=$?
      done
      pids=()
    fi
  done < "$OUT/shape_aware_policy_summary.tsv"

  for p in "${pids[@]}"; do
    wait "$p" || lat_rc=$?
  done
  if [ "$lat_rc" -ne 0 ]; then
    echo "[ERROR] final latency failed rc=$lat_rc"
    exit "$lat_rc"
  fi
  echo "========== stage6: final real-kernel latency for shape-aware policies END $(date -Is) =========="
fi

echo
echo "========== stage7: collect final tradeoff =========="
if [ "$RUN_POLICY_SEARCH" = "1" ] && [ "$RUN_PPL" = "1" ] && [ "$RUN_FINAL_LATENCY" = "1" ]; then
  "$PYBIN" "$TOOLS/collect_policy_tradeoff_v1.py" \
    --policy_summary "$OUT/shape_aware_policy_summary.tsv" \
    --ppl_log_dir "$LOGDIR" \
    --ppl_run_dir "$OUT/ppl_runs" \
    --latency_run_dir "$OUT/final_latency_runs" \
    --out_csv "$OUT/shape_aware_policy_tradeoff.csv"
else
  echo "[SKIP] collect final tradeoff requires RUN_POLICY_SEARCH=1 RUN_PPL=1 RUN_FINAL_LATENCY=1"
fi

echo
echo "========== compact outputs =========="
echo "[MODULE_SHAPES] $MODULE_SHAPES"
echo "[UNIQUE_SHAPES] $UNIQUE_SHAPES"
echo "[SHAPE_LATENCY_CSV] $SHAPE_LATENCY_CSV"
echo "[POLICY_SUMMARY] $OUT/shape_aware_policy_summary.tsv"
echo "[TRADEOFF] $OUT/shape_aware_policy_tradeoff.csv"

"$PYBIN" - <<PY_SUMMARY
import csv
from pathlib import Path
p = Path("$OUT/shape_aware_policy_tradeoff.csv")
if p.exists():
    rows = list(csv.DictReader(open(p)))
    def f(x, default=1e18):
        try: return float(x)
        except Exception: return default
    for r in sorted(rows, key=lambda x: (f(x.get("ppl")), f(x.get("latency_sum_b64_ms")))):
        print("[TRADEOFF]",
              r.get("tag"),
              "mean_ratio=", r.get("mean_ratio"),
              "nonzero=", r.get("nonzero_modules"),
              "sum_R=", r.get("sum_R"),
              "ppl=", r.get("ppl"),
              "b16=", r.get("latency_sum_b16_ms"),
              "b64=", r.get("latency_sum_b64_ms"),
              "b256=", r.get("latency_sum_b256_ms"))
PY_SUMMARY

echo "[DONE] $(date -Is)"
