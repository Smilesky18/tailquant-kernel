#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
OUT=${OUT:-$EXP/results/qwen3_8b_cegrad_ratio_search_v1}
LOGDIR=${LOGDIR:-$OUT/logs}
STATUS=${STATUS:-$EXP/status/qwen3_8b_cegrad_ratio_search_v1.status}

PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}
PPL_PYBIN=${PPL_PYBIN:-/data/yzy/miniconda3/envs/quarot-clean/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-8B}
ROTATION_CONFIG=${ROTATION_CONFIG:-/data/yzy/quarot/qwen3-8B_layer_all.csv}
CE_SCRIPT=${CE_SCRIPT:-$EXP/tools/calibrate_shape_latency_cegrad_ratio_v1.py}
PPL_SCRIPT=${PPL_SCRIPT:-$ROOT/kernel_quant/scripts/eval_policy_v6_weightmode_v1.py}
V53_SCRIPT=${V53_SCRIPT:-$EXP/tools/bench_prefill_bf16_romeoquarotdense_split_total_v53.py}

LONG_LATENCY_TABLE=${LONG_LATENCY_TABLE:-$EXP/results/shape_aware_ratio_search_qwen3_8b_v1_rtn_fixed/shape_latency_cost_table.csv}
WIDE_LATENCY_TABLE=${WIDE_LATENCY_TABLE:-$OUT/shape_latency_cost_wide_per_module.csv}
LATENCY_BATCH=${LATENCY_BATCH:-b16}

LAMBDAS=${LAMBDAS:-0,0.01,0.05,0.1,0.2,0.5,1}
SEARCH_MAX_PARALLEL=${SEARCH_MAX_PARALLEL:-4}
PPL_MAX_PARALLEL=${PPL_MAX_PARALLEL:-4}
LAT_MAX_PARALLEL=${LAT_MAX_PARALLEL:-8}
RUN_SEARCH=${RUN_SEARCH:-1}
RUN_PPL=${RUN_PPL:-1}
RUN_FINAL_LATENCY=${RUN_FINAL_LATENCY:-1}

CE_NSAMPLES=${CE_NSAMPLES:-4}
CE_SEQLEN=${CE_SEQLEN:-128}
CE_CAPTURE_ROWS=${CE_CAPTURE_ROWS:-1024}
CE_OUT_CHANNELS=${CE_OUT_CHANNELS:-256}
CE_RATIO_CANDIDATES=${CE_RATIO_CANDIDATES:-0,0.00125,0.0025,0.005,0.01,0.02,0.04}
CE_PERCENTILE_CANDIDATES=${CE_PERCENTILE_CANDIDATES:-99.5,99.75,99.9}
CE_METRIC=${CE_METRIC:-ce_pos}

PPL_N_WINDOWS=${PPL_N_WINDOWS:-128}
PPL_SEQLEN=${PPL_SEQLEN:-2048}
PPL_GPTQ_SEQLEN=${PPL_GPTQ_SEQLEN:-2048}
PPL_NSAMPLES=${PPL_NSAMPLES:-128}
PPL_WEIGHT_METHOD=${PPL_WEIGHT_METHOD:-gptq}

SEQ_LEN=${SEQ_LEN:-128}
BATCHES=${BATCHES:-16,64,256}
LAYERS=${LAYERS:-all}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-20}

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDACXX=${CUDACXX:-$CUDA_HOME/bin/nvcc}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export QFACTORY_ARCH=${QFACTORY_ARCH:-sm120}
export NO_USE_FASTER_HADAMARD_TRANSFORM=${NO_USE_FASTER_HADAMARD_TRANSFORM:-1}
export HF_HOME=${HF_HOME:-/data/yzy/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TMPDIR=${TMPDIR:-$EXP/tmp}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$EXP/torch_extensions}
export PATH="$(dirname "$PYBIN"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH=${PYTHONPATH:-/data/yzy/RoMeo:$ROOT:$ROOT/kernel_quant/scripts:$EXP/tools}

mkdir -p "$OUT" "$LOGDIR" "$(dirname "$STATUS")" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
echo "running start=$(date -Is) out=$OUT" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[OUT] $OUT"
echo "[MODEL] $MODEL"
echo "[LATENCY_TABLE] $LONG_LATENCY_TABLE"
echo "[LAMBDAS] $LAMBDAS"
echo "[PPL_WEIGHT_METHOD] $PPL_WEIGHT_METHOD"
echo "[RUN_SEARCH] $RUN_SEARCH [RUN_PPL] $RUN_PPL [RUN_FINAL_LATENCY] $RUN_FINAL_LATENCY"

for f in "$PYBIN" "$PPL_PYBIN" "$CE_SCRIPT" "$PPL_SCRIPT" "$V53_SCRIPT" "$LONG_LATENCY_TABLE" "$ROTATION_CONFIG"; do
  if [ ! -e "$f" ]; then
    echo "[ERROR] missing required file: $f"
    echo "failed missing=$f end=$(date -Is)" > "$STATUS"
    exit 2
  fi
done

"$PYBIN" -m py_compile "$CE_SCRIPT" "$V53_SCRIPT"
"$PPL_PYBIN" -m py_compile "$PPL_SCRIPT"

echo
echo "========== stage0: convert latency table =========="
"$PYBIN" - "$LONG_LATENCY_TABLE" "$WIDE_LATENCY_TABLE" <<'PY'
import csv
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = {}
with src.open("r", newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row.get("status", "ok") != "ok":
            continue
        try:
            key = (int(float(row["K"])), int(float(row["N"])), float(row["ratio"]))
            batch = int(float(row["batch"]))
            cost = float(row["delta_ms_per_module"])
        except Exception:
            continue
        rows.setdefault(key, {})[f"b{batch}"] = cost

out_rows = []
for (k, n, ratio), values in sorted(rows.items()):
    out_rows.append({
        "K": k,
        "N": n,
        "ratio": ratio,
        "b16": values.get("b16", ""),
        "b64": values.get("b64", ""),
        "b256": values.get("b256", ""),
    })

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["K", "N", "ratio", "b16", "b64", "b256"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(out_rows)
print(f"[OK] rows={len(out_rows)} out={dst}")
PY

lambda_array=()
IFS=',' read -r -a lambda_array <<< "$LAMBDAS"

lambda_tag() {
  "$PYBIN" - "$1" <<'PY'
import sys
s = str(sys.argv[1])
s = s.replace("-", "m").replace(".", "p")
print("lambda_" + s)
PY
}

run_search() {
  local gpu="$1"
  local lam="$2"
  local tag
  tag=$(lambda_tag "$lam")
  local dir="$OUT/policies/$tag"
  local log="$LOGDIR/search_${tag}.log"
  mkdir -p "$dir"
  echo "[SEARCH START] tag=$tag lambda=$lam gpu=$gpu time=$(date -Is)" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" "$CE_SCRIPT" \
    --model "$MODEL" \
    --dataset wikitext2 \
    --mode split_v6 \
    --rotation_config "$ROTATION_CONFIG" \
    --out_dir "$dir" \
    --nsamples "$CE_NSAMPLES" \
    --seqlen "$CE_SEQLEN" \
    --capture_rows "$CE_CAPTURE_ROWS" \
    --out_channels "$CE_OUT_CHANNELS" \
    --ratio_candidates "$CE_RATIO_CANDIDATES" \
    --activation_percentile_candidates "$CE_PERCENTILE_CANDIDATES" \
    --weight_percentile_candidates "$CE_PERCENTILE_CANDIDATES" \
    --tail_percentile_candidates 100 \
    --anchor_activation_percentile 99.5 \
    --anchor_weight_percentile 99.5 \
    --ce_metric "$CE_METRIC" \
    --latency_table "$WIDE_LATENCY_TABLE" \
    --latency_batch "$LATENCY_BATCH" \
    --latency_lambda "$lam" \
    --missing_latency error \
    >> "$log" 2>&1
  local rc=$?
  echo "[SEARCH END] tag=$tag rc=$rc time=$(date -Is)" >> "$log"
  return "$rc"
}

if [ "$RUN_SEARCH" = "1" ]; then
  echo
  echo "========== stage1: CE search START $(date -Is) =========="
  idx=0
  search_rc=0
  pids=()
  for lam in "${lambda_array[@]}"; do
    gpu=$((idx % 8))
    run_search "$gpu" "$lam" &
    pids+=($!)
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "$SEARCH_MAX_PARALLEL" ]; then
      for p in "${pids[@]}"; do
        wait "$p" || search_rc=$?
      done
      pids=()
    fi
  done
  for p in "${pids[@]}"; do
    wait "$p" || search_rc=$?
  done
  if [ "$search_rc" -ne 0 ]; then
    echo "[ERROR] CE search failed rc=$search_rc"
    echo "failed stage=search rc=$search_rc end=$(date -Is)" > "$STATUS"
    exit "$search_rc"
  fi
  echo "========== stage1: CE search END $(date -Is) =========="
else
  echo
  echo "========== stage1: CE search SKIP $(date -Is) =========="
fi

echo
echo "========== stage2: policy summary =========="
POLICY_SUMMARY="$OUT/cegrad_policy_summary.tsv"
"$PYBIN" - "$OUT" "$POLICY_SUMMARY" "${lambda_array[@]}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = Path(sys.argv[2])
lambdas = sys.argv[3:]

def tag_for(lam: str) -> str:
    return "lambda_" + lam.replace("-", "m").replace(".", "p")

def hist(values):
    bins = {}
    for v in values:
        key = f"{v:.5g}"
        bins[key] = bins.get(key, 0) + 1
    return json.dumps(bins, sort_keys=True)

rows = []
for lam in lambdas:
    tag = tag_for(lam)
    policy = out / "policies" / tag / "policy.json"
    data = json.loads(policy.read_text(encoding="utf-8"))
    modules = list(data["modules"].values())
    ratios = [float(m.get("ratio_projected", m.get("ratio", 0.0))) for m in modules]
    mac = [float(m.get("mac_weight", 0.0)) for m in modules]
    sum_mac = sum(mac)
    mac_weighted = sum(a * b for a, b in zip(ratios, mac)) / sum_mac if sum_mac else 0.0
    rows.append({
        "tag": tag,
        "lambda": lam,
        "policy": str(policy),
        "module_count": len(modules),
        "nonzero_modules": sum(1 for r in ratios if r > 0),
        "mean_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "sum_R": sum(int(m.get("R", 0)) for m in modules),
        "sum_acc_proxy": sum(float(m.get("ce_gain", 0.0)) for m in modules),
        "sum_cost_proxy": sum(float(m.get("latency_delta", 0.0)) for m in modules),
        "ratio_hist": hist(ratios),
        "mac_weighted_ratio": mac_weighted,
    })

summary.parent.mkdir(parents=True, exist_ok=True)
fields = ["tag", "lambda", "policy", "module_count", "nonzero_modules", "mean_ratio", "sum_R", "sum_acc_proxy", "sum_cost_proxy", "ratio_hist", "mac_weighted_ratio"]
with summary.open("w", encoding="utf-8", newline="") as handle:
    handle.write("\t".join(fields) + "\n")
    for row in rows:
        handle.write("\t".join(str(row[f]) for f in fields) + "\n")
print(f"[OK] rows={len(rows)} out={summary}")
PY

run_ppl() {
  local gpu="$1"
  local tag="$2"
  local policy="$3"
  local out_dir="$OUT/ppl_runs/$tag"
  local log="$LOGDIR/ppl_${tag}.log"
  mkdir -p "$out_dir"
  echo "[PPL START] tag=$tag gpu=$gpu method=$PPL_WEIGHT_METHOD time=$(date -Is)" | tee "$log"
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
  echo "========== stage3: GPTQ PPL START $(date -Is) =========="
  idx=0
  ppl_rc=0
  pids=()
  while IFS=$'\t' read -r tag lam policy module_count nonzero_modules mean_ratio sum_R sum_acc_proxy sum_cost_proxy ratio_hist mac_weighted_ratio; do
    [ "$tag" = "tag" ] && continue
    gpu=$((idx % 8))
    run_ppl "$gpu" "$tag" "$policy" &
    pids+=($!)
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "$PPL_MAX_PARALLEL" ]; then
      for p in "${pids[@]}"; do
        wait "$p" || ppl_rc=$?
      done
      pids=()
    fi
  done < "$POLICY_SUMMARY"
  for p in "${pids[@]}"; do
    wait "$p" || ppl_rc=$?
  done
  if [ "$ppl_rc" -ne 0 ]; then
    echo "[ERROR] PPL failed rc=$ppl_rc"
    echo "failed stage=ppl rc=$ppl_rc end=$(date -Is)" > "$STATUS"
    exit "$ppl_rc"
  fi
  echo "========== stage3: GPTQ PPL END $(date -Is) =========="
else
  echo
  echo "========== stage3: GPTQ PPL SKIP $(date -Is) =========="
fi

run_latency() {
  local gpu="$1"
  local tag="$2"
  local policy="$3"
  local out_dir="$OUT/final_latency_runs/$tag"
  local log="$LOGDIR/final_latency_${tag}.log"
  mkdir -p "$out_dir"
  echo "[LAT START] tag=$tag gpu=$gpu time=$(date -Is)" | tee "$log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" "$V53_SCRIPT" \
    --model "$MODEL" \
    --label "qwen3_8b_${tag}_cegrad_ratio_v1" \
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
  echo "[LAT END] tag=$tag rc=$rc time=$(date -Is)" >> "$log"
  return "$rc"
}

if [ "$RUN_FINAL_LATENCY" = "1" ]; then
  echo
  echo "========== stage4: final latency START $(date -Is) =========="
  idx=0
  lat_rc=0
  pids=()
  while IFS=$'\t' read -r tag lam policy module_count nonzero_modules mean_ratio sum_R sum_acc_proxy sum_cost_proxy ratio_hist mac_weighted_ratio; do
    [ "$tag" = "tag" ] && continue
    gpu=$((idx % 8))
    run_latency "$gpu" "$tag" "$policy" &
    pids+=($!)
    idx=$((idx + 1))
    if [ "${#pids[@]}" -ge "$LAT_MAX_PARALLEL" ]; then
      for p in "${pids[@]}"; do
        wait "$p" || lat_rc=$?
      done
      pids=()
    fi
  done < "$POLICY_SUMMARY"
  for p in "${pids[@]}"; do
    wait "$p" || lat_rc=$?
  done
  if [ "$lat_rc" -ne 0 ]; then
    echo "[ERROR] latency failed rc=$lat_rc"
    echo "failed stage=latency rc=$lat_rc end=$(date -Is)" > "$STATUS"
    exit "$lat_rc"
  fi
  echo "========== stage4: final latency END $(date -Is) =========="
else
  echo
  echo "========== stage4: final latency SKIP $(date -Is) =========="
fi

echo
echo "========== stage5: collect tradeoff =========="
if [ "$RUN_PPL" = "1" ] && [ "$RUN_FINAL_LATENCY" = "1" ]; then
  "$PYBIN" "$EXP/tools/collect_policy_tradeoff_v1.py" \
    --policy_summary "$POLICY_SUMMARY" \
    --ppl_log_dir "$LOGDIR" \
    --ppl_run_dir "$OUT/ppl_runs" \
    --latency_run_dir "$OUT/final_latency_runs" \
    --out_csv "$OUT/cegrad_policy_tradeoff.csv"

  echo "[TRADEOFF] $OUT/cegrad_policy_tradeoff.csv"
  while IFS= read -r line; do
    echo "[TRADEOFF] $line"
  done < "$OUT/cegrad_policy_tradeoff.csv"
else
  echo "[SKIP] collect tradeoff requires RUN_PPL=1 and RUN_FINAL_LATENCY=1"
fi

echo "done rc=0 end=$(date -Is)" > "$STATUS"
echo "[DONE] $(date -Is)"
echo "[END] $(date -Is) rc=0"
