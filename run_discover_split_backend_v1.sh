#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/quarot-clean/bin/python}
STATUS=$EXP/status/discover_split_backend_v1.status

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/reports" "$EXP/results"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"
echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[PYBIN] $PYBIN"

cd "$ROOT"

export HF_HOME=/data/yzy/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:$EXP/tools:${PYTHONPATH:-}"

"$PYBIN" -u "$EXP/tools/bench_layer_bf16_vs_split_v1.py" \
  --mode discover \
  --out_dir "$EXP/reports"

echo "[DONE] $(date -Is)"
