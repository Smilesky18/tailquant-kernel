#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
cd "$ROOT"

PY=${PY:-/data/yzy/miniconda3/envs/quarot-clean/bin/python}
PPL_SCRIPT="$ROOT/kernel_quant/scripts/eval_policy_v6_weightmode_v1.py"
SEARCH_ROOT="$ROOT/experiments/kernel_quant/layer_latency_split_v1/results/oldv6_lowratio_search_v2"
OUT_ROOT="$ROOT/experiments/kernel_quant/layer_latency_split_v1/results/oldv6_lowratio_ppl_v1"

export PYTHONPATH="$ROOT:$ROOT/fake_quant:$ROOT/kernel_quant:$ROOT/kernel_quant/scripts:${PYTHONPATH:-}"
export HF_HOME=${HF_HOME:-/data/yzy/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

QWEN_POLICY="$SEARCH_ROOT/qwen3_8b/split_lambda_0p08_lowratio/policy.json"
LLAMA_POLICY="$SEARCH_ROOT/llama3_8b/split_lambda_0p08_lowratio/policy.json"
if [[ ! -f "$QWEN_POLICY" ]]; then
  echo "[ERROR] missing Qwen policy: $QWEN_POLICY" >&2
  exit 2
fi
if [[ ! -f "$LLAMA_POLICY" ]]; then
  echo "[ERROR] missing Llama policy: $LLAMA_POLICY" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT/logs"
pids=()

echo "[LAUNCH] Qwen3-8B GPTQ PPL on GPU ${QWEN_PPL_GPU:-3}"
(
  export CUDA_VISIBLE_DEVICES=${QWEN_PPL_GPU:-3}
  "$PY" "$PPL_SCRIPT" \
    --policy "$QWEN_POLICY" \
    --out_dir "$OUT_ROOT/qwen3_8b/split_lambda_0p08_lowratio_gptq" \
    --model Qwen/Qwen3-8B \
    --dataset wikitext2 \
    --cal_dataset wikitext2 \
    --window_start 0 \
    --n_windows 128 \
    --seqlen 2048 \
    --gptq_seqlen 2048 \
    --nsamples 128 \
    --seed 0 \
    --rotation_config /data/yzy/quarot/qwen3-8B_layer_all.csv \
    --percdamp 0.01 \
    --weight_method gptq \
    --use_projected_ratio \
    --eps 1e-8
) > "$OUT_ROOT/logs/qwen3_8b_ppl.log" 2>&1 &
pids+=("$!")

echo "[LAUNCH] Llama3-8B GPTQ PPL on GPU ${LLAMA_PPL_GPU:-4}"
(
  export CUDA_VISIBLE_DEVICES=${LLAMA_PPL_GPU:-4}
  "$PY" "$PPL_SCRIPT" \
    --policy "$LLAMA_POLICY" \
    --out_dir "$OUT_ROOT/llama3_8b/split_lambda_0p08_lowratio_gptq" \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset wikitext2 \
    --cal_dataset wikitext2 \
    --window_start 0 \
    --n_windows 128 \
    --seqlen 2048 \
    --gptq_seqlen 2048 \
    --nsamples 128 \
    --seed 0 \
    --rotation_config /data/yzy/quarot/llama3-8B_layer.csv \
    --percdamp 0.01 \
    --weight_method gptq \
    --use_projected_ratio \
    --eps 1e-8
) > "$OUT_ROOT/logs/llama3_8b_ppl.log" 2>&1 &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
