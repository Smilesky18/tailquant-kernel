#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
cd "$ROOT"

PY=${PY:-/data/yzy/miniconda3/envs/romeo_sm120/bin/python}
BENCH="$ROOT/experiments/kernel_quant/layer_latency_split_v1/tools/bench_prefill_bf16_romeoquarotdense_split_total_v57.py"
SEARCH_ROOT="$ROOT/experiments/kernel_quant/layer_latency_split_v1/results/oldv6_lowratio_search_v2"
OUT_ROOT="$ROOT/experiments/kernel_quant/layer_latency_split_v1/results/oldv6_lowratio_latency_v57_v2"

export PATH="/data/yzy/miniconda3/envs/romeo_sm120/bin:${PATH:-}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDACXX=${CUDACXX:-/usr/local/cuda/bin/nvcc}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}
export QFACTORY_ARCH=${QFACTORY_ARCH:-120}
export NO_USE_FASTER_HADAMARD_TRANSFORM=${NO_USE_FASTER_HADAMARD_TRANSFORM:-1}
export PYTHONPATH="/data/yzy/RoMeo:$ROOT:$ROOT/experiments/kernel_quant/layer_latency_split_v1/tools:$ROOT/fake_quant:$ROOT/kernel_quant:${PYTHONPATH:-}"
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

echo "[LAUNCH] Qwen3-8B v57 full-layer latency on GPU ${QWEN_LAT_GPU:-5}"
(
  export CUDA_VISIBLE_DEVICES=${QWEN_LAT_GPU:-5}
  "$PY" "$BENCH" \
    --model Qwen/Qwen3-8B \
    --label qwen3_8b_oldv6_lowratio_v57 \
    --policy "$QWEN_POLICY" \
    --rotation_config /data/yzy/quarot/qwen3-8B_layer_all.csv \
    --seq_len 128 \
    --batches 16,64,256 \
    --layers all \
    --variants bf16,romeo,split \
    --device cuda:0 \
    --warmup 5 \
    --iters 20 \
    --out_dir "$OUT_ROOT/qwen3_8b" \
    --local_files_only \
    --qfactory_fast_preset qwen3_sm120_v1
) > "$OUT_ROOT/logs/qwen3_8b_latency_v57.log" 2>&1 &
pids+=("$!")

echo "[LAUNCH] Llama3-8B v57 full-layer latency on GPU ${LLAMA_LAT_GPU:-6}"
(
  export CUDA_VISIBLE_DEVICES=${LLAMA_LAT_GPU:-6}
  "$PY" "$BENCH" \
    --model meta-llama/Meta-Llama-3-8B \
    --label llama3_8b_oldv6_lowratio_v57 \
    --policy "$LLAMA_POLICY" \
    --rotation_config /data/yzy/quarot/llama3-8B_layer.csv \
    --seq_len 128 \
    --batches 16,64,256 \
    --layers all \
    --variants bf16,romeo,split \
    --device cuda:0 \
    --warmup 5 \
    --iters 20 \
    --out_dir "$OUT_ROOT/llama3_8b" \
    --local_files_only \
    --qfactory_fast_preset none
) > "$OUT_ROOT/logs/llama3_8b_latency_v57.log" 2>&1 &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
exit "$failed"
