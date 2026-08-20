#!/usr/bin/env bash
set -u

EXP=/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1
PY=/data/yzy/miniconda3/envs/romeo_sm120/bin/python
OUT_DIR="$EXP/results/qwen3_8b_l18_split_opts_v58"

mkdir -p "$OUT_DIR"

echo "[START] $(date -Is) qwen3_l18_split_opts_v58"

export PATH=/data/yzy/miniconda3/envs/romeo_sm120/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export TORCH_CUDA_ARCH_LIST=12.0
export QFACTORY_ARCH=120
export NO_USE_FASTER_HADAMARD_TRANSFORM=1
export HF_HOME=/data/yzy/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/data/yzy/RoMeo:/data/yzy/quarot-gpt-2:$EXP/tools

"$PY" "$EXP/tools/bench_qwen3_l18_split_opts_v58.py" \
  --out_dir "$OUT_DIR" \
  --local_files_only \
  --batches 16,64,256 \
  --warmup 5 \
  --iters 30
rc=$?

echo "[END] $(date -Is) rc=$rc"
exit "$rc"
