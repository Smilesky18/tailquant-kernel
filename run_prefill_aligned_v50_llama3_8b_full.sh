#!/usr/bin/env bash
set -euo pipefail

cd /data/yzy/quarot-gpt-2

export CUDA_VISIBLE_DEVICES=1
export HF_HOME=/data/yzy/.cache/huggingface
export HF_HUB_CACHE=/data/yzy/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/data/yzy/miniconda3/envs/romeo_sm120/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export TORCH_CUDA_ARCH_LIST=12.0
export QFACTORY_ARCH=120
export NO_USE_FASTER_HADAMARD_TRANSFORM=1
export TORCH_EXTENSIONS_DIR=/data/yzy/tmp/torch_extensions_v50_llama3
export PYTHONPATH=/data/yzy/RoMeo:/data/yzy/quarot-gpt-2:/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/tools

mkdir -p /data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/results/prefill_split_aligned_v50_llama3_8b_full

/data/yzy/miniconda3/envs/romeo_sm120/bin/python -u \
  /data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/tools/bench_prefill_bf16_quarot_split_aligned_v50.py \
  --model meta-llama/Meta-Llama-3-8B \
  --label llama3_8b \
  --policy /data/yzy/quarot-gpt-2/experiments/kernel_quant/llama3_8b_v71_search_rotate/split_lambda_0p08/policy.json \
  --rotation_config /data/yzy/quarot/llama3-8B_layer.csv \
  --seq_len 128 \
  --batches 16,64,256 \
  --layers all \
  --variants bf16,quarot,split \
  --warmup 5 \
  --iters 20 \
  --component_warmup 1 \
  --component_iters 3 \
  --out_dir /data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/results/prefill_split_aligned_v50_llama3_8b_full \
  --local_files_only \
  --fused_topr_pack
