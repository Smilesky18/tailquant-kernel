# CE-gradient + shape-latency ratio search v1

Files:

- `calibrate_shape_latency_cegrad_ratio_v1.py`

Install location on the server:

```bash
mkdir -p /data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/tools
cp calibrate_shape_latency_cegrad_ratio_v1.py \
  /data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/tools/
```

This script does not modify `calibrate_per_linear_v74.py` or `diff_quant_v7.py`.
It performs a discrete candidate search using the first-order CE proxy:

```text
CE_proxy ~= <Y_quant - Y_fp, dCE/dY_fp>
score = delta_CE + latency_lambda * delta_latency(shape, ratio)
```

Recommended smoke run: 1 experiment group.

```bash
cd /data/yzy/quarot-gpt-2
conda activate quarot-clean

export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export TORCH_CUDA_ARCH_LIST=12.0
export QFACTORY_ARCH=120
export NO_USE_FASTER_HADAMARD_TRANSFORM=1
export HF_HOME=/data/yzy/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/data/yzy/RoMeo:/data/yzy/quarot-gpt-2:/data/yzy/quarot-gpt-2/kernel_quant/scripts:/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/tools

OUT=/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/results/qwen3_8b_cegrad_ratio_v1_smoke
mkdir -p "$OUT/logs"

nohup bash -lc '
set -euo pipefail
printf "[START] %s\n" "$(date -Is)"
python /data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/tools/calibrate_shape_latency_cegrad_ratio_v1.py \
  --model Qwen/Qwen3-8B \
  --dataset wikitext2 \
  --mode split_v6 \
  --rotation_config /data/yzy/quarot/qwen3-8B_layer_all.csv \
  --out_dir /data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/results/qwen3_8b_cegrad_ratio_v1_smoke/policy \
  --nsamples 1 \
  --seqlen 128 \
  --max_layers 1 \
  --capture_rows 256 \
  --out_channels 128 \
  --latency_lambda 0.0 \
  --missing_latency proxy
printf "[END] %s\n" "$(date -Is)"
' > "$OUT/logs/run.log" 2>&1 &
```

Full Qwen3-8B run should pass the measured shape-latency table:

```bash
--latency_table /path/to/shape_latency_cost_table.csv \
--latency_batch b16 \
--latency_lambda 0.01
```

Expected latency table columns are flexible, but the safest format is:

```csv
K,N,ratio,b16,b64,b256
4096,4096,0,0,0,0
4096,4096,0.00125,...
4096,12288,0.00125,...
12288,4096,0.00125,...
```
