#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/data/yzy/quarot-gpt-2}
EXP=${EXP:-$ROOT/experiments/kernel_quant/layer_latency_split_v1}
PYBIN=${PYBIN:-/data/yzy/miniconda3/envs/quarot-clean/bin/python}
MODEL=${MODEL:-Qwen/Qwen3-8B}

STATUS=$EXP/status/qwen3_8b_layer0_bf16_then_split_v4.status
BF16_OUT=$EXP/results/qwen3_8b_layer0_bf16_v4
SPLIT_OUT=$EXP/results/qwen3_8b_layer0_split_real_v4
SUMMARY=$EXP/reports/qwen3_8b_layer0_bf16_vs_split_v4_summary.json
CSV=$EXP/reports/qwen3_8b_layer0_bf16_vs_split_v4_summary.csv

mkdir -p "$EXP/logs" "$EXP/status" "$EXP/reports" "$EXP/results" "$BF16_OUT" "$SPLIT_OUT"

trap 'rc=$?; echo "[END] $(date -Is) rc=$rc"; if [ $rc -eq 0 ]; then echo "done rc=0 end=$(date -Is)" > "$STATUS"; else echo "failed rc=$rc end=$(date -Is)" > "$STATUS"; fi; exit $rc' EXIT

echo "running start=$(date -Is)" > "$STATUS"

echo "[START] $(date -Is)"
echo "[ROOT] $ROOT"
echo "[EXP] $EXP"
echo "[PYBIN] $PYBIN"
echo "[MODEL] $MODEL"
echo "[BF16_OUT] $BF16_OUT"
echo "[SPLIT_OUT] $SPLIT_OUT"
echo "[SUMMARY] $SUMMARY"

cd "$ROOT"

export HF_HOME=/data/yzy/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export PYTHONPATH="$ROOT:$EXP/tools:${PYTHONPATH:-}"

echo
echo "========== env =========="
"$PYBIN" - <<'PY'
import os, sys, torch, transformers
print("[PYTHON]", sys.executable)
print("[TORCH]", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("[TRANSFORMERS]", transformers.__version__, transformers.__file__)
print("[CUDA_VISIBLE_DEVICES]", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

echo
echo "========== locate policy / rotation =========="

POLICY=${POLICY:-}
if [ -z "$POLICY" ]; then
  for p in \
    "$ROOT/experiments/kernel_quant/qwen_per_linear_diff_calibration_v6_rotate/lambda_0p08/policy.json" \
    "$ROOT/experiments/kernel_quant/qwen_per_linear_diff_calibration_v6/lambda_0p08/policy.json" \
    "$ROOT/experiments/kernel_quant/real_kernel_fixed_v1/full_qwen3_8b/policy.json" \
    "$ROOT/experiments/kernel_quant/real_kernel_fixed_v1/full_qwen3_8b/split_policy.json"
  do
    if [ -f "$p" ]; then
      POLICY="$p"
      break
    fi
  done
fi

if [ -z "$POLICY" ] || [ ! -f "$POLICY" ]; then
  echo "[ERROR] Cannot find Qwen3-8B split policy.json"
  echo "[INFO] candidate policy files:"
  find "$ROOT/experiments" -type f -name 'policy.json' 2>/dev/null | grep -iE 'qwen|lambda|split|kernel|real' | sort | tail -n 80 || true
  exit 2
fi

ROTATION_CONFIG=${ROTATION_CONFIG:-}
if [ -z "$ROTATION_CONFIG" ]; then
  for p in \
    /data/yzy/quarot/qwen3-8B_layer_all.csv \
    /data/yzy/quarot/qwen3-8b_layer_all.csv \
    /data/yzy/quarot-gpt-2/qwen3-8B_layer_all.csv \
    /data/yzy/quarot-gpt-2/qwen3-8b_layer_all.csv
  do
    if [ -f "$p" ]; then
      ROTATION_CONFIG="$p"
      break
    fi
  done
fi

if [ -z "$ROTATION_CONFIG" ] || [ ! -f "$ROTATION_CONFIG" ]; then
  echo "[WARN] Cannot find rotation config; split benchmark will run without --rotation_config"
else
  echo "[ROTATION_CONFIG] $ROTATION_CONFIG"
fi

echo "[POLICY] $POLICY"

echo
echo "========== policy sanity =========="
POLICY="$POLICY" "$PYBIN" - <<'PY'
import json, os
p = os.environ["POLICY"]
obj = json.load(open(p))
mods = obj.get("modules", {})
hist = {}
for _, v in mods.items():
    r = float(v.get("ratio", 0.0))
    hist[str(r)] = hist.get(str(r), 0) + 1
print("[POLICY_MODULES]", len(mods))
print("[POLICY_RATIO_HIST]", hist)
nonzero = sum(1 for v in mods.values() if float(v.get("ratio", 0.0)) > 0)
print("[POLICY_NONZERO]", nonzero)
if len(mods) <= 0:
    raise SystemExit("[ERROR] empty policy")
if nonzero <= 0:
    raise SystemExit("[ERROR] zero nonzero policy")
PY

echo
echo "========== BF16 layer0 latency START $(date -Is) =========="
"$PYBIN" -u "$EXP/tools/bench_layer_bf16_probe_qwen3_v2.py" \
  --model "$MODEL" \
  --layer_idx 0 \
  --seq_len 128 \
  --batches 16,64,256 \
  --dtype bf16 \
  --device cuda:0 \
  --warmup 20 \
  --iters 100 \
  --use_cuda_graph \
  --out_dir "$BF16_OUT"
echo "========== BF16 layer0 latency END $(date -Is) =========="

echo
echo "========== real split layer benchmark START $(date -Is) =========="

SPLIT_CMD=(
  "$PYBIN" -u "$ROOT/kernel_quant/scripts/bench_real_split_fullstack_v1.py"
  --model "$MODEL"
  --policy "$POLICY"
  --storage_mode dual
  --cal_dataset wikitext2
  --nsamples 128
  --gptq_seqlen 2048
  --percdamp 0.01
  --seed 0
  --eps 1e-8
  --out_dir "$SPLIT_OUT"
  --layer_warmup 20
  --layer_runs 100
  --layer_repeats 1
  --timing graph
)

if [ -n "${ROTATION_CONFIG:-}" ] && [ -f "$ROTATION_CONFIG" ]; then
  SPLIT_CMD+=(--rotation_config "$ROTATION_CONFIG")
fi

printf '[SPLIT_CMD]'
printf ' %q' "${SPLIT_CMD[@]}"
printf '\n'

"${SPLIT_CMD[@]}"

echo "========== real split layer benchmark END $(date -Is) =========="

echo
echo "========== collect summary =========="
BF16_OUT="$BF16_OUT" SPLIT_OUT="$SPLIT_OUT" SUMMARY="$SUMMARY" CSV="$CSV" "$PYBIN" - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

bf16_out = Path(os.environ["BF16_OUT"])
split_out = Path(os.environ["SPLIT_OUT"])
summary_path = Path(os.environ["SUMMARY"])
csv_path = Path(os.environ["CSV"])

bf16_json = bf16_out / "bf16_layer_latency_probe_v2.json"
bf16_rows = json.load(open(bf16_json))

bf16_by_batch = {}
for r in bf16_rows:
    bf16_by_batch[int(r["batch"])] = float(r["bf16_ms"])

# 尽量鲁棒地解析 split 输出。不同版本文件名可能不同。
candidates = []
for p in split_out.rglob("*"):
    if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".csv", ".txt", ".log"}:
        candidates.append(p)

print("[SPLIT_FILES]")
for p in candidates:
    print(p)

split_by_batch = {}

# 1) 先扫 CSV
for p in candidates:
    if p.suffix.lower() != ".csv":
        continue
    try:
        rows = list(csv.DictReader(open(p)))
    except Exception:
        continue
    for row in rows:
        keys = {k.lower(): k for k in row.keys()}
        batch_key = None
        for k in keys:
            if k in {"batch", "bsz", "batch_size"}:
                batch_key = keys[k]
                break
        latency_key = None
        for k in keys:
            if k in {"latency_ms", "graph_ms", "mean_ms", "ms", "layer_ms"}:
                latency_key = keys[k]
                break
        if batch_key and latency_key:
            try:
                b = int(float(row[batch_key]))
                ms = float(row[latency_key])
                if b in {16, 64, 256} and ms > 0:
                    split_by_batch[b] = ms
            except Exception:
                pass

# 2) 再扫 JSON
def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

for p in candidates:
    if p.suffix.lower() not in {".json", ".jsonl"}:
        continue
    try:
        txt = p.read_text(errors="ignore")
        if p.suffix.lower() == ".jsonl":
            objs = [json.loads(x) for x in txt.splitlines() if x.strip().startswith("{")]
        else:
            objs = [json.loads(txt)]
    except Exception:
        continue

    for obj in objs:
        for d in walk(obj):
            low = {str(k).lower(): k for k in d.keys()}
            batch_key = None
            for k in low:
                if k in {"batch", "bsz", "batch_size"}:
                    batch_key = low[k]
                    break
            latency_key = None
            for k in low:
                if k in {"latency_ms", "graph_ms", "mean_ms", "ms", "layer_ms"}:
                    latency_key = low[k]
                    break
            if batch_key and latency_key:
                try:
                    b = int(float(d[batch_key]))
                    ms = float(d[latency_key])
                    if b in {16, 64, 256} and ms > 0:
                        split_by_batch[b] = ms
                except Exception:
                    pass

# 3) 最后扫文本模式
for p in candidates:
    if p.suffix.lower() not in {".txt", ".log"}:
        continue
    txt = p.read_text(errors="ignore")
    # 常见格式兜底：batch=16 ... graph_ms=xx / latency_ms=xx / mean_ms=xx
    for b in [16, 64, 256]:
        pat = re.compile(rf"(?:batch|bsz|batch_size)\s*[=:]\s*{b}.*?(?:graph_ms|latency_ms|mean_ms|layer_ms|ms)\s*[=:]\s*([0-9.]+)", re.I | re.S)
        m = pat.search(txt)
        if m:
            split_by_batch[b] = float(m.group(1))

rows = []
for b in sorted(bf16_by_batch):
    bf16_ms = bf16_by_batch[b]
    split_ms = split_by_batch.get(b)
    row = {
        "batch": b,
        "seq_len": 128,
        "bf16_ms": bf16_ms,
        "split_ms": split_ms,
        "speedup_bf16_over_split": (bf16_ms / split_ms) if split_ms else None,
        "normalized_split_latency": (split_ms / bf16_ms) if split_ms else None,
    }
    rows.append(row)

summary = {
    "bf16_json": str(bf16_json),
    "split_out": str(split_out),
    "split_files": [str(p) for p in candidates],
    "bf16_by_batch": bf16_by_batch,
    "split_by_batch": split_by_batch,
    "rows": rows,
}

summary_path.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(summary_path, "w"), indent=2, ensure_ascii=False)

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "batch",
            "seq_len",
            "bf16_ms",
            "split_ms",
            "speedup_bf16_over_split",
            "normalized_split_latency",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("[SUMMARY_JSON]", summary_path)
print("[SUMMARY_CSV]", csv_path)
print("[SUMMARY]")
print(json.dumps(summary, indent=2, ensure_ascii=False))

missing = [r["batch"] for r in rows if r["split_ms"] is None]
if missing:
    print("[WARN] Could not auto-parse split latency for batches:", missing)
    print("[WARN] Split run may still be successful. Need inspect split output files above.")
else:
    print("[PASS] BF16 vs Split summary parsed for all batches")
PY

echo
echo "========== error scan =========="
grep -RInE "Traceback|RuntimeError|CUDA error|out of memory|OOM|\[ERROR\]" "$BF16_OUT" "$SPLIT_OUT" || true

echo "[DONE] $(date -Is)"
