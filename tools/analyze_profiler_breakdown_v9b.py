import csv
import json
import re
from pathlib import Path
from collections import defaultdict

OUT = Path("/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/results/profile_layer_pure_split_breakdown_v9")

def fnum(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def category(key: str):
    k = key.lower()
    if "policy_activation_pack" in k or "quant" in k or "pack" in k or "top" in k or "select" in k or "abs" in k:
        return "activation_quant_pack_topr"
    if "single_linear_real" in k or "gemm" in k or "cutlass" in k or "mma" in k or "matmul" in k:
        return "int4_gemm_or_linear_kernel"
    if "singlecopy_workspace" in k or "layout" in k or "row_to_col" in k or "col" in k:
        return "layout_workspace"
    if "scaled_dot_product" in k or "flash" in k or "attention" in k or "sdpa" in k:
        return "attention"
    if "layer_norm" in k or "rms" in k or "norm" in k:
        return "norm"
    if "silu" in k or "mul" in k or "add" in k or "residual" in k:
        return "elementwise"
    if "memcpy" in k or "copy" in k or "fill" in k or "zero" in k or "empty" in k:
        return "memory_misc"
    return "other"

def read_prof(path: Path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["self_cuda_us"] = fnum(r.get("self_cuda_time_total_us"))
        r["cuda_us"] = fnum(r.get("cuda_time_total_us"))
        r["cat"] = category(r.get("key", ""))
    return rows

all_reports = {}
for path in sorted(OUT.glob("torch_profiler_*_b*.csv")):
    name = path.stem
    rows = read_prof(path)
    total_self = sum(r["self_cuda_us"] for r in rows)

    by_cat = defaultdict(float)
    by_cat_count = defaultdict(int)
    for r in rows:
        by_cat[r["cat"]] += r["self_cuda_us"]
        by_cat_count[r["cat"]] += int(float(r.get("count", 0) or 0))

    cat_rows = []
    for c, us in sorted(by_cat.items(), key=lambda x: -x[1]):
        cat_rows.append({
            "category": c,
            "self_cuda_us": us,
            "self_cuda_ms": us / 1000.0,
            "percent": (100.0 * us / total_self) if total_self > 0 else 0.0,
            "kernel_count": by_cat_count[c],
        })

    top_rows = []
    for r in sorted(rows, key=lambda x: -x["self_cuda_us"])[:40]:
        top_rows.append({
            "category": r["cat"],
            "key": r.get("key"),
            "count": r.get("count"),
            "self_cuda_ms": r["self_cuda_us"] / 1000.0,
            "cuda_ms": r["cuda_us"] / 1000.0,
        })

    report = {
        "file": str(path),
        "total_self_cuda_ms": total_self / 1000.0,
        "by_category": cat_rows,
        "top_kernels": top_rows,
    }
    all_reports[name] = report

    print("\n" + "=" * 100)
    print("[FILE]", path)
    print("[TOTAL_SELF_CUDA_MS]", total_self / 1000.0)
    print("[BY_CATEGORY]")
    for x in cat_rows:
        print(json.dumps(x, ensure_ascii=False))
    print("[TOP_KERNELS]")
    for x in top_rows[:25]:
        print(json.dumps(x, ensure_ascii=False))

out_json = OUT / "profiler_category_summary_v9b.json"
json.dump(all_reports, open(out_json, "w"), indent=2, ensure_ascii=False)
print("\n[WROTE]", out_json)
