#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from pathlib import Path

def read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def find_layer_csv(run_dir):
    files = sorted(glob.glob(str(Path(run_dir) / "*_prefill_layer_total_v53.csv")))
    if not files:
        files = sorted(glob.glob(str(Path(run_dir) / "*.csv")))
    files = [f for f in files if not f.endswith("_partial.csv")]
    return files[0] if files else None

def sum_by_batch(layer_csv):
    rows = read_csv_rows(layer_csv)
    out = {}
    for r in rows:
        try:
            b = int(float(r.get("batch", "")))
            ms = float(r.get("split_ms", "nan"))
        except Exception:
            continue
        out[b] = out.get(b, 0.0) + ms
    return out, len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    args = ap.parse_args()

    manifest_rows = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    run_root = Path(args.run_root)

    zero = None
    tag_data = {}
    for m in manifest_rows:
        tag = m["tag"]
        layer_csv = find_layer_csv(run_root / tag)
        if not layer_csv:
            tag_data[tag] = {"missing": True, "manifest": m}
            continue
        sums, nrows = sum_by_batch(layer_csv)
        tag_data[tag] = {"missing": False, "manifest": m, "layer_csv": layer_csv, "sum_by_batch": sums, "num_rows": nrows}
        if tag == "zero_all":
            zero = tag_data[tag]

    if zero is None or zero.get("missing"):
        raise SystemExit("[ERROR] zero_all run is missing; cannot compute marginal cost")

    zero_sums = zero["sum_by_batch"]
    rows = []
    for tag, d in tag_data.items():
        if tag == "zero_all":
            continue
        m = d["manifest"]
        if d.get("missing"):
            rows.append({
                "tag": tag, "shape_id": m["shape_id"], "K": m["K"], "N": m["N"], "ratio": m["ratio"],
                "batch": "", "zero_sum_split_ms": "", "shape_sum_split_ms": "", "delta_ms": "",
                "delta_ms_per_module": "", "target_module_count": m.get("target_module_count", ""), "target_R_sum": m.get("target_R_sum", ""),
                "num_rows": "", "layer_csv": "", "status": "missing",
            })
            continue
        for b, shape_sum in sorted(d["sum_by_batch"].items()):
            zero_sum = zero_sums.get(b)
            if zero_sum is None:
                continue
            target_count = max(int(float(m.get("target_module_count") or 1)), 1)
            delta = shape_sum - zero_sum
            rows.append({
                "tag": tag,
                "shape_id": m["shape_id"],
                "K": m["K"],
                "N": m["N"],
                "ratio": m["ratio"],
                "batch": str(b),
                "zero_sum_split_ms": f"{zero_sum:.9f}",
                "shape_sum_split_ms": f"{shape_sum:.9f}",
                "delta_ms": f"{delta:.9f}",
                "delta_ms_per_module": f"{(delta / target_count):.9f}",
                "target_module_count": m.get("target_module_count", ""),
                "target_R_sum": m.get("target_R_sum", ""),
                "num_rows": str(d["num_rows"]),
                "layer_csv": d["layer_csv"],
                "status": "ok",
            })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["tag","shape_id","K","N","ratio","batch","zero_sum_split_ms","shape_sum_split_ms","delta_ms","delta_ms_per_module","target_module_count","target_R_sum","num_rows","layer_csv","status"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "manifest": args.manifest,
        "run_root": args.run_root,
        "out_csv": str(out),
        "zero_all": zero,
        "num_rows": len(rows),
        "missing_tags": [t for t, d in tag_data.items() if d.get("missing")],
    }
    # Avoid dumping huge per-tag layer sums into json.
    if "zero_all" in summary and isinstance(summary["zero_all"], dict):
        summary["zero_all"] = {
            "layer_csv": summary["zero_all"].get("layer_csv"),
            "sum_by_batch": summary["zero_all"].get("sum_by_batch"),
            "num_rows": summary["zero_all"].get("num_rows"),
        }
    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[OK] rows={len(rows)} missing={len(summary['missing_tags'])}")
    print(f"[OUT] {out}")

if __name__ == "__main__":
    main()
