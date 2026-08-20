#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import re
from pathlib import Path

def find_ppl(log_path):
    if not Path(log_path).exists():
        return ""
    txt = Path(log_path).read_text(errors="ignore")
    matches = re.findall(r"(?:WIKITEXT2 PPL|PPL|ppl)\s*[:=]\s*([0-9.]+)", txt)
    return matches[-1] if matches else ""

def find_ppl_json(ppl_run_dir, tag):
    path = Path(ppl_run_dir) / tag / "result.json"
    if not path.exists():
        return ""
    try:
        return str(json.load(open(path)).get("ppl", ""))
    except Exception:
        return ""

def find_layer_csv(run_dir):
    files = sorted(glob.glob(str(Path(run_dir) / "*_prefill_layer_total_v53.csv")))
    files = [f for f in files if not f.endswith("_partial.csv")]
    return files[0] if files else ""

def sum_latency(layer_csv):
    if not layer_csv or not Path(layer_csv).exists():
        return {}, 0
    rows = list(csv.DictReader(open(layer_csv)))
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
    ap.add_argument("--policy_summary", required=True)
    ap.add_argument("--ppl_log_dir", required=True)
    ap.add_argument("--ppl_run_dir", required=True)
    ap.add_argument("--latency_run_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.policy_summary), delimiter="\t"))
    out_rows = []
    for r in rows:
        tag = r["tag"]
        ppl_log = str(Path(args.ppl_log_dir) / f"ppl_{tag}.log")
        ppl = find_ppl_json(args.ppl_run_dir, tag) or find_ppl(ppl_log)
        layer_csv = find_layer_csv(Path(args.latency_run_dir) / tag)
        lat, nrows = sum_latency(layer_csv)
        out = dict(r)
        out.update({
            "ppl": ppl,
            "latency_sum_b16_ms": f"{lat.get(16, float('nan')):.9f}" if 16 in lat else "",
            "latency_sum_b64_ms": f"{lat.get(64, float('nan')):.9f}" if 64 in lat else "",
            "latency_sum_b256_ms": f"{lat.get(256, float('nan')):.9f}" if 256 in lat else "",
            "latency_rows": str(nrows),
            "ppl_log": ppl_log,
            "latency_csv": layer_csv,
        })
        out_rows.append(out)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[OK] rows={len(out_rows)}")
    print(f"[OUT] {out_path}")

if __name__ == "__main__":
    main()
