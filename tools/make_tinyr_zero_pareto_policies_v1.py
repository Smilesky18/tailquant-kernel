#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate Pareto policies by zeroing tiny non-zero R modules.

This is a post-search export step for the capped grouped search results. It
does not modify the original search files or policies. For each requested
`min_R_keep`, any module with `0 < R < min_R_keep` is projected to zero.
Because the input policy is already grouped, q/k/v and gate/up group members
remain aligned when they share the same ratio/R.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

def parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def module_r(cfg: dict) -> int:
    k = int(cfg.get("in_features", 0))
    ratio = float(cfg.get("ratio_projected", cfg.get("ratio", 0.0)))
    return int(math.ceil(k * ratio)) if ratio > 0.0 else 0


def recompute_summary(policy: dict) -> dict:
    ratios = [float(x.get("ratio_projected", 0.0)) for x in policy["modules"].values()]
    mac_total = sum(int(x.get("mac_weight", 0)) for x in policy["modules"].values())
    mac_ratio = (
        sum(int(x.get("mac_weight", 0)) * float(x.get("ratio_projected", 0.0)) for x in policy["modules"].values())
        / max(mac_total, 1)
    )
    buckets = {"R0": 0, "R1_16": 0, "R17_32": 0, "R33_64": 0, "R65_128": 0, "R129_plus": 0}
    for cfg in policy["modules"].values():
        r = module_r(cfg)
        if r == 0:
            buckets["R0"] += 1
        elif r <= 16:
            buckets["R1_16"] += 1
        elif r <= 32:
            buckets["R17_32"] += 1
        elif r <= 64:
            buckets["R33_64"] += 1
        elif r <= 128:
            buckets["R65_128"] += 1
        else:
            buckets["R129_plus"] += 1
    summary = {
        "module_count": len(policy["modules"]),
        "mean_projected_ratio_unweighted": float(sum(ratios) / len(ratios)) if ratios else 0.0,
        "mac_weighted_projected_ratio": float(mac_ratio),
        "zero_ratio_module_count": sum(r == 0.0 for r in ratios),
        "split_module_count": sum(r > 0.0 for r in ratios),
        "sum_R": sum(module_r(cfg) for cfg in policy["modules"].values()),
    }
    summary.update(buckets)
    return summary


def write_summary_csv(path: Path, rows: list[dict]):
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_policy(base_policy: dict, min_r_keep: int) -> tuple[dict, dict]:
    policy = json.loads(json.dumps(base_policy))
    changes = []
    for name, cfg in policy["modules"].items():
        r = module_r(cfg)
        if 0 < r < int(min_r_keep):
            old_ratio = float(cfg.get("ratio_projected", 0.0))
            cfg["ratio_projected_before_tinyr_zero"] = old_ratio
            cfg["ratio_continuous_before_tinyr_zero"] = float(cfg.get("ratio_continuous", old_ratio))
            cfg["R_before_tinyr_zero"] = r
            cfg["ratio_projected"] = 0.0
            cfg["ratio_continuous"] = 0.0
            cfg["tinyr_zero_reason"] = f"0 < R={r} < min_R_keep={min_r_keep}"
            changes.append({
                "module_name": name,
                "old_ratio_projected": old_ratio,
                "old_R": r,
                "new_ratio_projected": 0.0,
            })
    policy.setdefault("metadata", {})
    policy["metadata"].update({
        "postprocess": "tinyr_zero_pareto_v1",
        "min_R_keep": int(min_r_keep),
        "source_policy": policy["metadata"].get("source_policy", ""),
    })
    policy["summary"] = recompute_summary(policy)
    details = {"min_R_keep": int(min_r_keep), "changes": changes, "summary": policy["summary"]}
    return policy, details


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_policy", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--min_R_keep_values", default="1,17,33,65")
    p.add_argument("--label_prefix", default="tinyr")
    args = p.parse_args()

    base_path = Path(args.base_policy)
    base_policy = json.loads(base_path.read_text(encoding="utf-8"))
    base_policy.setdefault("metadata", {})
    base_policy["metadata"]["source_policy"] = str(base_path)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for min_r_keep in parse_ints(args.min_R_keep_values):
        label = f"{args.label_prefix}_keep_ge{min_r_keep}"
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        policy, details = make_policy(base_policy, min_r_keep)
        (out_dir / "policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
        row = {"label": label, "policy": str(out_dir / "policy.json"), "min_R_keep": min_r_keep}
        row.update(policy["summary"])
        summary_rows.append(row)
        print("[TINYR_POLICY] " + json.dumps(row, ensure_ascii=False), flush=True)
    write_summary_csv(out_root / "pareto_summary.csv", summary_rows)
    print(f"[TINYR_SUMMARY_CSV] {out_root / 'pareto_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
