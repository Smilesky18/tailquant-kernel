#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate group-aware lower-ratio Pareto policies from capped search history.

The input is a grouped capped policy produced by
calibrate_per_linear_v6_lowratio_grouped_capped_v3.py. This script re-selects
recorded per-linear history candidates with a stronger ratio penalty while
preserving v61 shared-prepare groups:

* q/k/v in the same layer share ratio/R and activation_percentile.
* gate/up in the same layer share ratio/R and activation_percentile.
* selected tiny non-zero R values are projected to zero for the whole group.

It never modifies the source policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


def parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def ratio_of(cfg: dict) -> float:
    return float(cfg.get("ratio_projected", cfg.get("ratio", 0.0)))


def module_r(cfg: dict, ratio: float | None = None) -> int:
    k = int(cfg.get("in_features", 0))
    r = ratio_of(cfg) if ratio is None else float(ratio)
    return int(math.ceil(k * r)) if r > 0.0 else 0


def candidate_ratio(item: dict) -> float:
    return float(item.get("ratio_projected_custom", item.get("ratio_projected", item.get("ratio", 0.0))))


def candidate_recon(item: dict) -> float:
    return float(item.get("val_reconstruction", item.get("best_val_reconstruction", 0.0)))


def candidate_total(item: dict) -> float:
    return float(item.get("val_total", item.get("best_val_total", candidate_recon(item))))


def make_current_candidate(cfg: dict) -> dict:
    return {
        "ratio_projected_custom": ratio_of(cfg),
        "ratio": float(cfg.get("ratio_continuous", ratio_of(cfg))),
        "activation_percentile": float(cfg.get("activation_percentile", 100.0)),
        "weight_percentile": float(cfg.get("weight_percentile", 100.0)),
        "val_reconstruction": float(cfg.get("best_val_reconstruction", 0.0)),
        "val_total": float(cfg.get("best_val_total", 0.0)),
        "source": "current_policy",
    }


def collect_candidates(cfg: dict, cap_ratio: float) -> list[dict]:
    items = [make_current_candidate(cfg)]
    items.extend(dict(x, source="history") for x in cfg.get("history", []))
    best_by_ratio: dict[float, dict] = {}
    for item in items:
        ratio = candidate_ratio(item)
        if ratio > cap_ratio + 1e-12:
            continue
        item = dict(item)
        item["ratio_projected_custom"] = ratio
        prev = best_by_ratio.get(ratio)
        if prev is None or (candidate_recon(item), candidate_total(item)) < (candidate_recon(prev), candidate_total(prev)):
            best_by_ratio[ratio] = item
    return [best_by_ratio[k] for k in sorted(best_by_ratio)]


def score_candidate(cfg: dict, item: dict, shrink_lambda: float) -> float:
    ratio = candidate_ratio(item)
    cost_weight = float(cfg.get("cost_weight_relative_to_mean_linear", 1.0))
    return candidate_recon(item) + float(shrink_lambda) * cost_weight * ratio


def choose_individual(cfg: dict, shrink_lambda: float, min_r_zero: int) -> tuple[float, dict | None, str]:
    cap = ratio_of(cfg)
    candidates = collect_candidates(cfg, cap)
    if not candidates:
        return cap, None, "no_candidate"
    best = min(candidates, key=lambda x: (score_candidate(cfg, x, shrink_lambda), candidate_ratio(x)))
    ratio = candidate_ratio(best)
    r = module_r(cfg, ratio)
    if 0 < r < min_r_zero:
        return 0.0, best, f"tiny_R_zero_individual_R{r}_lt_{min_r_zero}"
    return ratio, best, "selected_individual_sharedlambda"


def choose_group(cfgs: list[dict], shrink_lambda: float, min_r_zero: int) -> tuple[float, list[dict], str]:
    caps = [ratio_of(cfg) for cfg in cfgs]
    group_cap = min(caps) if caps else 0.0
    candidate_lists = [collect_candidates(cfg, group_cap) for cfg in cfgs]
    common_ratios = set(float(x["ratio_projected_custom"]) for x in candidate_lists[0])
    for items in candidate_lists[1:]:
        common_ratios &= set(float(x["ratio_projected_custom"]) for x in items)
    if not common_ratios:
        return group_cap, [make_current_candidate(c) for c in cfgs], "no_common_group_candidate"

    by_ratio = [{float(x["ratio_projected_custom"]): x for x in items} for items in candidate_lists]
    best_ratio = None
    best_items: list[dict] = []
    best_score = float("inf")
    for ratio in sorted(common_ratios):
        items = [m[ratio] for m in by_ratio]
        score = sum(score_candidate(cfg, item, shrink_lambda) for cfg, item in zip(cfgs, items))
        if (score, ratio) < (best_score, best_ratio if best_ratio is not None else float("inf")):
            best_score = score
            best_ratio = ratio
            best_items = items

    assert best_ratio is not None
    r = module_r(cfgs[0], best_ratio)
    if 0 < r < min_r_zero:
        return 0.0, best_items, f"tiny_R_zero_group_R{r}_lt_{min_r_zero}"
    return float(best_ratio), best_items, "selected_group_sharedlambda"


def layer_count_from_policy(policy: dict) -> int:
    max_layer = -1
    for name in policy.get("modules", {}):
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
            try:
                max_layer = max(max_layer, int(parts[2]))
            except ValueError:
                pass
    return max_layer + 1


def apply_choice(cfg: dict, ratio: float, item: dict | None, reason: str, group_name: str | None = None):
    old_ratio = ratio_of(cfg)
    cfg["ratio_projected_before_sharedlambda"] = old_ratio
    cfg["ratio_continuous_before_sharedlambda"] = float(cfg.get("ratio_continuous", old_ratio))
    cfg["ratio_projected"] = float(ratio)
    cfg["ratio_continuous"] = 0.0 if ratio == 0.0 else min(float(item.get("ratio", ratio)) if item else ratio, float(ratio))
    if item is not None:
        cfg["activation_percentile"] = float(item.get("activation_percentile", cfg.get("activation_percentile", 100.0)))
        cfg["weight_percentile"] = float(item.get("weight_percentile", cfg.get("weight_percentile", 100.0)))
        cfg["best_val_reconstruction"] = candidate_recon(item)
        cfg["best_val_total"] = candidate_total(item)
    cfg["sharedlambda_selection_reason"] = reason
    if group_name:
        cfg["sharedlambda_group"] = group_name


def recompute_summary(policy: dict) -> dict:
    ratios = [ratio_of(x) for x in policy["modules"].values()]
    mac_total = sum(int(x.get("mac_weight", 0)) for x in policy["modules"].values())
    mac_ratio = sum(int(x.get("mac_weight", 0)) * ratio_of(x) for x in policy["modules"].values()) / max(mac_total, 1)
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


def write_summary_csv(path: Path, rows: Iterable[dict]):
    rows = list(rows)
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_policy(base_policy: dict, shrink_lambda: float, min_r_zero: int) -> tuple[dict, dict]:
    policy = json.loads(json.dumps(base_policy))
    modules = policy["modules"]
    touched: set[str] = set()
    changes = []
    n_layers = layer_count_from_policy(policy)

    group_specs = [
        ("shared_qkv_v1", ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]),
        ("shared_gate_up_v1", ["mlp.gate_proj", "mlp.up_proj"]),
    ]
    for layer_id in range(n_layers):
        for group_name, locals_ in group_specs:
            keys = [f"model.layers.{layer_id}.{local}" for local in locals_]
            if not all(k in modules for k in keys):
                continue
            cfgs = [modules[k] for k in keys]
            ratio, items, reason = choose_group(cfgs, shrink_lambda, min_r_zero)
            target_ap = max(float(item.get("activation_percentile", cfg.get("activation_percentile", 100.0))) for cfg, item in zip(cfgs, items))
            for key, cfg, item in zip(keys, cfgs, items):
                apply_choice(cfg, ratio, item, reason, group_name)
                cfg["activation_percentile"] = target_ap
                touched.add(key)
            changes.append({"group": group_name, "layer": layer_id, "members": keys, "ratio_projected": ratio, "reason": reason})

    for key, cfg in modules.items():
        if key in touched:
            continue
        ratio, item, reason = choose_individual(cfg, shrink_lambda, min_r_zero)
        apply_choice(cfg, ratio, item, reason)
        changes.append({"module_name": key, "ratio_projected": ratio, "reason": reason})

    policy.setdefault("metadata", {})
    policy["metadata"].update({
        "postprocess": "grouped_capped_sharedlambda_pareto_v1",
        "source_policy": policy["metadata"].get("source_policy", ""),
        "sharedlambda": float(shrink_lambda),
        "min_R_zero": int(min_r_zero),
        "note": "Re-selects existing capped search history candidates while preserving q/k/v and gate/up shared-prepare groups.",
    })
    policy["summary"] = recompute_summary(policy)
    return policy, {"sharedlambda": float(shrink_lambda), "min_R_zero": int(min_r_zero), "changes": changes, "summary": policy["summary"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_policy", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--lambda_values", default="0.08,0.16,0.32,0.64")
    parser.add_argument("--min_R_zero", type=int, default=17)
    parser.add_argument("--label_prefix", default="sharedlambda")
    args = parser.parse_args()

    base_path = Path(args.base_policy)
    base_policy = json.loads(base_path.read_text(encoding="utf-8"))
    base_policy.setdefault("metadata", {})
    base_policy["metadata"]["source_policy"] = str(base_path)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in parse_floats(args.lambda_values):
        label_value = str(value).replace(".", "p")
        label = f"{args.label_prefix}_lambda_{label_value}"
        out_dir = out_root / label
        out_dir.mkdir(parents=True, exist_ok=True)
        policy, details = make_policy(base_policy, value, args.min_R_zero)
        (out_dir / "policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
        row = {"label": label, "policy": str(out_dir / "policy.json"), "sharedlambda": value, "min_R_zero": int(args.min_R_zero)}
        row.update(policy["summary"])
        rows.append(row)
        print("[SHAREDLAMBDA_POLICY] " + json.dumps(row, ensure_ascii=False), flush=True)
    write_summary_csv(out_root / "pareto_summary.csv", rows)
    print(f"[SHAREDLAMBDA_SUMMARY_CSV] {out_root / 'pareto_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
