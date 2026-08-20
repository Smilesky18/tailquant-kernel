#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped low-ratio variant of the old v6 per-Linear calibration flow.

This script intentionally does not modify the original v6/v1 files. It reuses
`calibrate_per_linear_v6_lowratio_v1.py` for per-linear optimization, then
post-processes the final policy for kernel-friendly grouping:

* q/k/v in the same layer share projected ratio/R and activation percentile.
* gate/up in the same layer share projected ratio/R and activation percentile.
* tiny non-zero ratios are projected to zero only when a zero-ratio candidate
  was observed within a configurable validation-reconstruction tolerance.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant" / "scripts"):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import calibrate_per_linear_v6_lowratio_v1 as V1  # noqa: E402


LAST_ARGS = None


def parse_args():
    global LAST_ARGS
    args = V1.parse_args()
    args.group_qkv = bool(int(os.environ.get("GROUP_QKV", "1")))
    args.group_gate_up = bool(int(os.environ.get("GROUP_GATE_UP", "1")))
    args.tiny_ratio_zero_threshold = float(os.environ.get("TINY_RATIO_ZERO_THRESHOLD", "0.01"))
    args.tiny_zero_recon_tolerance_rel = float(os.environ.get("TINY_ZERO_RECON_TOLERANCE_REL", str(args.recon_tolerance_rel)))
    args.tiny_zero_recon_tolerance_abs = float(os.environ.get("TINY_ZERO_RECON_TOLERANCE_ABS", str(args.recon_tolerance_abs)))
    LAST_ARGS = args
    return args


def layer_count_from_policy(policy: dict) -> int:
    max_layer = -1
    for name in policy.get("modules", {}):
        parts = name.split(".")
        try:
            idx = parts.index("layers")
            max_layer = max(max_layer, int(parts[idx + 1]))
        except Exception:
            continue
    return max_layer + 1


def safe_zero_tiny_ratios(policy: dict, args) -> list[dict]:
    changes = []
    threshold = float(args.tiny_ratio_zero_threshold)
    rel = float(args.tiny_zero_recon_tolerance_rel)
    abs_tol = float(args.tiny_zero_recon_tolerance_abs)
    for name, cfg in policy.get("modules", {}).items():
        ratio = float(cfg.get("ratio_projected", 0.0))
        if not (0.0 < ratio < threshold):
            continue
        selected_recon = float(cfg.get("best_val_reconstruction", cfg.get("old_best_val_reconstruction", float("inf"))))
        allowed = selected_recon * (1.0 + rel) + abs_tol
        zero_records = [
            h for h in cfg.get("history", [])
            if float(h.get("ratio_projected_custom", -1.0)) == 0.0
        ]
        if not zero_records:
            continue
        best_zero = min(zero_records, key=lambda h: (float(h.get("val_reconstruction", float("inf"))), float(h.get("val_total", float("inf")))))
        zero_recon = float(best_zero.get("val_reconstruction", float("inf")))
        if zero_recon <= allowed:
            cfg["ratio_projected_before_tiny_zero"] = ratio
            cfg["ratio_projected"] = 0.0
            cfg["ratio_continuous"] = 0.0
            cfg["activation_percentile"] = float(best_zero.get("activation_percentile", cfg.get("activation_percentile", 100.0)))
            cfg["best_val_reconstruction_before_tiny_zero"] = selected_recon
            cfg["best_val_reconstruction"] = zero_recon
            cfg["tiny_zero_reason"] = (
                f"zero candidate within reconstruction tolerance "
                f"(zero={zero_recon:.8g}, allowed={allowed:.8g})"
            )
            changes.append({
                "module_name": name,
                "old_ratio_projected": ratio,
                "new_ratio_projected": 0.0,
                "old_reconstruction": selected_recon,
                "zero_reconstruction": zero_recon,
            })
    return changes


def group_modules(policy: dict, layer_id: int, local_names: list[str], group_name: str) -> dict | None:
    modules = policy.get("modules", {})
    keys = [f"model.layers.{layer_id}.{local}" for local in local_names]
    if not all(k in modules for k in keys):
        return None

    cfgs = [modules[k] for k in keys]
    # Conservative for PPL: use the largest projected ratio/R in the group.
    target_ratio = max(float(c.get("ratio_projected", 0.0)) for c in cfgs)
    # Conservative for clipping: use the least clipping activation percentile.
    target_ap = max(float(c.get("activation_percentile", 100.0)) for c in cfgs)
    before = []
    for key, cfg in zip(keys, cfgs):
        before.append({
            "module_name": key,
            "ratio_projected": float(cfg.get("ratio_projected", 0.0)),
            "activation_percentile": float(cfg.get("activation_percentile", 100.0)),
        })
        cfg[f"{group_name}_before_ratio_projected"] = float(cfg.get("ratio_projected", 0.0))
        cfg[f"{group_name}_before_activation_percentile"] = float(cfg.get("activation_percentile", 100.0))
        cfg["ratio_projected"] = target_ratio
        cfg["activation_percentile"] = target_ap
        cfg["group_constraint"] = group_name

    return {
        "layer": layer_id,
        "group": group_name,
        "members": keys,
        "target_ratio_projected": target_ratio,
        "target_activation_percentile": target_ap,
        "before": before,
    }


def apply_group_constraints(policy: dict, args) -> tuple[dict, dict]:
    policy = json.loads(json.dumps(policy))
    tiny_changes = safe_zero_tiny_ratios(policy, args)
    group_changes = []
    n_layers = layer_count_from_policy(policy)
    for layer_id in range(n_layers):
        if args.group_qkv:
            item = group_modules(
                policy,
                layer_id,
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "group_qkv_v2",
            )
            if item is not None:
                group_changes.append(item)
        if args.group_gate_up:
            item = group_modules(
                policy,
                layer_id,
                ["mlp.gate_proj", "mlp.up_proj"],
                "group_gate_up_v2",
            )
            if item is not None:
                group_changes.append(item)

    ratios = [float(x.get("ratio_projected", 0.0)) for x in policy["modules"].values()]
    mac_total = sum(int(x.get("mac_weight", 0)) for x in policy["modules"].values())
    mac_ratio = (
        sum(int(x.get("mac_weight", 0)) * float(x.get("ratio_projected", 0.0)) for x in policy["modules"].values())
        / max(mac_total, 1)
    )
    policy["summary"] = {
        "module_count": len(policy["modules"]),
        "mean_projected_ratio_unweighted": float(np.mean(ratios)) if ratios else 0.0,
        "mac_weighted_projected_ratio": float(mac_ratio),
        "zero_ratio_module_count": sum(r == 0.0 for r in ratios),
        "split_module_count": sum(r > 0.0 for r in ratios),
    }
    policy.setdefault("metadata", {})
    policy["metadata"].update({
        "postprocess": "grouped_lowratio_v2",
        "group_qkv": bool(args.group_qkv),
        "group_gate_up": bool(args.group_gate_up),
        "tiny_ratio_zero_threshold": float(args.tiny_ratio_zero_threshold),
        "tiny_zero_recon_tolerance_rel": float(args.tiny_zero_recon_tolerance_rel),
        "tiny_zero_recon_tolerance_abs": float(args.tiny_zero_recon_tolerance_abs),
        "note": "Group ratios use max projected ratio and max activation percentile to preserve accuracy conservatively.",
    })
    details = {
        "tiny_zero_changes": tiny_changes,
        "group_changes": group_changes,
        "summary": policy["summary"],
    }
    return policy, details


def write_policy_summary_csv(path: Path, policy: dict):
    rows = []
    for name, cfg in sorted(policy["modules"].items()):
        rows.append({
            "module_name": name,
            "ratio_continuous": cfg.get("ratio_continuous", ""),
            "ratio_projected": cfg.get("ratio_projected", ""),
            "activation_percentile": cfg.get("activation_percentile", ""),
            "weight_percentile": cfg.get("weight_percentile", ""),
            "best_val_reconstruction": cfg.get("best_val_reconstruction", ""),
            "mac_weight": cfg.get("mac_weight", ""),
            "group_constraint": cfg.get("group_constraint", ""),
            "tiny_zero_reason": cfg.get("tiny_zero_reason", ""),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["module_name"])
        writer.writeheader()
        writer.writerows(rows)


def postprocess_output(args):
    out = Path(args.out_dir)
    policy_path = out / "policy.json"
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)
    raw_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    grouped_policy, details = apply_group_constraints(raw_policy, args)
    policy_path.write_text(json.dumps(grouped_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "policy_grouped_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    write_policy_summary_csv(out / "policy_summary.csv", grouped_policy)
    print("[GROUPED_V2_SUMMARY] " + json.dumps(details["summary"], ensure_ascii=False), flush=True)
    print(f"[GROUPED_V2_DETAILS] {out / 'policy_grouped_details.json'}", flush=True)


def main():
    V1.V6.parse_args = parse_args
    V1.V6.optimize_one_linear = V1.optimize_one_linear
    V1.V6.main()
    if LAST_ARGS is None:
        raise RuntimeError("parse_args was not called")
    postprocess_output(LAST_ARGS)


if __name__ == "__main__":
    main()
