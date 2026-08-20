#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped low-ratio search capped by an existing v1 policy.

This script reuses the v1 differentiable per-linear search, but constrains the
exported policy so every module's final projected ratio is no larger than the
corresponding ratio in a reference policy. For q/k/v and gate/up groups, the
shared group ratio is additionally capped by the minimum reference ratio among
members, so the equality constraint never violates any member's cap.

The reference policy is supplied with:

```bash
RATIO_CAP_POLICY=/path/to/oldv6_lowratio/policy.json
```
"""
from __future__ import annotations

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

import calibrate_per_linear_v6_lowratio_grouped_v2 as V2  # noqa: E402


LAST_ARGS = None
CAP_POLICY = None
CAP_MODULES: dict[str, dict] = {}


def _ratio_from_cfg(cfg: dict) -> float:
    return float(cfg.get("ratio_projected", cfg.get("ratio", 0.0)))


def load_cap_policy(path_text: str | None):
    if not path_text:
        return None, {}
    path = Path(path_text)
    policy = json.loads(path.read_text(encoding="utf-8"))
    modules = policy.get("modules", {})
    return policy, modules


def parse_args():
    global LAST_ARGS, CAP_POLICY, CAP_MODULES
    args = V2.parse_args()
    args.ratio_cap_policy = os.environ.get("RATIO_CAP_POLICY", "")
    CAP_POLICY, CAP_MODULES = load_cap_policy(args.ratio_cap_policy)
    if CAP_POLICY is None:
        raise RuntimeError("RATIO_CAP_POLICY must point to the v1 lowratio policy.json")
    LAST_ARGS = args
    return args


def _choose_capped_history_record(result: dict, cap_ratio: float, args) -> dict | None:
    history = list(result.get("history", []))
    if not history:
        return None
    feasible_cap = [
        h for h in history
        if float(h.get("ratio_projected_custom", float("inf"))) <= cap_ratio + 1e-12
    ]
    if not feasible_cap:
        return None

    best_recon = min(float(h.get("val_reconstruction", float("inf"))) for h in history)
    allowed = best_recon * (1.0 + float(args.recon_tolerance_rel)) + float(args.recon_tolerance_abs)
    feasible = [
        h for h in feasible_cap
        if float(h.get("val_reconstruction", float("inf"))) <= allowed
    ]
    if not feasible:
        feasible = feasible_cap
    return min(
        feasible,
        key=lambda h: (
            float(h.get("ratio_projected_custom", float("inf"))),
            float(h.get("ratio", float("inf"))),
            float(h.get("val_reconstruction", float("inf"))),
            float(h.get("val_total", float("inf"))),
        ),
    )


def optimize_one_linear_capped(**kwargs):
    result = V2.V1.optimize_one_linear(**kwargs)
    args = kwargs["args"]
    module_name = str(result["module_name"])
    cap_cfg = CAP_MODULES.get(module_name)
    if cap_cfg is None:
        return result

    cap_ratio = _ratio_from_cfg(cap_cfg)
    current = float(result.get("ratio_projected", 0.0))
    if current <= cap_ratio + 1e-12:
        result["ratio_cap_policy"] = getattr(args, "ratio_cap_policy", "")
        result["ratio_cap_projected"] = cap_ratio
        return result

    chosen = _choose_capped_history_record(result, cap_ratio, args)
    result["ratio_before_cap"] = current
    result["ratio_cap_policy"] = getattr(args, "ratio_cap_policy", "")
    result["ratio_cap_projected"] = cap_ratio
    if chosen is None:
        result["ratio_projected"] = cap_ratio
        result["ratio_continuous"] = min(float(result.get("ratio_continuous", cap_ratio)), cap_ratio)
        result["cap_selection_reason"] = "forced cap; no searched candidate under cap"
        return result

    result["ratio_projected"] = float(chosen.get("ratio_projected_custom", cap_ratio))
    result["ratio_continuous"] = float(chosen.get("ratio", result["ratio_projected"]))
    result["activation_percentile"] = float(chosen.get("activation_percentile", result.get("activation_percentile", 100.0)))
    result["weight_percentile"] = float(chosen.get("weight_percentile", result.get("weight_percentile", 100.0)))
    result["best_val_total"] = float(chosen.get("val_total", result.get("best_val_total", 0.0)))
    result["best_val_reconstruction"] = float(chosen.get("val_reconstruction", result.get("best_val_reconstruction", 0.0)))
    result["best_val_cost"] = float(chosen.get("val_cost", result.get("best_val_cost", 0.0)))
    result["cap_selection_reason"] = "selected searched candidate under v1 ratio cap"
    return result


def cap_individual_modules(policy: dict) -> list[dict]:
    changes = []
    for name, cfg in policy.get("modules", {}).items():
        cap_cfg = CAP_MODULES.get(name)
        if cap_cfg is None:
            continue
        cap = _ratio_from_cfg(cap_cfg)
        old = float(cfg.get("ratio_projected", 0.0))
        if old <= cap + 1e-12:
            cfg["ratio_cap_projected"] = cap
            continue
        cfg["ratio_projected_before_cap_postprocess"] = old
        cfg["ratio_projected"] = cap
        cfg["ratio_continuous"] = min(float(cfg.get("ratio_continuous", cap)), cap)
        cfg["ratio_cap_projected"] = cap
        cfg["cap_postprocess_reason"] = "individual module ratio clipped to v1 cap"
        changes.append({"module_name": name, "old_ratio_projected": old, "new_ratio_projected": cap})
    return changes


def group_modules_capped(policy: dict, layer_id: int, local_names: list[str], group_name: str) -> dict | None:
    modules = policy.get("modules", {})
    keys = [f"model.layers.{layer_id}.{local}" for local in local_names]
    if not all(k in modules for k in keys):
        return None

    cfgs = [modules[k] for k in keys]
    raw_target = max(float(c.get("ratio_projected", 0.0)) for c in cfgs)
    caps = [_ratio_from_cfg(CAP_MODULES[k]) for k in keys if k in CAP_MODULES]
    group_cap = min(caps) if caps else raw_target
    target_ratio = min(raw_target, group_cap)
    target_ap = max(float(c.get("activation_percentile", 100.0)) for c in cfgs)
    before = []
    for key, cfg in zip(keys, cfgs):
        old_ratio = float(cfg.get("ratio_projected", 0.0))
        before.append({
            "module_name": key,
            "ratio_projected": old_ratio,
            "ratio_cap_projected": _ratio_from_cfg(CAP_MODULES[key]) if key in CAP_MODULES else None,
            "activation_percentile": float(cfg.get("activation_percentile", 100.0)),
        })
        cfg[f"{group_name}_before_ratio_projected"] = old_ratio
        cfg[f"{group_name}_before_activation_percentile"] = float(cfg.get("activation_percentile", 100.0))
        cfg["ratio_projected"] = target_ratio
        cfg["ratio_continuous"] = min(float(cfg.get("ratio_continuous", target_ratio)), target_ratio)
        cfg["activation_percentile"] = target_ap
        cfg["group_constraint"] = group_name
        cfg["group_ratio_cap_projected"] = group_cap

    return {
        "layer": layer_id,
        "group": group_name,
        "members": keys,
        "raw_target_ratio_projected": raw_target,
        "group_cap_projected": group_cap,
        "target_ratio_projected": target_ratio,
        "target_activation_percentile": target_ap,
        "before": before,
    }


def apply_capped_group_constraints(policy: dict, args) -> tuple[dict, dict]:
    policy = json.loads(json.dumps(policy))
    tiny_changes = V2.safe_zero_tiny_ratios(policy, args)
    individual_cap_changes = cap_individual_modules(policy)
    group_changes = []
    n_layers = V2.layer_count_from_policy(policy)
    for layer_id in range(n_layers):
        if args.group_qkv:
            item = group_modules_capped(
                policy,
                layer_id,
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "group_qkv_capped_v3",
            )
            if item is not None:
                group_changes.append(item)
        if args.group_gate_up:
            item = group_modules_capped(
                policy,
                layer_id,
                ["mlp.gate_proj", "mlp.up_proj"],
                "group_gate_up_capped_v3",
            )
            if item is not None:
                group_changes.append(item)

    ratios = [float(x.get("ratio_projected", 0.0)) for x in policy["modules"].values()]
    mac_total = sum(int(x.get("mac_weight", 0)) for x in policy["modules"].values())
    mac_ratio = (
        sum(int(x.get("mac_weight", 0)) * float(x.get("ratio_projected", 0.0)) for x in policy["modules"].values())
        / max(mac_total, 1)
    )
    violations = []
    for name, cfg in policy["modules"].items():
        if name not in CAP_MODULES:
            continue
        cap = _ratio_from_cfg(CAP_MODULES[name])
        ratio = float(cfg.get("ratio_projected", 0.0))
        if ratio > cap + 1e-12:
            violations.append({"module_name": name, "ratio_projected": ratio, "cap": cap})

    policy["summary"] = {
        "module_count": len(policy["modules"]),
        "mean_projected_ratio_unweighted": float(np.mean(ratios)) if ratios else 0.0,
        "mac_weighted_projected_ratio": float(mac_ratio),
        "zero_ratio_module_count": sum(r == 0.0 for r in ratios),
        "split_module_count": sum(r > 0.0 for r in ratios),
        "ratio_cap_violation_count": len(violations),
    }
    policy.setdefault("metadata", {})
    policy["metadata"].update({
        "postprocess": "grouped_lowratio_capped_v3",
        "ratio_cap_policy": getattr(args, "ratio_cap_policy", ""),
        "group_qkv": bool(args.group_qkv),
        "group_gate_up": bool(args.group_gate_up),
        "tiny_ratio_zero_threshold": float(args.tiny_ratio_zero_threshold),
        "note": "Group ratios are capped by the minimum v1 reference ratio among group members.",
    })
    details = {
        "tiny_zero_changes": tiny_changes,
        "individual_cap_changes": individual_cap_changes,
        "group_changes": group_changes,
        "cap_violations": violations,
        "summary": policy["summary"],
    }
    return policy, details


def postprocess_output(args):
    out = Path(args.out_dir)
    policy_path = out / "policy.json"
    raw_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    capped_policy, details = apply_capped_group_constraints(raw_policy, args)
    policy_path.write_text(json.dumps(capped_policy, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "policy_grouped_capped_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    V2.write_policy_summary_csv(out / "policy_summary.csv", capped_policy)
    print("[GROUPED_CAPPED_V3_SUMMARY] " + json.dumps(details["summary"], ensure_ascii=False), flush=True)
    print(f"[GROUPED_CAPPED_V3_DETAILS] {out / 'policy_grouped_capped_details.json'}", flush=True)


def main():
    V2.V1.V6.parse_args = parse_args
    V2.V1.V6.optimize_one_linear = optimize_one_linear_capped
    V2.V1.V6.main()
    if LAST_ARGS is None:
        raise RuntimeError("parse_args was not called")
    postprocess_output(LAST_ARGS)


if __name__ == "__main__":
    main()
