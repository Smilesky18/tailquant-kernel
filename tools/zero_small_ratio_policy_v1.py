#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.01)
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.input)
    dst = Path(args.output)
    policy = json.loads(src.read_text(encoding="utf-8"))
    modules = policy.get("modules", {})

    changed = []
    for name, cfg in modules.items():
        ratio = float(cfg.get("ratio_projected", 0.0))
        if 0.0 < ratio < args.threshold:
            changed.append({
                "module_name": name,
                "old_ratio_projected": ratio,
                "old_ratio_continuous": float(cfg.get("ratio_continuous", 0.0)),
            })
            cfg["ratio_projected"] = 0.0
            cfg["ratio_continuous"] = 0.0

    ratios = [float(cfg.get("ratio_projected", 0.0)) for cfg in modules.values()]
    mac_total = sum(int(cfg.get("mac_weight", 0)) for cfg in modules.values())
    mac_weighted = (
        sum(int(cfg.get("mac_weight", 0)) * float(cfg.get("ratio_projected", 0.0)) for cfg in modules.values())
        / max(mac_total, 1)
    )
    policy["summary"] = {
        "module_count": len(modules),
        "mean_projected_ratio_unweighted": sum(ratios) / max(len(ratios), 1),
        "mac_weighted_projected_ratio": mac_weighted,
        "zero_ratio_module_count": sum(r == 0.0 for r in ratios),
        "split_module_count": sum(r > 0.0 for r in ratios),
    }
    policy.setdefault("metadata", {})
    policy["metadata"]["derived_from_policy"] = str(src)
    policy["metadata"]["small_ratio_zero_threshold"] = float(args.threshold)
    policy["metadata"]["small_ratio_zero_changed_count"] = len(changed)
    policy["small_ratio_zero_changes"] = changed

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "input": str(src),
        "output": str(dst),
        "threshold": args.threshold,
        "changed_count": len(changed),
        "summary": policy["summary"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
