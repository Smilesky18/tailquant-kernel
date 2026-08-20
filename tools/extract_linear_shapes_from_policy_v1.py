#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

RATIO_KEYS = ["ratio_projected", "ratio_continuous", "ratio", "projected_ratio", "used_ratio", "split_ratio"]
R_KEYS = ["R", "projected_R", "split_R", "tail_R"]
K_KEYS = ["K", "k", "in_features", "in_dim"]
N_KEYS = ["N", "n", "out_features", "out_dim"]

def as_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def as_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def get_first(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default

def get_ratio(e):
    for k in RATIO_KEYS:
        if k in e:
            return as_float(e[k], 0.0)
    K = as_int(get_first(e, K_KEYS), None)
    R = as_float(get_first(e, R_KEYS), None)
    if K and R is not None:
        return float(R) / float(K)
    return 0.0

def iter_entries(obj):
    if isinstance(obj, dict) and isinstance(obj.get("modules"), dict):
        for name, e in obj["modules"].items():
            if isinstance(e, dict):
                yield str(name), e
    elif isinstance(obj, dict) and isinstance(obj.get("modules"), list):
        for i, e in enumerate(obj["modules"]):
            if isinstance(e, dict):
                yield str(e.get("name", i)), e
    elif isinstance(obj, list):
        for i, e in enumerate(obj):
            if isinstance(e, dict):
                yield str(e.get("name", i)), e
    elif isinstance(obj, dict):
        for name, e in obj.items():
            if isinstance(e, dict):
                yield str(name), e

def parse_layer_and_module(name):
    # Common names:
    # model.layers.18.self_attn.o_proj, layers.18.mlp.down_proj, layer18...
    layer_idx = ""
    m = re.search(r"(?:layers?|layer)[._]?(\d+)", name)
    if m:
        layer_idx = m.group(1)

    module_type = name.split(".")[-1] if "." in name else name
    for cand in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]:
        if cand in name:
            module_type = cand
            break
    return layer_idx, module_type

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_shape_csv", required=True)
    args = ap.parse_args()

    policy = Path(args.policy)
    obj = json.load(open(policy))

    rows = []
    for idx, (name, e) in enumerate(iter_entries(obj)):
        K = as_int(get_first(e, K_KEYS), None)
        N = as_int(get_first(e, N_KEYS), None)
        if K is None or N is None:
            continue
        ratio = get_ratio(e)
        R = as_int(get_first(e, R_KEYS), math.ceil(K * ratio))
        layer_idx, module_type = parse_layer_and_module(name)
        shape_id = f"K{K}_N{N}"
        rows.append({
            "idx": idx,
            "name": name,
            "layer_idx": layer_idx,
            "module_type": module_type,
            "K": K,
            "N": N,
            "shape_id": shape_id,
            "base_ratio": ratio,
            "base_R": R,
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx","name","layer_idx","module_type","K","N","shape_id","base_ratio","base_R"])
        w.writeheader()
        w.writerows(rows)

    shape_map = {}
    for r in rows:
        sid = r["shape_id"]
        s = shape_map.setdefault(sid, {
            "shape_id": sid,
            "K": r["K"],
            "N": r["N"],
            "module_count": 0,
            "module_types": set(),
            "layers": set(),
            "base_ratio_mean_num": 0.0,
            "base_R_sum": 0,
        })
        s["module_count"] += 1
        s["module_types"].add(r["module_type"])
        if r["layer_idx"] != "":
            s["layers"].add(r["layer_idx"])
        s["base_ratio_mean_num"] += float(r["base_ratio"])
        s["base_R_sum"] += int(r["base_R"])

    shape_rows = []
    for s in shape_map.values():
        shape_rows.append({
            "shape_id": s["shape_id"],
            "K": s["K"],
            "N": s["N"],
            "module_count": s["module_count"],
            "module_types": ",".join(sorted(s["module_types"])),
            "layers": ",".join(sorted(s["layers"], key=lambda x: int(x) if str(x).isdigit() else 10**9)),
            "base_ratio_mean": s["base_ratio_mean_num"] / max(s["module_count"], 1),
            "base_R_sum": s["base_R_sum"],
        })

    out_shape = Path(args.out_shape_csv)
    with open(out_shape, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shape_id","K","N","module_count","module_types","layers","base_ratio_mean","base_R_sum"])
        w.writeheader()
        w.writerows(sorted(shape_rows, key=lambda x: (int(x["K"]), int(x["N"]))))

    print(f"[OK] modules={len(rows)} unique_shapes={len(shape_rows)}")
    print(f"[OUT] {out_csv}")
    print(f"[OUT_SHAPES] {out_shape}")

if __name__ == "__main__":
    main()
