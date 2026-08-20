#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import math
from pathlib import Path

RATIO_KEYS = ["ratio_projected", "ratio_continuous", "ratio", "projected_ratio", "used_ratio", "split_ratio"]
R_KEYS = ["R", "projected_R", "split_R", "tail_R"]
K_KEYS = ["K", "k", "in_features", "in_dim"]
N_KEYS = ["N", "n", "out_features", "out_dim"]
BODY_KEYS = ["activation_percentile", "a_percentile", "act_percentile", "body_activation_percentile", "activation_body_percentile", "a_body_percentile", "body_percentile"]
TAIL_KEYS = ["tail_activation_percentile", "activation_tail_percentile", "a_tail_percentile", "tail_percentile"]
W_KEYS = ["weight_percentile", "w_percentile", "weight_clip_percentile", "w_clip_percentile"]

def as_int(x, default=None):
    try: return int(x)
    except Exception: return default

def as_float(x, default=0.0):
    try: return float(x)
    except Exception: return default

def get_first(d, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def iter_entries(obj):
    if isinstance(obj, dict) and isinstance(obj.get("modules"), dict):
        for name, e in obj["modules"].items():
            if isinstance(e, dict): yield str(name), e
    elif isinstance(obj, dict) and isinstance(obj.get("modules"), list):
        for i, e in enumerate(obj["modules"]):
            if isinstance(e, dict): yield str(e.get("name", i)), e
    elif isinstance(obj, list):
        for i, e in enumerate(obj):
            if isinstance(e, dict): yield str(e.get("name", i)), e
    elif isinstance(obj, dict):
        for name, e in obj.items():
            if isinstance(e, dict): yield str(name), e

def set_ratio(e, ratio):
    K = as_int(get_first(e, K_KEYS), None)
    ratio = float(ratio)
    for k in RATIO_KEYS:
        if k in e:
            e[k] = ratio
    e["ratio_projected"] = ratio
    e["ratio_continuous"] = ratio
    R = int(math.ceil(K * ratio)) if K else 0
    for k in R_KEYS:
        if k in e:
            e[k] = R
    if K and "R" not in e:
        e["R"] = R
    return R

def set_percentiles(e, body_p, tail_p, w_p):
    if body_p is not None:
        e["activation_percentile"] = float(body_p)
    if tail_p is not None:
        e["tail_activation_percentile"] = float(tail_p)
    if w_p is not None:
        e["weight_percentile"] = float(w_p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_policy", required=True)
    ap.add_argument("--shapes_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ratios", default="0.00125,0.0025,0.005,0.01,0.02,0.04,0.08")
    ap.add_argument("--body_percentile", type=float, default=99.75)
    ap.add_argument("--tail_percentile", type=float, default=100.0)
    ap.add_argument("--weight_percentile", type=float, default=99.75)
    args = ap.parse_args()

    base = json.load(open(args.base_policy))
    shapes = list(csv.DictReader(open(args.shapes_csv)))
    ratios = [float(x) for x in args.ratios.replace(" ", "").split(",") if x != ""]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    # zero-all baseline policy
    obj0 = copy.deepcopy(base)
    zero_R_sum = 0
    for _, e in iter_entries(obj0):
        zero_R_sum += set_ratio(e, 0.0)
        set_percentiles(e, args.body_percentile, args.tail_percentile, args.weight_percentile)
    if isinstance(obj0, dict):
        obj0["shape_latency_profile_meta"] = {
            "profile_type": "zero_all",
            "target_shape_id": "zero_all",
            "target_K": "",
            "target_N": "",
            "ratio": 0.0,
            "body_percentile": args.body_percentile,
            "tail_percentile": args.tail_percentile,
            "weight_percentile": args.weight_percentile,
        }
    zero_path = out_dir / "zero_all.json"
    json.dump(obj0, open(zero_path, "w"), indent=2, sort_keys=True)
    rows.append({
        "tag": "zero_all",
        "shape_id": "zero_all",
        "K": "-",
        "N": "-",
        "ratio": "0",
        "policy": str(zero_path),
        "target_module_count": "0",
        "target_R_sum": str(zero_R_sum),
    })

    for s in shapes:
        shape_id = s["shape_id"]
        K = int(float(s["K"]))
        N = int(float(s["N"]))
        for ratio in ratios:
            obj = copy.deepcopy(base)
            target_count = 0
            target_R_sum = 0
            total_R_sum = 0
            for _, e in iter_entries(obj):
                ek = as_int(get_first(e, K_KEYS), None)
                en = as_int(get_first(e, N_KEYS), None)
                r = ratio if (ek == K and en == N) else 0.0
                R = set_ratio(e, r)
                total_R_sum += R
                if ek == K and en == N:
                    target_count += 1
                    target_R_sum += R
                set_percentiles(e, args.body_percentile, args.tail_percentile, args.weight_percentile)
            tag = f"{shape_id}_r{str(ratio).replace('.', 'p')}"
            if isinstance(obj, dict):
                obj["shape_latency_profile_meta"] = {
                    "profile_type": "shape_isolation",
                    "target_shape_id": shape_id,
                    "target_K": K,
                    "target_N": N,
                    "ratio": ratio,
                    "target_module_count": target_count,
                    "target_R_sum": target_R_sum,
                    "total_R_sum": total_R_sum,
                    "body_percentile": args.body_percentile,
                    "tail_percentile": args.tail_percentile,
                    "weight_percentile": args.weight_percentile,
                }
            p = out_dir / f"{tag}.json"
            json.dump(obj, open(p, "w"), indent=2, sort_keys=True)
            rows.append({
                "tag": tag,
                "shape_id": shape_id,
                "K": str(K),
                "N": str(N),
                "ratio": str(ratio),
                "policy": str(p),
                "target_module_count": str(target_count),
                "target_R_sum": str(target_R_sum),
            })

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag","shape_id","K","N","ratio","policy","target_module_count","target_R_sum"], delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] policies={len(rows)}")
    print(f"[MANIFEST] {manifest}")

if __name__ == "__main__":
    main()
