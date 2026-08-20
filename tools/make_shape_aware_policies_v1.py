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

def get_ratio(e):
    for k in RATIO_KEYS:
        if k in e:
            return max(0.0, as_float(e[k], 0.0))
    K = as_int(get_first(e, K_KEYS), None)
    R = as_float(get_first(e, R_KEYS), None)
    if K and R is not None:
        return max(0.0, float(R) / float(K))
    return 0.0

def set_ratio(e, ratio):
    ratio = max(0.0, float(ratio))
    K = as_int(get_first(e, K_KEYS), None)
    R = int(math.ceil(K * ratio)) if K else 0
    for k in RATIO_KEYS:
        if k in e:
            e[k] = ratio
    e["ratio_projected"] = ratio
    e["ratio_continuous"] = ratio
    for k in R_KEYS:
        if k in e:
            e[k] = R
    if K and "R" not in e:
        e["R"] = R
    return R

def set_percentiles(e, body_p, tail_p, w_p):
    e["activation_percentile"] = float(body_p)
    e["tail_activation_percentile"] = float(tail_p)
    e["weight_percentile"] = float(w_p)

def parse_weights(s):
    out = {}
    if not s:
        return out
    for item in s.split(","):
        if not item.strip():
            continue
        k, v = item.split(":")
        out[int(k)] = float(v)
    return out

def load_cost_table(path, batch_weights, ratios):
    rows = list(csv.DictReader(open(path)))
    # cost[(shape_id, ratio)] = weighted delta, using non-negative delta by default.
    cost = {}
    raw = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        sid = r["shape_id"]
        try:
            ratio = float(r["ratio"])
            batch = int(float(r["batch"]))
            delta = float(r["delta_ms"])
        except Exception:
            continue
        w = batch_weights.get(batch, 0.0)
        if w <= 0:
            continue
        raw.setdefault((sid, ratio), []).append((w, delta))
    for key, xs in raw.items():
        den = sum(w for w, _ in xs)
        val = sum(w * max(0.0, d) for w, d in xs) / max(den, 1e-12)
        cost[key] = val

    # normalize cost so lambdas are easier to sweep.
    max_cost = max(cost.values()) if cost else 1.0
    if max_cost <= 0:
        max_cost = 1.0
    cost_norm = {k: v / max_cost for k, v in cost.items()}

    # Ensure ratio=0 exists as zero cost.
    shapes = {sid for sid, _ in cost.keys()}
    for sid in list(shapes):
        cost[(sid, 0.0)] = 0.0
        cost_norm[(sid, 0.0)] = 0.0

    return cost, cost_norm, max_cost

def nearest_cost(cost_norm, sid, ratio):
    if ratio <= 0:
        return 0.0
    vals = [(abs(ratio - r), c) for (s, r), c in cost_norm.items() if s == sid]
    if not vals:
        return ratio
    vals.sort(key=lambda x: x[0])
    return vals[0][1]

def shape_id_of(e):
    K = as_int(get_first(e, K_KEYS), None)
    N = as_int(get_first(e, N_KEYS), None)
    if K is None or N is None:
        return None, K, N
    return f"K{K}_N{N}", K, N

def choose_ratio(base_r, sid, ratio_grid, lam, cost_norm, min_ratio_if_on, max_ratio_cap, importance_power):
    # Candidate ratios are only allowed up to base_r and max_ratio_cap.
    # This makes the policy a low-cost version of the accuracy-oriented search result,
    # not an arbitrary high-ratio policy.
    eps = 1e-6
    if base_r <= eps:
        return 0.0, {"acc_proxy": 0.0, "cost_proxy": 0.0, "score": 0.0}
    cap = min(float(base_r), float(max_ratio_cap))
    cands = [0.0]
    for r in ratio_grid:
        r = float(r)
        if r <= cap + 1e-12:
            if r == 0 or r >= min_ratio_if_on:
                cands.append(r)
    if cap > 0 and cap not in cands:
        cands.append(cap)
    cands = sorted(set(round(x, 8) for x in cands))

    best = None
    # importance grows with base_r: modules that original search wanted more ratio
    # pay a larger proxy penalty when aggressively reduced.
    importance = max(base_r, eps) ** float(importance_power)
    for r in cands:
        # Loss proxy is zero at base_r and increases as ratio is reduced.
        # ratio=0 is legal, but receives the largest proxy error.
        rel_gap = max(0.0, base_r - r) / max(base_r, eps)
        acc_proxy = importance * (rel_gap ** 2)
        cost_proxy = nearest_cost(cost_norm, sid, r)
        score = acc_proxy + float(lam) * cost_proxy
        item = (score, r, acc_proxy, cost_proxy)
        if best is None or item[0] < best[0]:
            best = item
    score, r, acc, cost = best
    return r, {"acc_proxy": acc, "cost_proxy": cost, "score": score}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_policy", required=True)
    ap.add_argument("--shape_latency_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--summary_tsv", required=True)
    ap.add_argument("--lambdas", default="0,0.03,0.1,0.3,1,3,10,30")
    ap.add_argument("--ratio_grid", default="0,0.00125,0.0025,0.005,0.01,0.02,0.04,0.08")
    ap.add_argument("--batch_weights", default="16:1,64:1,256:1")
    ap.add_argument("--body_percentile", type=float, default=99.75)
    ap.add_argument("--tail_percentile", type=float, default=100.0)
    ap.add_argument("--weight_percentile", type=float, default=99.75)
    ap.add_argument("--min_ratio_if_on", type=float, default=0.00125)
    ap.add_argument("--max_ratio_cap", type=float, default=0.08)
    ap.add_argument("--importance_power", type=float, default=1.0)
    args = ap.parse_args()

    base = json.load(open(args.base_policy))
    lambdas = [float(x) for x in args.lambdas.replace(" ", "").split(",") if x != ""]
    ratio_grid = [float(x) for x in args.ratio_grid.replace(" ", "").split(",") if x != ""]
    batch_weights = parse_weights(args.batch_weights)
    cost, cost_norm, max_cost = load_cost_table(args.shape_latency_csv, batch_weights, ratio_grid)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    # Always include a strict single-scale baseline.
    variants = [("single_scale", None)] + [(f"shapeaware_lam_{str(lam).replace('.', 'p')}", lam) for lam in lambdas]

    for tag, lam in variants:
        obj = copy.deepcopy(base)
        n = 0; nonzero = 0; sum_R = 0; sum_ratio = 0.0; sum_acc = 0.0; sum_cost = 0.0
        hist = {}
        examples = []
        for name, e in iter_entries(obj):
            sid, K, N = shape_id_of(e)
            if sid is None:
                continue
            base_r = get_ratio(e)
            if tag == "single_scale":
                new_r = 0.0
                info = {"acc_proxy": base_r ** 2, "cost_proxy": 0.0, "score": base_r ** 2}
            else:
                new_r, info = choose_ratio(
                    base_r=base_r,
                    sid=sid,
                    ratio_grid=ratio_grid,
                    lam=lam,
                    cost_norm=cost_norm,
                    min_ratio_if_on=args.min_ratio_if_on,
                    max_ratio_cap=args.max_ratio_cap,
                    importance_power=args.importance_power,
                )
            R = set_ratio(e, new_r)
            set_percentiles(e, args.body_percentile, args.tail_percentile, args.weight_percentile)
            e["_shape_aware_base_ratio"] = base_r
            e["_shape_aware_ratio"] = new_r
            e["_shape_aware_shape_id"] = sid
            e["_shape_aware_cost_proxy"] = info["cost_proxy"]
            e["_shape_aware_acc_proxy"] = info["acc_proxy"]

            n += 1
            nonzero += int(new_r > 0)
            sum_R += R
            sum_ratio += new_r
            sum_acc += info["acc_proxy"]
            sum_cost += info["cost_proxy"]
            key = f"{new_r:.6g}"
            hist[key] = hist.get(key, 0) + 1
            if len(examples) < 12:
                examples.append({"name": name, "shape_id": sid, "K": K, "N": N, "base_ratio": base_r, "new_ratio": new_r, "R": R, **info})

        meta = {
            "tag": tag,
            "lambda": lam,
            "base_policy": args.base_policy,
            "shape_latency_csv": args.shape_latency_csv,
            "batch_weights": args.batch_weights,
            "ratio_grid": ratio_grid,
            "body_percentile": args.body_percentile,
            "tail_percentile": args.tail_percentile,
            "weight_percentile": args.weight_percentile,
            "max_cost_ms_for_normalization": max_cost,
            "module_count": n,
            "nonzero_modules": nonzero,
            "mean_ratio": sum_ratio / max(n, 1),
            "sum_R": sum_R,
            "sum_acc_proxy": sum_acc,
            "sum_cost_proxy": sum_cost,
            "ratio_hist": hist,
            "examples": examples,
        }
        if isinstance(obj, dict):
            obj["shape_aware_search_meta"] = meta
        p = out_dir / f"{tag}.json"
        json.dump(obj, open(p, "w"), indent=2, sort_keys=True)
        summary_rows.append({
            "tag": tag,
            "lambda": "-" if lam is None else str(lam),
            "policy": str(p),
            "module_count": str(n),
            "nonzero_modules": str(nonzero),
            "mean_ratio": f"{meta['mean_ratio']:.9f}",
            "sum_R": str(sum_R),
            "sum_acc_proxy": f"{sum_acc:.9f}",
            "sum_cost_proxy": f"{sum_cost:.9f}",
            "ratio_hist": json.dumps(hist, sort_keys=True),
        })

    summary = Path(args.summary_tsv)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", newline="") as f:
        fieldnames = ["tag","lambda","policy","module_count","nonzero_modules","mean_ratio","sum_R","sum_acc_proxy","sum_cost_proxy","ratio_hist"]
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[OK] policies={len(summary_rows)}")
    print(f"[OUT_DIR] {out_dir}")
    print(f"[SUMMARY] {summary}")

if __name__ == "__main__":
    main()
