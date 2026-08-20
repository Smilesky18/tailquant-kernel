import csv
import json
import argparse
from pathlib import Path


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    if x is None or x == "":
        return None
    return float(x)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--continuous_dir", required=True)
    ap.add_argument("--projected_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cont_dir = Path(args.continuous_dir)
    proj_dir = Path(args.projected_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cont_path = cont_dir / "layer_latency_all_v22.csv"
    proj_path = proj_dir / "layer_latency_all_v22.csv"

    if not cont_path.exists():
        raise FileNotFoundError(f"continuous result not found: {cont_path}")
    if not proj_path.exists():
        raise FileNotFoundError(f"projected result not found: {proj_path}")

    cont = read_csv(cont_path)
    proj = read_csv(proj_path)

    cont_map = {(int(r["layer_idx"]), int(r["batch"])): r for r in cont}
    proj_map = {(int(r["layer_idx"]), int(r["batch"])): r for r in proj}

    rows = []
    for key in sorted(set(cont_map) & set(proj_map)):
        c = cont_map[key]
        p = proj_map[key]
        layer_idx, batch = key

        c_ms = f(c["split_policy_qfactory_ms"])
        p_ms = f(p["split_policy_qfactory_ms"])

        row = {
            "layer_idx": layer_idx,
            "batch": batch,
            "seq_len": int(c["seq_len"]),
            "bf16_ms": f(c["bf16_ms"]),
            "pure_w4a4_qfactory_ms": f(c["pure_w4a4_qfactory_ms"]),
            "split_fixed_qfactory_ms": f(c["split_fixed_qfactory_ms"]),

            "continuous_policy_ms": c_ms,
            "projected_policy_ms": p_ms,
            "projected_minus_continuous_ms": p_ms - c_ms,
            "projected_over_continuous": p_ms / c_ms if c_ms and c_ms > 0 else None,
            "continuous_speedup_over_fixed": f(c["split_fixed_qfactory_ms"]) / c_ms,
            "projected_speedup_over_fixed": f(p["split_fixed_qfactory_ms"]) / p_ms,

            "continuous_over_pure": c_ms / f(c["pure_w4a4_qfactory_ms"]),
            "projected_over_pure": p_ms / f(p["pure_w4a4_qfactory_ms"]),

            "continuous_policy_avg_ratio": f(c["policy_avg_ratio"]),
            "projected_policy_avg_ratio": f(p["policy_avg_ratio"]),
            "continuous_policy_mac_weighted_ratio": f(c["policy_mac_weighted_ratio"]),
            "projected_policy_mac_weighted_ratio": f(p["policy_mac_weighted_ratio"]),
            "continuous_policy_max_ratio": f(c["policy_max_ratio"]),
            "projected_policy_max_ratio": f(p["policy_max_ratio"]),
        }
        rows.append(row)

    write_csv(out_dir / "continuous_vs_projected_layer_compare_v22b.csv", rows)
    json.dump(rows, open(out_dir / "continuous_vs_projected_layer_compare_v22b.json", "w"), indent=2, ensure_ascii=False)

    summary_rows = []
    for batch in sorted({int(r["batch"]) for r in rows}):
        rs = [r for r in rows if int(r["batch"]) == batch]

        def s(k):
            return sum(float(r[k]) for r in rs)

        sum_fixed = s("split_fixed_qfactory_ms")
        sum_pure = s("pure_w4a4_qfactory_ms")
        sum_bf16 = s("bf16_ms")
        sum_cont = s("continuous_policy_ms")
        sum_proj = s("projected_policy_ms")

        summary = {
            "batch": batch,
            "num_layers": len(rs),
            "sum_bf16_ms": sum_bf16,
            "sum_pure_w4a4_qfactory_ms": sum_pure,
            "sum_split_fixed_qfactory_ms": sum_fixed,
            "sum_continuous_policy_ms": sum_cont,
            "sum_projected_policy_ms": sum_proj,

            "continuous_policy_over_bf16": sum_cont / sum_bf16,
            "projected_policy_over_bf16": sum_proj / sum_bf16,
            "continuous_policy_over_pure": sum_cont / sum_pure,
            "projected_policy_over_pure": sum_proj / sum_pure,
            "continuous_policy_over_fixed": sum_cont / sum_fixed,
            "projected_policy_over_fixed": sum_proj / sum_fixed,
            "continuous_speedup_over_fixed": sum_fixed / sum_cont,
            "projected_speedup_over_fixed": sum_fixed / sum_proj,

            "projected_minus_continuous_ms": sum_proj - sum_cont,
            "projected_over_continuous": sum_proj / sum_cont,
            "projected_slowdown_vs_continuous_pct": (sum_proj / sum_cont - 1.0) * 100.0,

            "avg_continuous_mac_weighted_ratio": sum(r["continuous_policy_mac_weighted_ratio"] for r in rs) / len(rs),
            "avg_projected_mac_weighted_ratio": sum(r["projected_policy_mac_weighted_ratio"] for r in rs) / len(rs),
        }
        summary_rows.append(summary)

    write_csv(out_dir / "continuous_vs_projected_summary_v22b.csv", summary_rows)
    json.dump(summary_rows, open(out_dir / "continuous_vs_projected_summary_v22b.json", "w"), indent=2, ensure_ascii=False)

    top_slow = sorted(rows, key=lambda r: r["projected_minus_continuous_ms"], reverse=True)[:10]
    write_csv(out_dir / "top10_projected_slower_layers_v22b.csv", top_slow)

    print("[COMPARE_CSV]", out_dir / "continuous_vs_projected_layer_compare_v22b.csv", flush=True)
    print("[SUMMARY_CSV]", out_dir / "continuous_vs_projected_summary_v22b.csv", flush=True)
    print("[TOP10_CSV]", out_dir / "top10_projected_slower_layers_v22b.csv", flush=True)

    for r in summary_rows:
        print("[COMPARE_SUMMARY] " + json.dumps(r, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
