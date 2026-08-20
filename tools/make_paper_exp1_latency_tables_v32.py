#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(row, key, default=0.0):
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qwen_prefill_csv", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(Path(args.qwen_prefill_csv))

    per_layer = []
    for row in rows:
        bf16 = f(row, "bf16_ms")
        quarot = f(row, "pure_current_ms")
        romeo = f(row, "romeo_qfactory_ms")
        split = f(row, "split_policy_qfactory_ms")
        split_extra = split - romeo
        out = {
            "workload": "prefill",
            "model": row.get("model", "Qwen/Qwen3-8B"),
            "layer_idx": int(row["layer_idx"]),
            "batch": int(row["batch"]),
            "seq_len": int(row["seq_len"]),
            "bf16_ms": bf16,
            "quarot_w4a4_ms": quarot,
            "romeo_w4a4_ms": romeo,
            "split_ms": split,
            "split_extra_vs_romeo_ms": split_extra,
            "quarot_over_bf16": quarot / bf16,
            "romeo_over_bf16": romeo / bf16,
            "split_over_bf16": split / bf16,
            "split_over_romeo": split / romeo,
            "split_extra_over_romeo": split_extra / romeo,
            "policy_mean_ratio": f(row, "split_policy_qfactory_mean_ratio"),
            "policy_max_ratio": f(row, "split_policy_qfactory_max_ratio"),
            "policy_nonzero_modules": f(row, "split_policy_qfactory_nonzero_modules"),
            "policy_sum_R": f(row, "split_policy_qfactory_sum_R"),
            "timing": row.get("timing", "cuda_graph_events"),
            "note": (
                "prefill layer latency; QuaRot=current pure W4A4 kernel; "
                "RoMeo=qfactory W4A4 kernel; Split=policy qfactory W4A4+sparse correction"
            ),
        }
        per_layer.append(out)

    summary = []
    for batch in sorted({r["batch"] for r in per_layer}):
        part = [r for r in per_layer if r["batch"] == batch]
        def avg(key):
            return sum(float(r[key]) for r in part) / len(part)
        summary.append(
            {
                "workload": "prefill",
                "model": part[0]["model"],
                "batch": batch,
                "seq_len": part[0]["seq_len"],
                "layers": len(part),
                "bf16_ms_mean": avg("bf16_ms"),
                "quarot_w4a4_ms_mean": avg("quarot_w4a4_ms"),
                "romeo_w4a4_ms_mean": avg("romeo_w4a4_ms"),
                "split_ms_mean": avg("split_ms"),
                "split_extra_vs_romeo_ms_mean": avg("split_extra_vs_romeo_ms"),
                "quarot_over_bf16_mean": avg("quarot_over_bf16"),
                "romeo_over_bf16_mean": avg("romeo_over_bf16"),
                "split_over_bf16_mean": avg("split_over_bf16"),
                "split_over_romeo_mean": avg("split_over_romeo"),
                "split_extra_over_romeo_mean": avg("split_extra_over_romeo"),
                "policy_mean_ratio_mean": avg("policy_mean_ratio"),
                "policy_nonzero_modules_mean": avg("policy_nonzero_modules"),
                "policy_sum_R_mean": avg("policy_sum_R"),
            }
        )

    extremes = []
    for batch in sorted({r["batch"] for r in per_layer}):
        part = [r for r in per_layer if r["batch"] == batch]
        for kind, key, reverse in (
            ("worst_split_over_romeo", "split_over_romeo", True),
            ("best_split_over_romeo", "split_over_romeo", False),
            ("worst_split_extra_ms", "split_extra_vs_romeo_ms", True),
        ):
            r = sorted(part, key=lambda x: float(x[key]), reverse=reverse)[0]
            extremes.append(
                {
                    "workload": "prefill",
                    "batch": batch,
                    "kind": kind,
                    "layer_idx": r["layer_idx"],
                    "bf16_ms": r["bf16_ms"],
                    "quarot_w4a4_ms": r["quarot_w4a4_ms"],
                    "romeo_w4a4_ms": r["romeo_w4a4_ms"],
                    "split_ms": r["split_ms"],
                    "split_over_romeo": r["split_over_romeo"],
                    "split_extra_vs_romeo_ms": r["split_extra_vs_romeo_ms"],
                    "policy_mean_ratio": r["policy_mean_ratio"],
                    "policy_max_ratio": r["policy_max_ratio"],
                }
            )

    write_csv(out_dir / "exp1_qwen3_prefill_per_layer_split_romeo_quarot_v32.csv", per_layer)
    write_csv(out_dir / "exp1_qwen3_prefill_summary_split_romeo_quarot_v32.csv", summary)
    write_csv(out_dir / "exp1_qwen3_prefill_extremes_split_romeo_quarot_v32.csv", extremes)
    (out_dir / "exp1_qwen3_prefill_split_romeo_quarot_v32.json").write_text(
        json.dumps(
            {
                "source": args.qwen_prefill_csv,
                "per_layer": per_layer,
                "summary": summary,
                "extremes": extremes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
