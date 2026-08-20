#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


QWEN3_MODULES = (
    ("self_attn.q_proj", "q_proj"),
    ("self_attn.k_proj", "k_proj"),
    ("self_attn.v_proj", "v_proj"),
    ("self_attn.o_proj", "o_proj"),
    ("mlp.gate_proj", "gate_proj"),
    ("mlp.up_proj", "up_proj"),
    ("mlp.down_proj", "down_proj"),
)


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


def load_cutlass_dense(cutlass_csv: Path):
    rows = read_csv(cutlass_csv)
    by_batch_shape = {}
    for row in rows:
        batch = int(row["batch"])
        shape = row["shape_name"]
        by_batch_shape[(batch, shape)] = f(row, "avg_runtime_ms")

    dense_by_batch = {}
    missing = []
    for batch in sorted({int(r["batch"]) for r in rows}):
        total = 0.0
        module_rows = []
        for module, shape in QWEN3_MODULES:
            key = (batch, shape)
            if key not in by_batch_shape:
                missing.append(key)
                continue
            ms = by_batch_shape[key]
            total += ms
            module_rows.append(
                {
                    "batch": batch,
                    "module": module,
                    "shape_name": shape,
                    "cutlass_fp4_dense_ms": ms,
                }
            )
        dense_by_batch[batch] = {"total_ms": total, "modules": module_rows}
    if missing:
        raise RuntimeError(f"missing CUTLASS rows: {missing}")
    return dense_by_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_layer_csv", required=True)
    parser.add_argument("--cutlass_e0m3_csv", required=True)
    parser.add_argument("--cutlass_nvfp4_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_rows = read_csv(Path(args.qwen_layer_csv))
    e0m3_dense = load_cutlass_dense(Path(args.cutlass_e0m3_csv))
    nvfp4_dense = load_cutlass_dense(Path(args.cutlass_nvfp4_csv))

    projected = []
    for row in layer_rows:
        batch = int(row["batch"])
        bf16 = f(row, "bf16_ms")
        qfactory = f(row, "romeo_qfactory_ms")
        split = f(row, "split_policy_qfactory_ms")
        sparse_overhead = split - qfactory
        e0m3_pure = e0m3_dense[batch]["total_ms"]
        nvfp4_pure = nvfp4_dense[batch]["total_ms"]
        e0m3_split = e0m3_pure + sparse_overhead
        nvfp4_split = nvfp4_pure + sparse_overhead
        out = {
            "model": row.get("model", "Qwen/Qwen3-8B"),
            "layer_idx": int(row["layer_idx"]),
            "batch": batch,
            "seq_len": int(row["seq_len"]),
            "bf16_ms": bf16,
            "qfactory_w4a4_ms": qfactory,
            "split_qfactory_ms": split,
            "measured_sparse_extra_ms": sparse_overhead,
            "cutlass_e0m3_pure_dense_7linear_ms": e0m3_pure,
            "cutlass_e0m3_split_projected_ms": e0m3_split,
            "cutlass_nvfp4_pure_dense_7linear_ms": nvfp4_pure,
            "cutlass_nvfp4_split_projected_ms": nvfp4_split,
            "e0m3_pure_over_bf16": e0m3_pure / bf16,
            "e0m3_split_projected_over_bf16": e0m3_split / bf16,
            "nvfp4_pure_over_bf16": nvfp4_pure / bf16,
            "nvfp4_split_projected_over_bf16": nvfp4_split / bf16,
            "qfactory_w4a4_over_bf16": qfactory / bf16,
            "split_qfactory_over_bf16": split / bf16,
            "split_extra_over_qfactory": sparse_overhead / qfactory if qfactory else 0.0,
            "policy_mean_ratio": f(row, "split_policy_qfactory_mean_ratio"),
            "policy_max_ratio": f(row, "split_policy_qfactory_max_ratio"),
            "policy_nonzero_modules": f(row, "split_policy_qfactory_nonzero_modules"),
            "policy_sum_R": f(row, "split_policy_qfactory_sum_R"),
            "note": (
                "smoke_projection_only: CUTLASS fp4/e0m3 dense binary is not "
                "integrated as a PyTorch Linear; sparse_extra_ms is measured "
                "from qfactory split minus qfactory dense."
            ),
        }
        projected.append(out)

    summary = []
    for batch in sorted({int(r["batch"]) for r in projected}):
        rows = [r for r in projected if int(r["batch"]) == batch]
        def avg(key):
            return sum(float(r[key]) for r in rows) / len(rows)
        summary.append(
            {
                "batch": batch,
                "layers": len(rows),
                "bf16_ms_mean": avg("bf16_ms"),
                "qfactory_w4a4_ms_mean": avg("qfactory_w4a4_ms"),
                "split_qfactory_ms_mean": avg("split_qfactory_ms"),
                "measured_sparse_extra_ms_mean": avg("measured_sparse_extra_ms"),
                "cutlass_e0m3_pure_dense_ms": e0m3_dense[batch]["total_ms"],
                "cutlass_e0m3_split_projected_ms_mean": avg("cutlass_e0m3_split_projected_ms"),
                "cutlass_nvfp4_pure_dense_ms": nvfp4_dense[batch]["total_ms"],
                "cutlass_nvfp4_split_projected_ms_mean": avg("cutlass_nvfp4_split_projected_ms"),
                "e0m3_pure_over_bf16_mean": avg("e0m3_pure_over_bf16"),
                "e0m3_split_projected_over_bf16_mean": avg("e0m3_split_projected_over_bf16"),
                "nvfp4_pure_over_bf16_mean": avg("nvfp4_pure_over_bf16"),
                "nvfp4_split_projected_over_bf16_mean": avg("nvfp4_split_projected_over_bf16"),
                "split_qfactory_over_bf16_mean": avg("split_qfactory_over_bf16"),
                "projection_note": "optimistic smoke, not integrated kernel timing",
            }
        )

    write_csv(out_dir / "qwen3_split_cutlass_fp4_projection_smoke_v31.csv", projected)
    write_csv(out_dir / "qwen3_split_cutlass_fp4_projection_smoke_summary_v31.csv", summary)
    (out_dir / "qwen3_split_cutlass_fp4_projection_smoke_v31.json").write_text(
        json.dumps({"rows": projected, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
