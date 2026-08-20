#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matrix-only dense/sparse breakdown for the v61 Split compute path.

This benchmark deliberately excludes model-layer concerns:
  - no Hadamard rotation
  - no q/k/v or gate/up shared prepare
  - no model forward

It times two compute components on synthetic matrices:
  1. dense int4 GEMM: quarot_dense_gemm(qext, A_pack, B_col)
  2. sparse correction: v58 scale_sparse_epilogue_oct with C_body_i32 = 0

The sparse timing still includes the epilogue kernel's output write and scale
loads, but the dense contribution is zeroed so the measured work is dominated by
top-R sparse correction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
for item in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import bench_full_model_policy_workspace_v32 as BASE  # noqa: E402
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension  # noqa: E402


FULL_M_VALUES = [256, 512, 1024, 2048, 8192, 32768]
FULL_SHAPES = [
    (4096, 4096),
    (4096, 11008),
    (11008, 4096),
    (5120, 5120),
    (5120, 13824),
    (13824, 5120),
    (4096, 1024),
    (4096, 14336),
    (14336, 4096),
    (4096, 12288),
    (12288, 4096),
]
FULL_RATIOS = [0.0025, 0.0050, 0.0075, 0.01, 0.0125, 0.02]


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_shapes(text: str) -> list[tuple[int, int]]:
    shapes = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.lower().split("x")
        shapes.append((int(left), int(right)))
    return shapes


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def ceil_ratio_count(k: int, ratio: float) -> int:
    if ratio <= 0.0:
        return 0
    return min(k - 1, max(1, int(math.ceil(k * ratio))))


def time_cuda(fn, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        out = fn()
        if torch.is_tensor(out):
            out.record_stream(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
        if torch.is_tensor(out):
            out.record_stream(torch.cuda.current_stream(device))
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / max(iters, 1))


def make_case_tensors(
    *,
    main_ext,
    M: int,
    K: int,
    N: int,
    R: int,
    device: torch.device,
    eps: float,
) -> dict:
    if K % 2 != 0:
        raise ValueError(f"K must be even for int4 packing: {K}")
    if N % 2 != 0:
        raise ValueError(f"N must be even for int4 packing: {N}")

    # A_pack is already the post-prepare dense body input; no Hadamard/topk work.
    A_pack = torch.randint(0, 256, ((M * K + 1) // 2,), dtype=torch.uint8, device=device)

    weight = torch.randn(N, K, dtype=torch.float16, device=device)
    W_T = weight.t().contiguous()
    weight_bytes = (K * N + 1) // 2
    B_col = torch.empty(weight_bytes, dtype=torch.uint8, device=device)
    B_row = torch.empty(weight_bytes, dtype=torch.uint8, device=device)
    w_scale = torch.empty(N, dtype=torch.float32, device=device)
    main_ext.pack_weight_colmajor_s4(W_T, B_col, w_scale, eps)
    main_ext.pack_weight_rowmajor_s4_from_scale(W_T, B_row, w_scale)
    del weight, W_T

    C_zero = torch.zeros((M, N), dtype=torch.int32, device=device)
    body_scale = torch.ones(M, dtype=torch.float32, device=device)
    top_scale = torch.ones(M, dtype=torch.float32, device=device)
    output = torch.empty((M, N), dtype=torch.float16, device=device)

    indices = torch.randint(0, K, (M, R), dtype=torch.int32, device=device)
    top_q = torch.randint(-8, 8, (M, R), dtype=torch.int8, device=device)
    torch.cuda.synchronize(device)
    return {
        "A_pack": A_pack,
        "B_col": B_col,
        "B_row": B_row,
        "w_scale": w_scale,
        "C_zero": C_zero,
        "body_scale": body_scale,
        "top_scale": top_scale,
        "output": output,
        "indices": indices,
        "top_q": top_q,
    }


def bench_case(
    *,
    qext,
    epilogue_ext,
    main_ext,
    M: int,
    K: int,
    N: int,
    ratio: float,
    warmup: int,
    iters: int,
    device: torch.device,
    eps: float,
) -> dict:
    R = ceil_ratio_count(K, ratio)
    tensors = make_case_tensors(
        main_ext=main_ext,
        M=M,
        K=K,
        N=N,
        R=R,
        device=device,
        eps=eps,
    )

    def dense_fn():
        A_view = tensors["A_pack"].view(M, K // 2).contiguous()
        B_view = tensors["B_col"].view(N, K // 2).contiguous()
        try:
            return qext.matmul(A_view, B_view, M, N, K)
        except TypeError:
            return qext.matmul(A_view, B_view)

    def sparse_fn():
        epilogue_ext.scale_sparse_epilogue_oct(
            tensors["C_zero"],
            tensors["body_scale"],
            tensors["top_q"],
            tensors["indices"],
            tensors["B_row"],
            tensors["top_scale"],
            tensors["w_scale"],
            tensors["output"],
            K,
        )
        return tensors["output"]

    dense_ms = time_cuda(dense_fn, warmup, iters, device)
    sparse_ms = time_cuda(sparse_fn, warmup, iters, device)
    total_ms = dense_ms + sparse_ms
    return {
        "M": M,
        "K": K,
        "N": N,
        "ratio": ratio,
        "R": R,
        "dense_gemm_ms": dense_ms,
        "sparse_correction_ms": sparse_ms,
        "dense_plus_sparse_ms": total_ms,
        "dense_pct_of_two": dense_ms / max(total_ms, 1e-12) * 100.0,
        "sparse_pct_of_two": sparse_ms / max(total_ms, 1e-12) * 100.0,
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", str(ROOT / "third_party/cutlass")))
    parser.add_argument("--verbose_compile", action="store_true")
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--M_values", default="")
    parser.add_argument("--shapes", default="")
    parser.add_argument("--ratios", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        m_values = parse_csv_ints(args.M_values) if args.M_values else [256, 2048]
        shapes = parse_shapes(args.shapes) if args.shapes else [(4096, 4096), (4096, 11008)]
        ratios = parse_floats(args.ratios) if args.ratios else [0.005, 0.01]
        warmup = min(args.warmup, 3)
        iters = min(args.iters, 10)
    else:
        m_values = parse_csv_ints(args.M_values) if args.M_values else FULL_M_VALUES
        shapes = parse_shapes(args.shapes) if args.shapes else FULL_SHAPES
        ratios = parse_floats(args.ratios) if args.ratios else FULL_RATIOS
        warmup = args.warmup
        iters = args.iters

    qext = load_quarot_sm120_extension(verbose=args.verbose_compile)
    epilogue_ext = load_fused_sparse_epilogue_ext(verbose=args.verbose_compile)
    cutlass_path = BASE.find_cutlass_path(args.cutlass_path)
    main_ext = BASE.load_ext(cutlass_path, verbose=args.verbose_compile)

    rows = []
    for M in m_values:
        for K, N in shapes:
            for ratio in ratios:
                row = bench_case(
                    qext=qext,
                    epilogue_ext=epilogue_ext,
                    main_ext=main_ext,
                    M=M,
                    K=K,
                    N=N,
                    ratio=ratio,
                    warmup=warmup,
                    iters=iters,
                    device=device,
                    eps=args.eps,
                )
                rows.append(row)
                print("[DENSE_SPARSE_BREAKDOWN] " + json.dumps(row, ensure_ascii=False), flush=True)

    csv_path = out_dir / "dense_sparse_breakdown.csv"
    json_path = out_dir / "dense_sparse_breakdown.json"
    meta_path = out_dir / "dense_sparse_breakdown_meta.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "mode": "smoke" if args.smoke else "full",
                "m_values": m_values,
                "shapes": shapes,
                "ratios": ratios,
                "warmup": warmup,
                "iters": iters,
                "note": "dense is qext.matmul only; sparse is v58 oct epilogue with C_body_i32 zeroed.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("[OUTPUT_CSV] " + str(csv_path), flush=True)


if __name__ == "__main__":
    main()
