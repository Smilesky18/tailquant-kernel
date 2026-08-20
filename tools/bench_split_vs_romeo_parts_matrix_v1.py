#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matrix-only Split dense/sparse vs RoMeO four-part compute benchmark.

No model, no Hadamard, no shared q/k/v or gate/up logic.

Split timing:
  - dense: qext.matmul(A4_body, W4)
  - sparse: v58 scale_sparse_epilogue_oct with C=0

RoMeO timing:
  - W4A4 body
  - W4A8 / A8W4 / W8A8 outlier blocks

The ratio maps to:
  - Split: R = ceil(K * ratio)
  - RoMeO A outliers: ceil(M * ratio / 256) * 256
  - RoMeO W outliers: ceil(N * ratio / 512) * 512
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Iterable

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
ROMEO_ROOT = Path(os.environ.get("ROMEO_ROOT", "/data/yzy/RoMeo")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
for item in (TOOLS, ROMEO_ROOT, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import quarot.functional  # noqa: E402
import bench_full_model_policy_workspace_v32 as BASE  # noqa: E402
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension  # noqa: E402
import qfactory.kernels.gemm_w4a4_mixed_precision as qg  # noqa: E402


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


def install_one_config_qfactory():
    def one_config():
        return {
            "NStage": [2],
            "TileM": [128],
            "TileN": [128],
            "TileK": [128],
            "WarpM": [64],
            "WarpN": [64],
            "WarpK": [128],
        }

    qg.generate_tunable_keys = one_config


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_shapes(text: str) -> list[tuple[int, int]]:
    result = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.lower().split("x")
        result.append((int(left), int(right)))
    return result


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def ceil_ratio_count(k: int, ratio: float) -> int:
    return min(k - 1, max(1, int(math.ceil(k * ratio))))


def align_up(value: int, align: int) -> int:
    return int(math.ceil(value / align) * align)


def aligned_outliers(size: int, ratio: float, align: int) -> int:
    raw = int(math.ceil(size * ratio / align) * align)
    return min(size, max(align, raw))


def pack_i4(q: torch.Tensor) -> torch.Tensor:
    return quarot.functional.pack_i4(q.contiguous()).contiguous()


def time_cuda(fn: Callable[[], torch.Tensor | None], warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / max(1, iters))


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    diff = (reference.float() - candidate.float()).abs()
    denom = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "mse": float(torch.mean(diff * diff).item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
    }


def make_case_data(M: int, K: int, N: int, ratio: float, device: torch.device, main_ext, eps: float) -> dict:
    R = ceil_ratio_count(K, ratio)
    R_raw = R
    R = min(K - 1, align_up(R, 8))
    a_out = aligned_outliers(M, ratio, 256)
    w_out = aligned_outliers(N, ratio, 512)
    n_body = N - w_out
    if n_body <= 0:
        raise ValueError(f"RoMeO body N must be positive: N={N}, w_out={w_out}")

    A_q = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    W_q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
    A_pack = pack_i4(A_q)
    W_pack = pack_i4(W_q)
    W_body_pack = W_pack[:n_body].contiguous()
    W_out8 = torch.randint(0, 256, (w_out, K), dtype=torch.uint8, device=device).contiguous()
    A_out8 = torch.randint(0, 256, (a_out, K), dtype=torch.uint8, device=device).contiguous()
    weight = torch.randn(N, K, dtype=torch.float16, device=device)
    W_T = weight.t().contiguous()
    weight_bytes = (K * N + 1) // 2
    B_col = torch.empty(weight_bytes, dtype=torch.uint8, device=device)
    B_row = torch.empty(weight_bytes, dtype=torch.uint8, device=device)
    split_w_scale = torch.empty(N, dtype=torch.float32, device=device)
    main_ext.pack_weight_colmajor_s4(W_T, B_col, split_w_scale, eps)
    main_ext.pack_weight_rowmajor_s4_from_scale(W_T, B_row, split_w_scale)
    del weight, W_T

    idx = torch.randint(0, K, (M, R), dtype=torch.int32, device=device)
    top_q = torch.randint(-8, 8, (M, R), dtype=torch.int8, device=device)

    return {
        "R": R,
        "R_raw": R_raw,
        "a_out": a_out,
        "w_out": w_out,
        "n_body": n_body,
        "A_pack": A_pack,
        "W_pack": W_pack,
        "B_col_pack": B_col,
        "W_body_pack": W_body_pack,
        "W_out8": W_out8,
        "A_out8": A_out8,
        "B_row_pack": B_row,
        "idx": idx.contiguous(),
        "top_q": top_q.contiguous(),
        "body_scale": torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02),
        "top_scale": torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02),
        "w_scale": split_w_scale,
        "a_scale_bf16": torch.empty(M, dtype=torch.bfloat16, device=device).uniform_(0.001, 0.02),
        "a_out_scale_bf16": torch.empty(a_out, dtype=torch.bfloat16, device=device).uniform_(0.001, 0.02),
        "w_scale_bf16": torch.empty(N, dtype=torch.bfloat16, device=device).uniform_(0.001, 0.02),
    }


def qfactory_call(
    act: torch.Tensor,
    act_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    name: str,
) -> torch.Tensor:
    ret = qg.gemm_mixed_nt_perchannel(act, act_scale, weight, weight_scale, output, name)
    if ret != 0:
        raise RuntimeError(f"qfactory {name} returned {ret}")
    return output


def bench_case(qext, sparse_ext, main_ext, M: int, K: int, N: int, ratio: float, warmup: int, iters: int, device: torch.device, eps: float) -> dict:
    data = make_case_data(M, K, N, ratio, device, main_ext, eps)
    C_zero = torch.zeros((M, N), dtype=torch.int32, device=device)
    split_sparse_out = torch.empty((M, N), dtype=torch.float16, device=device)

    def split_dense():
        B_view = data["B_col_pack"].view(N, K // 2).contiguous()
        try:
            return qext.matmul(data["A_pack"], B_view, M, N, K)
        except TypeError:
            return qext.matmul(data["A_pack"], B_view)

    def split_sparse():
        sparse_ext.scale_sparse_epilogue_oct(
            C_zero,
            data["body_scale"],
            data["top_q"],
            data["idx"],
            data["B_row_pack"],
            data["top_scale"],
            data["w_scale"],
            split_sparse_out,
            K,
        )
        return split_sparse_out

    n_body = int(data["n_body"])
    a_out = int(data["a_out"])
    w_out = int(data["w_out"])
    out_a4w4 = torch.empty((M, n_body), dtype=torch.bfloat16, device=device)
    out_a4w8 = torch.empty((M, w_out), dtype=torch.bfloat16, device=device)
    out_a8w4 = torch.empty((a_out, n_body), dtype=torch.bfloat16, device=device)
    out_a8w8 = torch.empty((a_out, w_out), dtype=torch.bfloat16, device=device)
    a_scale = data["a_scale_bf16"]
    a_out_scale = data["a_out_scale_bf16"]
    w_scale = data["w_scale_bf16"]

    def romeo_w4a4():
        return qfactory_call(data["A_pack"], a_scale, data["W_body_pack"], w_scale[:n_body].contiguous(), out_a4w4, "a4w4")

    def romeo_w4a8():
        return qfactory_call(data["A_pack"], a_scale, data["W_out8"], w_scale[-w_out:].contiguous(), out_a4w8, "a4w8")

    def romeo_w8a4():
        return qfactory_call(data["A_out8"], a_out_scale, data["W_body_pack"], w_scale[:n_body].contiguous(), out_a8w4, "a8w4")

    def romeo_w8a8():
        return qfactory_call(data["A_out8"], a_out_scale, data["W_out8"], w_scale[-w_out:].contiguous(), out_a8w8, "a8w8")

    # Correctness guard for the dense body: QFactory a4w4 should match the
    # existing qext int4 GEMM plus per-row/per-column scale path.
    c_i32_body = qext.matmul(data["A_pack"], data["W_body_pack"])
    oracle_a4w4 = (c_i32_body.float() * a_scale.float().view(-1, 1) * w_scale[:n_body].float().view(1, -1)).to(torch.bfloat16)
    romeo_w4a4()
    correctness_a4w4 = compare_tensors(oracle_a4w4, out_a4w4)

    split_dense_ms = time_cuda(split_dense, warmup, iters, device)
    split_sparse_ms = time_cuda(split_sparse, warmup, iters, device)
    romeo_w4a4_ms = time_cuda(romeo_w4a4, warmup, iters, device)
    romeo_w4a8_ms = time_cuda(romeo_w4a8, warmup, iters, device)
    romeo_w8a4_ms = time_cuda(romeo_w8a4, warmup, iters, device)
    romeo_w8a8_ms = time_cuda(romeo_w8a8, warmup, iters, device)
    romeo_other_ms = romeo_w4a8_ms + romeo_w8a4_ms + romeo_w8a8_ms
    split_total_ms = split_dense_ms + split_sparse_ms
    romeo_total_ms = romeo_w4a4_ms + romeo_other_ms
    return {
        "M": M,
        "K": K,
        "N": N,
        "ratio": ratio,
        "split_R": int(data["R"]),
        "split_R_raw": int(data["R_raw"]),
        "split_R_alignment": 8,
        "romeo_a_outliers": a_out,
        "romeo_w_outliers": w_out,
        "romeo_w_outlier_alignment": 512,
        "romeo_body_N": n_body,
        "split_dense_ms": split_dense_ms,
        "split_sparse_ms": split_sparse_ms,
        "split_total_compute_ms": split_total_ms,
        "split_dense_pct": split_dense_ms / max(split_total_ms, 1e-12) * 100.0,
        "split_sparse_pct": split_sparse_ms / max(split_total_ms, 1e-12) * 100.0,
        "romeo_w4a4_ms": romeo_w4a4_ms,
        "romeo_w4a8_ms": romeo_w4a8_ms,
        "romeo_w8a4_ms": romeo_w8a4_ms,
        "romeo_w8a8_ms": romeo_w8a8_ms,
        "romeo_other_total_ms": romeo_other_ms,
        "romeo_total_compute_ms": romeo_total_ms,
        "romeo_w4a4_pct": romeo_w4a4_ms / max(romeo_total_ms, 1e-12) * 100.0,
        "romeo_other_pct": romeo_other_ms / max(romeo_total_ms, 1e-12) * 100.0,
        "split_over_romeo_total": split_total_ms / max(romeo_total_ms, 1e-12),
        "romeo_w4a4_correctness_relative_l2": correctness_a4w4["relative_l2"],
        "romeo_w4a4_correctness_mse": correctness_a4w4["mse"],
        "romeo_w4a4_correctness_max_abs": correctness_a4w4["max_abs"],
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", str(ROOT / "third_party/cutlass")))
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--M_values", default="")
    parser.add_argument("--shapes", default="")
    parser.add_argument("--ratios", default="")
    parser.add_argument("--verbose_compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    os.environ.setdefault("QFACTORY_ARCH", "120")
    os.environ.setdefault("QFACTORY_CACHE_DIR", str(out_dir / "qfactory_cache"))
    install_one_config_qfactory()

    if args.smoke:
        m_values = parse_csv_ints(args.M_values) if args.M_values else [256, 2048]
        shapes = parse_shapes(args.shapes) if args.shapes else [(4096, 4096)]
        ratios = parse_floats(args.ratios) if args.ratios else [0.005, 0.01]
        warmup = min(args.warmup, 2)
        iters = min(args.iters, 5)
    else:
        m_values = parse_csv_ints(args.M_values) if args.M_values else FULL_M_VALUES
        shapes = parse_shapes(args.shapes) if args.shapes else FULL_SHAPES
        ratios = parse_floats(args.ratios) if args.ratios else FULL_RATIOS
        warmup = args.warmup
        iters = args.iters

    qext = load_quarot_sm120_extension(verbose=args.verbose_compile)
    sparse_ext = load_fused_sparse_epilogue_ext(verbose=args.verbose_compile)
    cutlass_path = BASE.find_cutlass_path(args.cutlass_path)
    main_ext = BASE.load_ext(cutlass_path, verbose=args.verbose_compile)
    rows = []
    for M in m_values:
        for K, N in shapes:
            for ratio in ratios:
                row = bench_case(qext, sparse_ext, main_ext, M, K, N, ratio, warmup, iters, device, args.eps)
                rows.append(row)
                print("[SPLIT_VS_ROMEO_PARTS] " + json.dumps(row, ensure_ascii=False), flush=True)
                write_csv(out_dir / "split_vs_romeo_parts_partial.csv", rows)
    write_csv(out_dir / "split_vs_romeo_parts.csv", rows)
    (out_dir / "split_vs_romeo_parts.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "mode": "smoke" if args.smoke else "full",
                "m_values": m_values,
                "shapes": shapes,
                "ratios": ratios,
                "warmup": warmup,
                "iters": iters,
                "romeo_ratio_mapping": "A outliers=ceil(M*ratio/256)*256, W outliers=ceil(N*ratio/512)*512",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
