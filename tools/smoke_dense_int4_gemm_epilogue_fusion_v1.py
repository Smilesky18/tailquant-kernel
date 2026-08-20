#!/usr/bin/env python3
"""Smoke test for the experimental dense INT4 GEMM epilogue-fusion extension."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (ROOT, TOOLS):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import quarot  # noqa: E402
import quarot.functional  # noqa: E402
from dense_int4_gemm_epilogue_fusion_ext_v1 import (  # noqa: E402
    load_dense_int4_gemm_epilogue_fusion_ext_v1,
)
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/dense_int4_gemm_epilogue_fusion_v1/smoke.json"))
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verbose_compile", action="store_true")
    return parser.parse_args()


def pack_random_i4(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    q = torch.randint(-8, 8, (rows, cols), dtype=torch.int8, device=device)
    return quarot.functional.pack_i4(q).contiguous()


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / max(1, iters))


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs()
    denom = torch.linalg.vector_norm(a.float()).clamp_min(1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "mse": float(torch.mean(diff * diff).item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
    }


def run_stage(name: str, fn: Callable[[], dict]) -> dict:
    started = time.time()
    try:
        out = fn()
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - smoke report should capture compile/runtime failures.
        out = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-30:],
        }
    out["elapsed_sec"] = round(time.time() - started, 3)
    return {name: out}


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    if args.K % 64 != 0:
        raise ValueError("K must be a multiple of 64 for the existing quarot.matmul packed-K constraint")

    A = pack_random_i4(args.M, args.K, device)
    B = pack_random_i4(args.N, args.K, device)
    w_scale = torch.empty(args.N, dtype=torch.float32, device=device).uniform_(0.001, 0.02)

    report: dict = {
        "shape": {"M": args.M, "N": args.N, "K": args.K},
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "stages": {},
        "notes": [
            "Stage 1 removes the int32 C materialization for dense GEMM by writing fp16 from CUTLASS epilogue.",
            "Stage 2 tries CUTLASS PerChannelScaling for column-wise w_scale only.",
            "body_scale[m] and indexed sparse correction require a custom coordinate-aware epilogue/visitor and are not represented by this smoke extension.",
        ],
    }

    qext = load_quarot_sm120_extension(verbose=args.verbose_compile)
    ext = load_dense_int4_gemm_epilogue_fusion_ext_v1(verbose=args.verbose_compile)

    C_ref = qext.matmul(A, B)
    D_ref_unscaled = C_ref.to(torch.float16)

    def stage1():
        D = ext.dense_i4_gemm_fp16_unscaled(A, B)
        return {
            "correctness_vs_int32_cast_fp16": compare(D_ref_unscaled, D),
            "latency_ms": {
                "old_int32_gemm_only": time_cuda(lambda: qext.matmul(A, B), args.warmup, args.iters),
                "new_fp16_epilogue_gemm_only": time_cuda(lambda: ext.dense_i4_gemm_fp16_unscaled(A, B), args.warmup, args.iters),
            },
        }

    report["stages"].update(run_stage("stage1_dense_fp16_epilogue_unscaled", stage1))

    report["stages"]["stage2_dense_fp16_epilogue_wscale_only"] = {
        "status": "blocked_by_cutlass_interface",
        "reason": (
            "A direct CUTLASS 2.x device::Gemm attempt with "
            "LinearCombination<..., ScaleType::PerChannelScaling> failed to compile: "
            "the standard threadblock epilogue calls output_op(accumulator, source) "
            "and requires set_k_partition(), while this PerChannelScaling specialization "
            "expects vector-alpha fragments supplied by a different epilogue path. "
            "This keeps Stage 2 from being a simple output-op swap."
        ),
        "old_path_latency_ms": {
            "old_int32_gemm_plus_torch_wscale": time_cuda(
                lambda: (qext.matmul(A, B).float() * w_scale.view(1, -1)).to(torch.float16),
                args.warmup,
                args.iters,
            )
        },
    }
    report["stages"]["stage3_body_scale_plus_sparse_correction"] = {
        "status": "blocked_by_cutlass_interface",
        "reason": (
            "CUTLASS 2.x device::Gemm standard thread output-op used here has no direct "
            "m/n coordinate path for body_scale[m] or indexed sparse correction "
            "top_q[m,r], idx[m,r], B_row_pack[idx,n]. This needs a custom epilogue visitor "
            "or custom kernel, so no fake fusion smoke was wired."
        ),
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
