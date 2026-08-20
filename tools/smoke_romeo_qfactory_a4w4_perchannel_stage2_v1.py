#!/usr/bin/env python3
"""Smoke RoMeo/QFactory A4W4 per-channel GEMM as a Stage-2 fusion candidate.

This script is standalone and only imports RoMeo's qfactory kernel. It compares:

  old path: SM120 QuaRot int32 GEMM + scale_i32_to_fp16-style torch scaling
  new path: QFactory hand-written/CuTe A4W4 GEMM with A_scale[m] * B_scale[n]

The QFactory kernel writes bf16, so correctness is checked against a bf16 oracle.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
ROMEO_ROOT = Path(os.environ.get("ROMEO_ROOT", "/data/yzy/RoMeo")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (TOOLS, ROMEO_ROOT, ROOT):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import quarot.functional  # noqa: E402
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension  # noqa: E402
import qfactory.kernels.gemm_w4a4_mixed_precision as qg  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/romeo_qfactory_stage2_v1/smoke.json"))
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--verbose_qext", action="store_true")
    return p.parse_args()


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


def pack_random_i4(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    q = torch.randint(-8, 8, (rows, cols), dtype=torch.int8, device=device)
    return quarot.functional.pack_i4(q).contiguous()


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


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs()
    denom = torch.linalg.vector_norm(a.float()).clamp_min(1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "mse": float(torch.mean(diff * diff).item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
    }


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("QFACTORY_ARCH", "120")
    os.environ.setdefault("QFACTORY_CACHE_DIR", str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/romeo_qfactory_stage2_v1/qfactory_cache"))
    install_one_config_qfactory()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    A = pack_random_i4(args.M, args.K, device)
    B = pack_random_i4(args.N, args.K, device)
    a_scale = torch.empty(args.M, dtype=torch.bfloat16, device=device).uniform_(0.001, 0.02)
    b_scale = torch.empty(args.N, dtype=torch.bfloat16, device=device).uniform_(0.001, 0.02)
    qfactory_out = torch.empty((args.M, args.N), dtype=torch.bfloat16, device=device)

    qext = load_quarot_sm120_extension(verbose=args.verbose_qext)
    C_i32 = qext.matmul(A, B)
    oracle = (C_i32.float() * a_scale.float().view(-1, 1) * b_scale.float().view(1, -1)).to(torch.bfloat16)

    ret = qg.gemm_mixed_nt_perchannel(A, a_scale, B, b_scale, qfactory_out, "a4w4")
    torch.cuda.synchronize(device)
    if ret != 0:
        raise RuntimeError(f"qfactory gemm_mixed_nt_perchannel returned {ret}")

    def old_path():
        c = qext.matmul(A, B)
        return (c.float() * a_scale.float().view(-1, 1) * b_scale.float().view(1, -1)).to(torch.bfloat16)

    def new_path():
        qg.gemm_mixed_nt_perchannel(A, a_scale, B, b_scale, qfactory_out, "a4w4")
        return qfactory_out

    report = {
        "shape": {"M": args.M, "N": args.N, "K": args.K},
        "device": torch.cuda.get_device_name(device),
        "kernel": "RoMeo qfactory gemm_mixed_nt_perchannel name=a4w4, one-config preset",
        "correctness_vs_old_int32_scaled_bf16": compare(oracle, qfactory_out),
        "latency_ms": {
            "old_int32_gemm_plus_torch_scale_to_bf16": time_cuda(old_path, args.warmup, args.iters, device),
            "qfactory_a4w4_perchannel_bf16": time_cuda(new_path, args.warmup, args.iters, device),
        },
        "notes": [
            "This is Stage 2 only: dense int4 GEMM plus row/column scales.",
            "It does not include the current split sparse correction yet.",
            "QFactory output/scales are bf16, while the v58 path currently uses fp16/float scales.",
        ],
    }
    old = report["latency_ms"]["old_int32_gemm_plus_torch_scale_to_bf16"]
    new = report["latency_ms"]["qfactory_a4w4_perchannel_bf16"]
    report["speedup_percent"] = (old - new) / old * 100.0

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
