#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for QFactory dense+sparse fused epilogue prototype."""
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
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension  # noqa: E402
from qfactory_split_fused_epilogue_v1 import (  # noqa: E402
    gemm_a4w4_split_fused,
    install_one_config_qfactory,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/qfactory_split_fused_epilogue_v1/smoke.json"))
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--R", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verbose_compile", action="store_true")
    return parser.parse_args()


def pack_i4_from_q(q: torch.Tensor) -> torch.Tensor:
    return quarot.functional.pack_i4(q.contiguous()).contiguous()


def make_split_inputs(M: int, N: int, K: int, R: int, device: torch.device):
    A_q_full = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    B_q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
    idx = torch.empty((M, R), dtype=torch.int32, device=device)
    for m in range(M):
        idx[m] = torch.randperm(K, device=device, dtype=torch.int32)[:R]

    top_q = torch.gather(A_q_full, 1, idx.to(torch.int64)).contiguous()
    A_q_body = A_q_full.clone()
    A_q_body.scatter_(1, idx.to(torch.int64), 0)
    return {
        "A_pack": pack_i4_from_q(A_q_body),
        "B_pack": pack_i4_from_q(B_q),
        "B_row_pack": pack_i4_from_q(B_q.t().contiguous()),
        "body_scale": torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02),
        "top_scale": torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02),
        "w_scale": torch.empty(N, dtype=torch.float32, device=device).uniform_(0.001, 0.02),
        "top_q": top_q,
        "idx": idx.contiguous(),
    }


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
    os.environ.setdefault("QFACTORY_CACHE_DIR", str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/qfactory_split_fused_epilogue_v1/qfactory_cache"))
    install_one_config_qfactory()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.K % 128 != 0 or args.N % 128 != 0 or args.M % 128 != 0:
        raise ValueError("prototype smoke expects M/N/K multiples of 128")
    if not (0 < args.R < args.K):
        raise ValueError("R must satisfy 0 < R < K")

    data = make_split_inputs(args.M, args.N, args.K, args.R, device)
    body_scale_bf16 = data["body_scale"].to(torch.bfloat16).contiguous()
    w_scale_bf16 = data["w_scale"].to(torch.bfloat16).contiguous()
    qext = load_quarot_sm120_extension(verbose=args.verbose_compile)
    old_ext = load_fused_sparse_epilogue_ext(verbose=args.verbose_compile)

    old_out = torch.empty((args.M, args.N), dtype=torch.float16, device=device)
    fused_out = torch.empty((args.M, args.N), dtype=torch.bfloat16, device=device)

    def old_path():
        c = qext.matmul(data["A_pack"], data["B_pack"])
        old_ext.scale_sparse_epilogue_oct(
            c,
            data["body_scale"],
            data["top_q"],
            data["idx"],
            data["B_row_pack"],
            data["top_scale"],
            data["w_scale"],
            old_out,
            args.K,
        )
        return old_out

    def fused_path():
        ret = gemm_a4w4_split_fused(
            data["A_pack"],
            body_scale_bf16,
            data["B_pack"],
            w_scale_bf16,
            fused_out,
            data["top_q"],
            data["idx"],
            data["B_row_pack"],
            data["top_scale"],
        )
        if ret != 0:
            raise RuntimeError(f"fused kernel returned {ret}")
        return fused_out

    old_check = old_path().clone()
    fused_check = fused_path().clone()
    torch.cuda.synchronize(device)
    old_ms = time_cuda(old_path, args.warmup, args.iters, device)
    fused_ms = time_cuda(fused_path, args.warmup, args.iters, device)
    result = {
        "shape": {"M": args.M, "N": args.N, "K": args.K, "R": args.R},
        "old_path": "SM120 QuaRot int32 GEMM + v58 fused epilogue -> fp16",
        "new_path": "QFactory dense scale + sparse correction in shared-tile store path -> bf16",
        "correctness_new_vs_old_bf16": compare(old_check.to(torch.bfloat16), fused_check),
        "latency_ms": {
            "old_v58": old_ms,
            "qfactory_split_fused_v1": fused_ms,
        },
        "speedup_percent": (old_ms - fused_ms) / old_ms * 100.0,
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
