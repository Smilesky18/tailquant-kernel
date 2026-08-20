#!/usr/bin/env python3
"""Smoke QFactory dense-scaled GEMM plus standalone sparse-correction add.

This is an experimental Stage-2.5 path:

  old: QuaRot SM120 int32 GEMM + v58 dense-scale/sparse epilogue -> fp16
  new: QFactory A4W4 per-channel dense output -> bf16, then sparse add -> bf16

It intentionally does not modify the production v58/v61 files.
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
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension  # noqa: E402
from sparse_correction_add_ext_v1 import load_sparse_correction_add_ext_v1  # noqa: E402
import qfactory.kernels.gemm_w4a4_mixed_precision as qg  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/qfactory_dense_plus_sparse_add_v1/smoke.json"))
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--R", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--verbose_compile", action="store_true")
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


def encode_s4(x: torch.Tensor) -> torch.Tensor:
    return (x.to(torch.int16) & 0x0F).to(torch.uint8)


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

    A_pack = pack_i4_from_q(A_q_body)
    B_pack = pack_i4_from_q(B_q)
    B_row_pack = pack_i4_from_q(B_q.t().contiguous())

    body_scale = torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02)
    top_scale = torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02)
    w_scale = torch.empty(N, dtype=torch.float32, device=device).uniform_(0.001, 0.02)

    return {
        "A_pack": A_pack,
        "B_pack": B_pack,
        "B_row_pack": B_row_pack,
        "body_scale": body_scale,
        "top_scale": top_scale,
        "w_scale": w_scale,
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
    os.environ.setdefault("QFACTORY_CACHE_DIR", str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/qfactory_dense_plus_sparse_add_v1/qfactory_cache"))
    install_one_config_qfactory()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.K % 64 != 0:
        raise ValueError("K must be a multiple of 64")
    if args.N % 2 != 0:
        raise ValueError("N must be even for row-major int4 packing")
    if not (0 < args.R < args.K):
        raise ValueError("R must satisfy 0 < R < K")

    data = make_split_inputs(args.M, args.N, args.K, args.R, device)
    body_scale_bf16 = data["body_scale"].to(torch.bfloat16).contiguous()
    w_scale_bf16 = data["w_scale"].to(torch.bfloat16).contiguous()
    qext = load_quarot_sm120_extension(verbose=args.verbose_compile)
    old_ext = load_fused_sparse_epilogue_ext(verbose=args.verbose_compile)
    add_ext = load_sparse_correction_add_ext_v1(verbose=args.verbose_compile)

    old_out = torch.empty((args.M, args.N), dtype=torch.float16, device=device)
    dense_new = torch.empty((args.M, args.N), dtype=torch.bfloat16, device=device)

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

    def new_path():
        ret = qg.gemm_mixed_nt_perchannel(
            data["A_pack"],
            body_scale_bf16,
            data["B_pack"],
            w_scale_bf16,
            dense_new,
            "a4w4",
        )
        if ret != 0:
            raise RuntimeError(f"qfactory gemm_mixed_nt_perchannel returned {ret}")
        add_ext.sparse_correction_add_bf16_wscale_bf16_oct(
            dense_new,
            data["top_q"],
            data["idx"],
            data["B_row_pack"],
            data["top_scale"],
            w_scale_bf16,
        )
        return dense_new

    def new_dense_only():
        ret = qg.gemm_mixed_nt_perchannel(
            data["A_pack"],
            body_scale_bf16,
            data["B_pack"],
            w_scale_bf16,
            dense_new,
            "a4w4",
        )
        if ret != 0:
            raise RuntimeError(f"qfactory gemm_mixed_nt_perchannel returned {ret}")
        return dense_new

    def new_sparse_add_only_fp32_wscale():
        add_ext.sparse_correction_add_bf16_oct(
            dense_new,
            data["top_q"],
            data["idx"],
            data["B_row_pack"],
            data["top_scale"],
            data["w_scale"],
        )
        return dense_new

    def new_sparse_add_only_bf16_wscale():
        add_ext.sparse_correction_add_bf16_wscale_bf16_oct(
            dense_new,
            data["top_q"],
            data["idx"],
            data["B_row_pack"],
            data["top_scale"],
            w_scale_bf16,
        )
        return dense_new

    old_check = old_path().clone()
    new_check = new_path().clone()
    torch.cuda.synchronize(device)

    old_ms = time_cuda(old_path, args.warmup, args.iters, device)
    new_ms = time_cuda(new_path, args.warmup, args.iters, device)

    report = {
        "shape": {"M": args.M, "N": args.N, "K": args.K, "R": args.R},
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "old_path": "SM120 QuaRot int32 GEMM + v58 scale_sparse_epilogue_oct -> fp16",
        "new_path": "RoMeo/QFactory A4W4 per-channel dense -> bf16 + sparse_correction_add_bf16_wscale_bf16_oct",
        "correctness_new_vs_old": compare(old_check.to(torch.bfloat16), new_check),
        "latency_ms": {
            "old_int32_gemm_plus_v58_fused_epilogue_fp16": old_ms,
            "new_qfactory_dense_plus_sparse_add_bf16": new_ms,
            "new_qfactory_dense_only_bf16": time_cuda(new_dense_only, args.warmup, args.iters, device),
            "new_sparse_add_only_bf16_fp32_wscale": time_cuda(new_sparse_add_only_fp32_wscale, args.warmup, args.iters, device),
            "new_sparse_add_only_bf16_bf16_wscale": time_cuda(new_sparse_add_only_bf16_wscale, args.warmup, args.iters, device),
        },
        "speedup_percent": (old_ms - new_ms) / old_ms * 100.0,
        "notes": [
            "This is not full CUTLASS epilogue fusion yet; it removes int32 C materialization by using QFactory dense output, but sparse correction remains a second kernel.",
            "Scales are passed to QFactory as bf16 to match its public A4W4 per-channel interface; v58 uses fp32 scale inputs and fp16 output; the v2 smoke uses bf16 w_scale for the sparse add to match QFactory scale precision and reduce scale bandwidth.",
        ],
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
