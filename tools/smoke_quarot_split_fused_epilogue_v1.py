#!/usr/bin/env python3
"""Smoke true QuaRot-style split sparse correction inside CUTLASS GEMM output-op."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for p in (TOOLS, ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import quarot.functional  # noqa: E402
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension  # noqa: E402
from quarot_split_fused_epilogue_ext_v1 import load_quarot_split_fused_epilogue_ext_v1  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/quarot_split_fused_epilogue_v1/smoke.json"))
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--N", type=int, default=4096)
    p.add_argument("--K", type=int, default=4096)
    p.add_argument("--R", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--verbose_compile", action="store_true")
    return p.parse_args()


def pack_i4(q: torch.Tensor) -> torch.Tensor:
    return quarot.functional.pack_i4(q.contiguous()).contiguous()


def make_data(M: int, N: int, K: int, R: int, device: torch.device):
    A_q_full = torch.randint(-8, 8, (M, K), dtype=torch.int8, device=device)
    B_q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
    idx = torch.empty((M, R), dtype=torch.int32, device=device)
    for m in range(M):
        idx[m] = torch.randperm(K, device=device, dtype=torch.int32)[:R]
    top_q = torch.gather(A_q_full, 1, idx.to(torch.int64)).contiguous()
    A_q_body = A_q_full.clone()
    A_q_body.scatter_(1, idx.to(torch.int64), 0)
    return {
        "A_pack": pack_i4(A_q_body),
        "B_pack": pack_i4(B_q),
        "B_row_pack": pack_i4(B_q.t().contiguous()),
        "body_scale": torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02),
        "top_scale": torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02),
        "w_scale": torch.empty(N, dtype=torch.float32, device=device).uniform_(0.001, 0.02),
        "top_q": top_q,
        "idx": idx.contiguous(),
    }


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    ed.record()
    torch.cuda.synchronize(device)
    return float(st.elapsed_time(ed) / max(1, iters))


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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    data = make_data(args.M, args.N, args.K, args.R, device)

    qext = load_quarot_sm120_extension(verbose=args.verbose_compile)
    old_ext = load_fused_sparse_epilogue_ext(verbose=args.verbose_compile)
    fused_ext = load_quarot_split_fused_epilogue_ext_v1(verbose=args.verbose_compile)
    old_out = torch.empty((args.M, args.N), dtype=torch.float16, device=device)

    def old_path():
        c = qext.matmul(data["A_pack"], data["B_pack"])
        old_ext.scale_sparse_epilogue_oct(c, data["body_scale"], data["top_q"], data["idx"], data["B_row_pack"], data["top_scale"], data["w_scale"], old_out, args.K)
        return old_out

    def fused_path():
        return fused_ext.quarot_split_gemm_fused_epilogue_staged_v2(data["A_pack"], data["B_pack"], data["body_scale"], data["top_q"], data["idx"], data["B_row_pack"], data["top_scale"], data["w_scale"])

    old_check = old_path().clone()
    fused_check = fused_path().clone()
    torch.cuda.synchronize(device)

    old_ms = time_cuda(old_path, args.warmup, args.iters, device)
    fused_ms = time_cuda(fused_path, args.warmup, args.iters, device)

    report = {
        "shape": {"M": args.M, "N": args.N, "K": args.K, "R": args.R},
        "device": torch.cuda.get_device_name(device),
        "correctness_vs_v58_old": compare(old_check, fused_check),
        "latency_ms": {
            "old_quarot_int32_gemm_plus_v58_epilogue": old_ms,
            "new_quarot_gemm_custom_epilogue_staged_v2": fused_ms,
        },
        "speedup_percent": (old_ms - fused_ms) / old_ms * 100.0,
        "notes": [
            "New path stages idx/top_q in a custom CUTLASS threadblock epilogue, computes dense+sparse, and returns fp16 directly.",
            "No int32 C output tensor is materialized by the new path.",
            "This v1 uses a coordinate-aware output-op based on CUTLASS ThreadMap; correctness is the primary gate.",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
