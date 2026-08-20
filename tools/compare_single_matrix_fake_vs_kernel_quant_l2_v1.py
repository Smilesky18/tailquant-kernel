#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare fake Split activation QDQ with the real CUDA packing kernel.

This is a single-matrix smoke test. It reconstructs the quantized activation
matrix from pack_policy_split's packed body and tail buffers, then compares it
against the Python fake-quant implementation using the same thresholds/indices.
The main metrics are quantization error against the original matrix:
relative L2 error and MSE.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from pack_policy_activation_v1 import load_policy_pack_ext  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--K", type=int, default=4096)
    parser.add_argument("--ratio", type=float, default=0.005)
    parser.add_argument("--activation_percentile", type=float, default=98.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verbose_compile", action="store_true")
    return parser.parse_args()


def decode_signed_int4(packed: torch.Tensor, M: int, K: int) -> torch.Tensor:
    values = packed.to(torch.int16)
    low = values & 0x0F
    high = (values >> 4) & 0x0F
    decoded = torch.empty(M * K, dtype=torch.int16, device=packed.device)
    decoded[0::2] = low
    decoded[1::2] = high[: decoded[1::2].numel()]
    decoded = torch.where(decoded >= 8, decoded - 16, decoded)
    return decoded.view(M, K).to(torch.float32)


def compute_thresholds_and_indices(A: torch.Tensor, ratio: float, percentile: float):
    M, K = A.shape
    R = min(K - 1, max(1, int(math.ceil(K * ratio))))
    body_len = K - R
    body_kth = min(K, max(1, int(math.ceil(body_len * percentile / 100.0))))
    descending_rank = K - body_kth + 1
    select_k = max(R, descending_rank)

    abs_A = A.abs().float()
    top_values, top_indices = torch.topk(
        abs_A,
        k=select_k,
        dim=1,
        largest=True,
        sorted=True,
    )
    body_threshold = top_values[:, descending_rank - 1].contiguous()
    tail_threshold = top_values[:, 0].contiguous()
    tail_indices = top_indices[:, :R]
    tail_indices, _ = torch.sort(tail_indices, dim=1)
    return R, body_threshold, tail_threshold, tail_indices.to(torch.int32).contiguous()


def fake_quant_reconstruct(
    A: torch.Tensor,
    body_threshold: torch.Tensor,
    tail_threshold: torch.Tensor,
    tail_indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    A32 = A.float()
    body_scale = (body_threshold.view(-1, 1).clamp_min(eps) / 7.0)
    body_q = torch.round(torch.clamp(A32, -body_threshold.view(-1, 1), body_threshold.view(-1, 1)) / body_scale).clamp(-8, 7)
    out = body_q * body_scale

    tail_scale = (tail_threshold.view(-1, 1).clamp_min(eps) / 7.0)
    tail_q_full = torch.round(A32 / tail_scale).clamp(-8, 7)
    idx = tail_indices.long()
    out.scatter_(1, idx, tail_q_full.gather(1, idx) * tail_scale)
    return out


def kernel_quant_reconstruct(
    ext,
    A: torch.Tensor,
    body_threshold: torch.Tensor,
    tail_threshold: torch.Tensor,
    tail_indices: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    M, K = A.shape
    R = int(tail_indices.shape[1])
    A_pack = torch.empty((M * K + 1) // 2, dtype=torch.uint8, device=A.device)
    body_scale = torch.empty(M, dtype=torch.float32, device=A.device)
    tail_scale = torch.empty(M, dtype=torch.float32, device=A.device)
    tail_q = torch.empty(M, R, dtype=torch.int8, device=A.device)

    ext.pack_policy_split(
        A.contiguous(),
        tail_indices,
        body_threshold,
        tail_threshold,
        A_pack,
        body_scale,
        tail_scale,
        tail_q,
        float(eps),
    )
    body_q = decode_signed_int4(A_pack, M, K)
    out = body_q * body_scale.view(-1, 1)
    out.scatter_(1, tail_indices.long(), tail_q.float() * tail_scale.view(-1, 1))
    return out


def l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.float()).item())


def quant_error_metrics(original: torch.Tensor, quantized: torch.Tensor) -> dict:
    error = (original.float() - quantized.float()).abs()
    original_l2 = l2(original)
    error_l2 = l2(error)
    return {
        "relative_l2_error": error_l2 / max(original_l2, 1e-12),
        "mse": float(torch.mean(error * error).item()),
        "absolute_error_l2": error_l2,
        "max_abs_error": float(error.max().item()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ext = load_policy_pack_ext(verbose=args.verbose_compile)

    A = torch.randn(args.M, args.K, dtype=torch.float16, device=device)
    # Add a mild outlier structure so top-R is exercised deterministically.
    A[:, ::257] *= 8.0

    R, body_threshold, tail_threshold, tail_indices = compute_thresholds_and_indices(
        A,
        args.ratio,
        args.activation_percentile,
    )
    fake_q = fake_quant_reconstruct(A, body_threshold, tail_threshold, tail_indices, args.eps)
    kernel_q = kernel_quant_reconstruct(ext, A, body_threshold, tail_threshold, tail_indices, args.eps)
    torch.cuda.synchronize()

    fake_error = quant_error_metrics(A, fake_q)
    kernel_error = quant_error_metrics(A, kernel_q)
    fake_kernel_diff = kernel_q - fake_q
    result = {
        "matrix": {
            "M": args.M,
            "K": args.K,
            "dtype": str(A.dtype),
            "seed": args.seed,
        },
        "policy": {
            "ratio": args.ratio,
            "R": R,
            "activation_percentile": args.activation_percentile,
            "eps": args.eps,
        },
        "quantization_error": {
            "fake_quant": fake_error,
            "kernel_quant_reconstructed": kernel_error,
            "kernel_minus_fake": {
                "l2": l2(fake_kernel_diff),
                "relative_to_fake_quant_l2": l2(fake_kernel_diff) / max(l2(fake_q), 1e-12),
                "mse": float(torch.mean(fake_kernel_diff.float() * fake_kernel_diff.float()).item()),
                "max_abs": float(fake_kernel_diff.abs().max().item()),
            },
            "kernel_over_fake_relative_l2_error": (
                kernel_error["relative_l2_error"] / max(fake_error["relative_l2_error"], 1e-12)
            ),
            "kernel_over_fake_mse": kernel_error["mse"] / max(fake_error["mse"], 1e-30),
        },
        "reference_l2": {
            "original": l2(A),
            "fake_quant": l2(fake_q),
            "kernel_quant_reconstructed": l2(kernel_q),
        },
        "allclose": bool(torch.allclose(kernel_q, fake_q, rtol=0.0, atol=0.0)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
