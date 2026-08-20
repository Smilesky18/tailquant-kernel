#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare baseline RealPolicyLinear sparse add vs v20 sparse write-only kernels.

All code stays in the experiment directory. No original project files are
modified. The optimized path preserves the existing dense/sparse two-stream
execution but replaces:
  Y_sparse.zero_(); sparse_top_add(..., Y_sparse)
with:
  sparse_top_write(..., Y_sparse)
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
EXP = ROOT / "experiments/kernel_quant/layer_latency_split_v1"
TOOLS = EXP / "tools"
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts", TOOLS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

ENV_BIN = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = ENV_BIN + os.pathsep + os.environ.get("PATH", "")

import kernel_quant.scripts.bench_real_split_fullstack_v1 as REAL  # noqa: E402
from split_sparse_write_ext_v20 import load_sparse_write_ext  # noqa: E402

V1_PATH = TOOLS / "bench_layer_bf16_vs_real_split_no_gptq_v1.py"
spec = importlib.util.spec_from_file_location("split_no_gptq_v1", str(V1_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {V1_PATH}")
V1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V1)


class RealPolicyLinearSparseWrite(REAL.RealPolicyLinear):
    def __init__(self, *args, sparse_write_ext, sparse_write_variant: str, **kwargs):
        super().__init__(*args, **kwargs)
        if sparse_write_variant not in {"quad", "oct"}:
            raise ValueError(sparse_write_variant)
        self.sparse_write_ext = sparse_write_ext
        self.sparse_write_variant = sparse_write_variant

    def _split_compute(
        self,
        A: torch.Tensor,
        scratch: Dict[str, torch.Tensor],
        B_col: torch.Tensor,
        dense_ready_event: Optional[torch.cuda.Event],
        dense_stream: torch.cuda.Stream,
        sparse_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        M = int(A.shape[0])
        current = torch.cuda.current_stream(A.device)
        indices, top_q, _ = self._prepare_split(A, scratch)

        dense_stream.wait_stream(current)
        if dense_ready_event is not None:
            dense_stream.wait_event(dense_ready_event)
        sparse_stream.wait_stream(current)

        with torch.cuda.stream(dense_stream):
            self.ext.cutlass_s4_gemm(
                scratch["A_pack"],
                B_col,
                scratch["C_body_i32"],
                M,
                self.N,
                self.K,
            )
            self.ext.scale_i32_to_fp16(
                scratch["C_body_i32"],
                scratch["body_scale"],
                self.w_scale,
                scratch["Y_body"],
            )

        with torch.cuda.stream(sparse_stream):
            if self.sparse_write_variant == "oct":
                self.sparse_write_ext.sparse_top_write_rowmajor_oct_shared(
                    top_q,
                    indices,
                    self.B_row,
                    scratch["top_scale"],
                    self.w_scale,
                    scratch["Y_sparse"],
                    self.K,
                )
            else:
                self.sparse_write_ext.sparse_top_write_rowmajor_quad_shared(
                    top_q,
                    indices,
                    self.B_row,
                    scratch["top_scale"],
                    self.w_scale,
                    scratch["Y_sparse"],
                    self.K,
                )

        indices.record_stream(sparse_stream)
        current.wait_stream(dense_stream)
        current.wait_stream(sparse_stream)

        output = torch.empty((M, self.N), dtype=torch.float16, device=A.device)
        torch.add(scratch["Y_body"], scratch["Y_sparse"], out=output)
        return output


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--split_ratio", type=float, default=0.05)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--timing", choices=["graph", "eager"], default="graph")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=None)
    p.add_argument("--variants", default="baseline,write_quad,write_oct")
    p.add_argument("--verbose_compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(model_name: str) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def patch_sparse_write_layer(
    *,
    model: nn.Module,
    layer_idx: int,
    ratio: float,
    eps: float,
    device: torch.device,
    main_ext,
    layout_ext,
    policy_pack_ext,
    sparse_write_ext,
    sparse_write_variant: str,
) -> dict:
    targets = V1.collect_layer_targets(model, layer_idx)
    if len(targets) != len(V1.TARGET_SUFFIXES):
        raise RuntimeError(f"Expected {len(V1.TARGET_SUFFIXES)} targets, got {len(targets)}")

    max_r_by_shape: Dict[Tuple[int, int], int] = {}
    for _, module in targets:
        linear = REAL.unwrap_linear(module)
        N, K = map(int, linear.weight.shape)
        R = REAL.BASE.ceil_ratio_count(K, ratio)
        if R <= 0:
            raise RuntimeError(f"ratio={ratio} gives R=0 for K={K}")
        max_r_by_shape[(K, N)] = max(max_r_by_shape.get((K, N), 0), R)

    scratch_pool = REAL.BASE.SharedScratchPool(device=device, max_r_by_shape=max_r_by_shape, split=True)
    policy_cfg = V1.make_policy_cfg(ratio)
    records = []

    for raw_name, module in targets:
        linear = REAL.unwrap_linear(module)
        weight_cpu = linear.weight.detach().cpu()
        bias_cpu = None if linear.bias is None else linear.bias.detach().cpu()
        scale_cpu = V1.make_rtn_scale(weight_cpu, eps)
        replacement = RealPolicyLinearSparseWrite(
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            mode="dual_policy",
            weight_cpu=weight_cpu,
            bias_cpu=bias_cpu,
            policy_cfg=policy_cfg,
            gptq_scale_cpu=scale_cpu,
            eps=eps,
            device=device,
            name=raw_name,
            scratch_pool=scratch_pool,
            prefetch_workspace=None,
            rotate_online=False,
            had_k=None,
            had_factor=1,
            sparse_write_ext=sparse_write_ext,
            sparse_write_variant=sparse_write_variant,
        )
        REAL.BASE.set_submodule(model, raw_name, replacement)
        records.append({
            "name": replacement.name,
            "K": replacement.K,
            "N": replacement.N,
            "R": replacement.R,
            "variant": f"write_{sparse_write_variant}",
            "weight_pack_rel_l2": replacement.weight_pack_rel_l2,
        })
        log(f"[PATCH_WRITE_{sparse_write_variant}] {raw_name} K={replacement.K} N={replacement.N} R={replacement.R}")
        del module, linear, weight_cpu, bias_cpu, scale_cpu
        gc.collect()
        torch.cuda.empty_cache()

    return {"records": records, "scratch_allocated_bytes_after_patch": int(scratch_pool.allocated_bytes())}


def patch_baseline_layer(**kwargs) -> dict:
    return V1.patch_layer_with_real_split_no_gptq(**kwargs)


def bench_variant(args, variant: str, device: torch.device, exts, sparse_write_ext) -> tuple[Dict[int, float], dict]:
    main_ext, layout_ext, policy_pack_ext = exts
    model = load_model(args.model)
    hidden_size = V1.infer_hidden_size(model)
    layer = V1.get_layers(model)[args.layer_idx].to(device=device, dtype=torch.float16).eval()

    if variant == "baseline":
        patch_info = patch_baseline_layer(
            model=model,
            layer_idx=args.layer_idx,
            ratio=args.split_ratio,
            eps=args.eps,
            device=device,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
        )
    elif variant == "write_quad":
        patch_info = patch_sparse_write_layer(
            model=model,
            layer_idx=args.layer_idx,
            ratio=args.split_ratio,
            eps=args.eps,
            device=device,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            sparse_write_ext=sparse_write_ext,
            sparse_write_variant="quad",
        )
    elif variant == "write_oct":
        patch_info = patch_sparse_write_layer(
            model=model,
            layer_idx=args.layer_idx,
            ratio=args.split_ratio,
            eps=args.eps,
            device=device,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            sparse_write_ext=sparse_write_ext,
            sparse_write_variant="oct",
        )
    else:
        raise ValueError(variant)

    layer = V1.get_layers(model)[args.layer_idx].eval()
    log(f"[VARIANT_READY] {variant} forward={inspect.signature(layer.forward)}")
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    ms = V1.bench_cases(
        model,
        layer,
        hidden_size,
        batches,
        args.seq_len,
        torch.float16,
        device,
        args.warmup,
        args.iters,
        args.timing,
        variant,
    )
    del layer, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return ms, {"hidden_size": hidden_size, "patch_info": patch_info}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[VARIANTS] {args.variants}")
    log(f"[NOTE] v20 sparse-write removes Y_sparse.zero_ and sparse read-modify-write of zero buffer")

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability(device)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        log(f"[EXT] set TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

    exts = V1.load_real_split_extensions(args.cutlass_path, args.verbose_compile)
    sparse_write_ext = load_sparse_write_ext(args.verbose_compile)
    log(f"[SPARSE_WRITE_EXT] {sparse_write_ext}")

    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    by_variant: Dict[str, Dict[int, float]] = {}
    meta = {}
    for variant in variants:
        log(f"\n[RUN_VARIANT] {variant}")
        by_variant[variant], meta[variant] = bench_variant(args, variant, device, exts, sparse_write_ext)

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows: List[dict] = []
    for batch in batches:
        base = by_variant.get("baseline", {}).get(batch)
        for variant in variants:
            value = by_variant[variant].get(batch)
            rows.append({
                "model": args.model,
                "layer_idx": args.layer_idx,
                "batch": batch,
                "seq_len": args.seq_len,
                "split_ratio": args.split_ratio,
                "variant": variant,
                "split_ms": value,
                "baseline_split_ms": base,
                "speedup_vs_baseline": None if base is None or value is None else float(base / value),
                "normalized_vs_baseline": None if base is None or value is None else float(value / base),
                "timing": args.timing,
                "note": "no_gptq_sparse_write_v20",
            })

    csv_path = out_dir / "split_sparse_write_v20.csv"
    json_path = out_dir / "split_sparse_write_v20.json"
    meta_path = out_dir / "split_sparse_write_v20_meta.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json.dump(rows, open(json_path, "w"), indent=2)
    json.dump(meta, open(meta_path, "w"), indent=2)
    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[META] {meta_path}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
