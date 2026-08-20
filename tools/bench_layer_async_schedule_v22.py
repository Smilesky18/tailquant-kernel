#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment-only layer-level async scheduling for Qwen3 split linears.

No original repository sources are modified. The wrapper delays the per-linear
wait/merge boundary so independent q/k/v and gate/up projections can overlap
more of their dense/sparse work at the layer level.
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
from transformers.models.qwen3 import modeling_qwen3 as MQ

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
EXP = ROOT / "experiments/kernel_quant/layer_latency_split_v1"
TOOLS = EXP / "tools"
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts", TOOLS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

ENV_BIN = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = ENV_BIN + os.pathsep + os.environ.get("PATH", "")

import kernel_quant.scripts.bench_real_split_fullstack_v1 as REAL  # noqa: E402

V1_PATH = TOOLS / "bench_layer_bf16_vs_real_split_no_gptq_v1.py"
spec = importlib.util.spec_from_file_location("split_no_gptq_v1", str(V1_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot import {V1_PATH}")
V1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V1)


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
    p.add_argument("--variants", default="baseline,async_qkv,async_mlp,async_qkv_mlp")
    p.add_argument("--verbose_compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--check_correctness", action="store_true")
    return p.parse_args()


def load_model(model_name: str) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return model


def is_real_policy_linear(module: nn.Module) -> bool:
    return "RealPolicyLinear" in module.__class__.__name__ and hasattr(module, "_prepare_split")


def install_independent_scratch_pools(layer: nn.Module, device: torch.device) -> List[dict]:
    records = []
    for name, mod in layer.named_modules():
        if not is_real_policy_linear(mod):
            continue
        if not getattr(mod, "is_split", False):
            continue
        pool = REAL.BASE.SharedScratchPool(
            device=device,
            max_r_by_shape={(int(mod.K), int(mod.N)): int(mod.R)},
            split=True,
        )
        mod.scratch_pool = pool
        records.append({"name": name, "K": int(mod.K), "N": int(mod.N), "R": int(mod.R)})
    return records


class AsyncSplitPending:
    def __init__(self, module, original_shape, scratch, indices, dense_stream, sparse_stream):
        self.module = module
        self.original_shape = original_shape
        self.scratch = scratch
        self.indices = indices
        self.dense_stream = dense_stream
        self.sparse_stream = sparse_stream

    def wait(self) -> torch.Tensor:
        mod = self.module
        current = torch.cuda.current_stream(mod.w_scale.device)
        current.wait_stream(self.dense_stream)
        current.wait_stream(self.sparse_stream)
        output = torch.empty((self.scratch["Y_body"].shape[0], mod.N), dtype=torch.float16, device=mod.w_scale.device)
        torch.add(self.scratch["Y_body"], self.scratch["Y_sparse"], out=output)
        if getattr(mod, "bias", None) is not None:
            output = output + mod.bias
        return output.view(*self.original_shape, mod.N)


def launch_split_async(mod, x: torch.Tensor) -> AsyncSplitPending:
    if not is_real_policy_linear(mod):
        raise TypeError(f"Expected RealPolicyLinear, got {type(mod)}")
    if not getattr(mod, "is_split", False) or mod.mode != "dual_policy":
        raise RuntimeError(f"Async path expects split dual_policy linear, got mode={getattr(mod, 'mode', None)}")

    x = mod._rotate(x)
    original_shape = x.shape[:-1]
    A = x.reshape(-1, mod.K).contiguous()
    if A.dtype != torch.float16:
        A = A.to(torch.float16)

    M = int(A.shape[0])
    scratch = mod.scratch_pool.get(M, mod.K, mod.N)
    current = torch.cuda.current_stream(A.device)
    indices, top_q, _ = mod._prepare_split(A, scratch)

    mod.dense_stream.wait_stream(current)
    mod.sparse_stream.wait_stream(current)

    with torch.cuda.stream(mod.dense_stream):
        mod.ext.cutlass_s4_gemm(scratch["A_pack"], mod.B_col, scratch["C_body_i32"], M, mod.N, mod.K)
        mod.ext.scale_i32_to_fp16(scratch["C_body_i32"], scratch["body_scale"], mod.w_scale, scratch["Y_body"])

    with torch.cuda.stream(mod.sparse_stream):
        scratch["Y_sparse"].zero_()
        mod.ext.sparse_top_add_rowmajor_quad_shared(
            top_q,
            indices,
            mod.B_row,
            scratch["top_scale"],
            mod.w_scale,
            scratch["Y_sparse"],
            mod.K,
        )

    indices.record_stream(mod.sparse_stream)
    return AsyncSplitPending(mod, original_shape, scratch, indices, mod.dense_stream, mod.sparse_stream)


class AsyncQwen3DecoderLayer(nn.Module):
    def __init__(self, layer: nn.Module, *, async_qkv: bool, async_mlp: bool):
        super().__init__()
        self.layer = layer
        self.async_qkv = bool(async_qkv)
        self.async_mlp = bool(async_mlp)

    def _attention_forward(self, hidden_states, attention_mask, position_ids, past_key_values, position_embeddings, **kwargs):
        attn = self.layer.self_attn
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        if self.async_qkv:
            q_pending = launch_split_async(attn.q_proj, hidden_states)
            k_pending = launch_split_async(attn.k_proj, hidden_states)
            v_pending = launch_split_async(attn.v_proj, hidden_states)
            query_states = attn.q_norm(q_pending.wait().view(hidden_shape)).transpose(1, 2)
            key_states = attn.k_norm(k_pending.wait().view(hidden_shape)).transpose(1, 2)
            value_states = v_pending.wait().view(hidden_shape).transpose(1, 2)
        else:
            query_states = attn.q_norm(attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = MQ.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, attn.layer_idx)

        attention_interface = MQ.ALL_ATTENTION_FUNCTIONS.get_interface(
            attn.config._attn_implementation,
            MQ.eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            sliding_window=attn.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn.o_proj(attn_output)
        return attn_output, attn_weights

    def _mlp_forward(self, hidden_states):
        mlp = self.layer.mlp
        if self.async_mlp:
            gate_pending = launch_split_async(mlp.gate_proj, hidden_states)
            up_pending = launch_split_async(mlp.up_proj, hidden_states)
            gate = mlp.act_fn(gate_pending.wait())
            up = up_pending.wait()
            return mlp.down_proj(gate * up)
        return mlp(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer.input_layernorm(hidden_states)
        hidden_states, _ = self._attention_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values,
            position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        hidden_states = self._mlp_forward(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def make_variant_layer(layer: nn.Module, variant: str) -> nn.Module:
    if variant == "baseline":
        return layer
    if variant == "async_qkv":
        return AsyncQwen3DecoderLayer(layer, async_qkv=True, async_mlp=False)
    if variant == "async_mlp":
        return AsyncQwen3DecoderLayer(layer, async_qkv=False, async_mlp=True)
    if variant == "async_qkv_mlp":
        return AsyncQwen3DecoderLayer(layer, async_qkv=True, async_mlp=True)
    raise ValueError(variant)


def bench_variant(args, variant: str, device: torch.device, exts) -> tuple[Dict[int, float], dict]:
    main_ext, layout_ext, policy_pack_ext = exts
    model = load_model(args.model)
    hidden_size = V1.infer_hidden_size(model)
    layer = V1.get_layers(model)[args.layer_idx].to(device=device, dtype=torch.float16).eval()
    patch_info = V1.patch_layer_with_real_split_no_gptq(
        model=model,
        layer_idx=args.layer_idx,
        ratio=args.split_ratio,
        eps=args.eps,
        device=device,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
    )
    layer = V1.get_layers(model)[args.layer_idx].eval()
    isolated = []
    if variant != "baseline":
        isolated = install_independent_scratch_pools(layer, device)
        log(f"[INDEPENDENT_SCRATCH] {json.dumps(isolated)}")
    run_layer = make_variant_layer(layer, variant).eval()
    log(f"[VARIANT_READY] {variant} forward={inspect.signature(run_layer.forward)}")

    if args.check_correctness and variant != "baseline":
        batch0 = int(args.batches.split(",")[0])
        x = torch.randn(batch0, args.seq_len, hidden_size, device=device, dtype=torch.float16)
        pos = V1.make_position_ids(batch0, args.seq_len, device)
        pe = V1.build_position_embeddings(model, x, pos)
        with torch.no_grad():
            y_ref = V1.run_layer_once(layer, x, pos, pe, None)
            y_async = V1.run_layer_once(run_layer, x, pos, pe, None)
        torch.cuda.synchronize(device)
        diff = (y_ref - y_async).float()
        rel = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(y_ref.float()).clamp_min(1e-12)
        log(f"[CORRECTNESS:{variant}] max_abs={float(diff.abs().max().item()):.6e} rel_l2={float(rel.item()):.6e}")
        del x, pos, pe, y_ref, y_async, diff, rel
        torch.cuda.empty_cache()

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    ms = V1.bench_cases(
        model,
        run_layer,
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
    del run_layer, layer, model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return ms, {"hidden_size": hidden_size, "patch_info": patch_info, "independent_scratch": isolated}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[BATCHES] {args.batches}")
    log(f"[RATIO] {args.split_ratio}")
    log(f"[VARIANTS] {args.variants}")
    log("[NOTE] v22 layer-level async scheduling for q/k/v and gate/up split linears")

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability(device)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        log(f"[EXT] set TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

    exts = V1.load_real_split_extensions(args.cutlass_path, args.verbose_compile)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    by_variant: Dict[str, Dict[int, float]] = {}
    meta = {}
    for variant in variants:
        log(f"\n[RUN_VARIANT] {variant}")
        by_variant[variant], meta[variant] = bench_variant(args, variant, device, exts)

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
                "note": "no_gptq_layer_async_schedule_v22",
            })

    csv_path = out_dir / "split_async_schedule_v22.csv"
    json_path = out_dir / "split_async_schedule_v22.json"
    meta_path = out_dir / "split_async_schedule_v22_meta.json"
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
