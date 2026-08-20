#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layer-level BF16 vs real Split W4A4 latency without GPTQ/calibration.

This script is intentionally self-contained in the experiment directory. It
reuses the real Split runtime modules/extensions from the project, but does not
invoke the full-stack benchmark CLI, datasets, GPTQ, PPL, or calibration.
"""
from __future__ import annotations

import argparse
import csv
import gc
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

ENV_BIN = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = ENV_BIN + os.pathsep + os.environ.get("PATH", "")

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
for item in (
    ROOT,
    ROOT / "fake_quant",
    ROOT / "kernel_quant",
    ROOT / "kernel_quant/scripts",
):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import kernel_quant.scripts.bench_real_split_fullstack_v1 as REAL  # noqa: E402

TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

NOTE = "no_gptq_no_calibration_latency_only"


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
    p.add_argument("--out_dir", required=True)
    p.add_argument("--timing", default="graph", choices=["graph", "eager"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--cutlass_path", default=None)
    p.add_argument("--verbose_compile", action="store_true")
    p.add_argument("--skip_bf16", action="store_true")
    p.add_argument("--skip_split", action="store_true")
    return p.parse_args()


def get_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers.")


def infer_hidden_size(model: nn.Module) -> int:
    cfg = model.config
    for key in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    raise RuntimeError("Cannot infer hidden size.")


def make_position_ids(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1).contiguous()


def build_position_embeddings(model: nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb = model.model.rotary_emb.to(hidden_states.device)
        return rotary_emb(hidden_states, position_ids)

    layers = get_layers(model)
    attn = layers[0].self_attn if hasattr(layers[0], "self_attn") else None
    if attn is not None and hasattr(attn, "rotary_emb"):
        rotary_emb = attn.rotary_emb.to(hidden_states.device)
        return rotary_emb(hidden_states, position_ids)

    return None


def run_layer_once(layer, hidden_states, position_ids=None, position_embeddings=None, attention_mask=None):
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    try:
        out = layer(hidden_states, **kwargs)
    except TypeError:
        kwargs.pop("position_ids", None)
        try:
            out = layer(hidden_states, **kwargs)
        except TypeError:
            kwargs.pop("position_embeddings", None)
            try:
                out = layer(hidden_states, **kwargs)
            except TypeError:
                kwargs.pop("attention_mask", None)
                out = layer(hidden_states)

    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def bench_layer(layer, hidden_states, position_ids, position_embeddings, attention_mask, warmup: int, iters: int, timing: str) -> float:
    device = hidden_states.device

    for _ in range(max(3, warmup // 2)):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings, attention_mask)
    torch.cuda.synchronize(device)

    if timing == "graph":
        static_x = hidden_states.clone()
        static_pos = None if position_ids is None else position_ids.clone()
        static_pe = None if position_embeddings is None else tuple(t.clone() for t in position_embeddings)
        static_mask = None if attention_mask is None else attention_mask.clone()

        for _ in range(warmup):
            _ = run_layer_once(layer, static_x, static_pos, static_pe, static_mask)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _ = run_layer_once(layer, static_x, static_pos, static_pe, static_mask)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / iters)

    for _ in range(warmup):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings, attention_mask)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings, attention_mask)
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


def collect_layer_targets(model: nn.Module, layer_idx: int) -> List[Tuple[str, nn.Module]]:
    prefix = f"model.layers.{layer_idx}."
    targets: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if not name.startswith(prefix):
            continue
        if not any(name.endswith(suffix) or name.endswith(suffix + ".module") for suffix in TARGET_SUFFIXES):
            continue
        linear = REAL.unwrap_linear(module)
        if linear.weight is None or linear.weight.dim() != 2:
            continue
        targets.append((name, module))
    return targets


def make_policy_cfg(ratio: float) -> dict:
    return {
        "ratio": float(ratio),
        "ratio_continuous": float(ratio),
        "activation_percentile": 100.0,
        "weight_percentile": 100.0,
    }


def make_rtn_scale(weight_cpu: torch.Tensor, eps: float) -> torch.Tensor:
    max_abs = weight_cpu.detach().float().abs().amax(dim=1)
    return (max_abs / 7.0).clamp_min(float(eps)).contiguous()


def load_real_split_extensions(cutlass_path: Optional[str], verbose: bool):
    cutlass = REAL.BASE.find_cutlass_path(cutlass_path)
    log(f"[CUTLASS] {cutlass}")
    main_ext = REAL.BASE.load_ext(cutlass, verbose=verbose)
    layout_ext = REAL.BASE.load_layout_ext(verbose)
    policy_pack_ext = REAL.load_policy_pack_ext(verbose)
    return main_ext, layout_ext, policy_pack_ext


def patch_layer_with_real_split_no_gptq(
    *,
    model: nn.Module,
    layer_idx: int,
    ratio: float,
    eps: float,
    device: torch.device,
    main_ext,
    layout_ext,
    policy_pack_ext,
) -> dict:
    targets = collect_layer_targets(model, layer_idx)
    if len(targets) != len(TARGET_SUFFIXES):
        raise RuntimeError(
            f"Expected {len(TARGET_SUFFIXES)} layer Linear targets, got {len(targets)}: "
            f"{[name for name, _ in targets]}"
        )

    max_r_by_shape: Dict[Tuple[int, int], int] = {}
    for _, module in targets:
        linear = REAL.unwrap_linear(module)
        N, K = map(int, linear.weight.shape)
        R = REAL.BASE.ceil_ratio_count(K, ratio)
        if R <= 0:
            raise RuntimeError(f"split_ratio={ratio} gives R=0 for K={K}")
        max_r_by_shape[(K, N)] = max(max_r_by_shape.get((K, N), 0), R)

    scratch_pool = REAL.BASE.SharedScratchPool(
        device=device,
        max_r_by_shape=max_r_by_shape,
        split=True,
    )

    records = []
    policy_cfg = make_policy_cfg(ratio)
    for index, (raw_name, module) in enumerate(targets, 1):
        linear = REAL.unwrap_linear(module)
        weight_cpu = linear.weight.detach().cpu()
        bias_cpu = None if linear.bias is None else linear.bias.detach().cpu()
        rtn_scale_cpu = make_rtn_scale(weight_cpu, eps)

        replacement = REAL.RealPolicyLinear(
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            mode="dual_policy",
            weight_cpu=weight_cpu,
            bias_cpu=bias_cpu,
            policy_cfg=policy_cfg,
            gptq_scale_cpu=rtn_scale_cpu,
            eps=eps,
            device=device,
            name=raw_name,
            scratch_pool=scratch_pool,
            prefetch_workspace=None,
            rotate_online=False,
            had_k=None,
            had_factor=1,
        )
        REAL.BASE.set_submodule(model, raw_name, replacement)
        records.append({
            "name": replacement.name,
            "raw_name": raw_name,
            "K": replacement.K,
            "N": replacement.N,
            "R": replacement.R,
            "ratio": replacement.ratio,
            "mode": replacement.mode,
            "weight_scale_mode": "rtn_maxabs",
            "weight_pack_rel_l2": replacement.weight_pack_rel_l2,
            "weight_pack_max_abs": replacement.weight_pack_max_abs,
            "persistent_weight_bytes": replacement.persistent_weight_bytes,
        })
        log(
            f"[PATCH {index}/{len(targets)}] {raw_name} "
            f"K={replacement.K} N={replacement.N} R={replacement.R} "
            f"pack_rel={replacement.weight_pack_rel_l2:.3e}"
        )
        del module, linear, weight_cpu, bias_cpu, rtn_scale_cpu
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "records": records,
        "persistent_packed_weight_bytes": int(sum(r["persistent_weight_bytes"] for r in records)),
        "scratch_allocated_bytes_after_patch": int(scratch_pool.allocated_bytes()),
        "max_r_by_shape": [
            {"K": int(k), "N": int(n), "max_R": int(r)}
            for (k, n), r in sorted(max_r_by_shape.items())
        ],
    }


def load_model_for_layer(model_name: str, dtype: torch.dtype) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model


def bench_cases(model, layer, hidden_size: int, batches: List[int], seq_len: int, dtype: torch.dtype, device: torch.device, warmup: int, iters: int, timing: str, label: str) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for batch in batches:
        log(f"\n[CASE:{label}] batch={batch} seq_len={seq_len}")
        hidden_states = torch.randn(batch, seq_len, hidden_size, device=device, dtype=dtype)
        position_ids = make_position_ids(batch, seq_len, device)
        position_embeddings = build_position_embeddings(model, hidden_states, position_ids)
        if position_embeddings is None:
            log("[WARN] position_embeddings=None")
        else:
            log("[POSITION_EMBEDDINGS] " + str([tuple(t.shape) for t in position_embeddings]))
        torch.cuda.empty_cache()
        ms = bench_layer(
            layer,
            hidden_states,
            position_ids,
            position_embeddings,
            None,
            warmup=warmup,
            iters=iters,
            timing=timing,
        )
        result[batch] = ms
        log(f"[RESULT:{label}] batch={batch} ms={ms:.6f}")
        del hidden_states, position_ids, position_embeddings
        torch.cuda.empty_cache()
    return result


def write_outputs(out_dir: Path, rows: List[dict], patch_info: Optional[dict], metadata: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "bf16_vs_split_layer_latency_no_gptq_v1.csv"
    json_path = out_dir / "bf16_vs_split_layer_latency_no_gptq_v1.json"
    meta_path = out_dir / "bf16_vs_split_layer_latency_no_gptq_v1_meta.json"

    if rows:
        fieldnames = [
            "model",
            "layer_idx",
            "batch",
            "seq_len",
            "hidden_size",
            "split_ratio",
            "bf16_ms",
            "split_ms",
            "speedup_bf16_over_split",
            "normalized_split_latency",
            "timing",
            "weight_scale_mode",
            "note",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        json.dump(rows, open(json_path, "w"), indent=2)
        log(f"[CSV] {csv_path}")
        log(f"[JSON] {json_path}")

    payload = dict(metadata)
    if patch_info is not None:
        payload["patch_info"] = patch_info
    json.dump(payload, open(meta_path, "w"), indent=2)
    log(f"[META] {meta_path}")


def cleanup_model(model=None, layer=None) -> None:
    del layer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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
    log(f"[SPLIT_RATIO] {args.split_ratio}")
    log(f"[DEVICE] {device}")
    log(f"[TIMING] {args.timing}")
    log(f"[NOTE] {NOTE}")

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    bf16_ms: Dict[int, float] = {}
    split_ms: Dict[int, float] = {}
    patch_info = None
    hidden_size = None
    num_layers = None

    if not args.skip_bf16:
        log("\n[LOAD_BF16_MODEL]")
        model = load_model_for_layer(args.model, torch.bfloat16)
        hidden_size = infer_hidden_size(model)
        layers = get_layers(model)
        num_layers = len(layers)
        if args.layer_idx < 0 or args.layer_idx >= num_layers:
            raise ValueError(f"layer_idx out of range: {args.layer_idx}, num_layers={num_layers}")
        layer = layers[args.layer_idx].to(device=device, dtype=torch.bfloat16).eval()
        log(f"[HIDDEN_SIZE] {hidden_size}")
        log(f"[NUM_LAYERS] {num_layers}")
        log(f"[BF16_LAYER_FORWARD_SIGNATURE] {inspect.signature(layer.forward)}")
        bf16_ms = bench_cases(
            model,
            layer,
            hidden_size,
            batches,
            args.seq_len,
            torch.bfloat16,
            device,
            args.warmup,
            args.iters,
            args.timing,
            "bf16",
        )
        cleanup_model(model, layer)

    if not args.skip_split:
        log("\n[LOAD_SPLIT_MODEL_FP16]")
        main_ext, layout_ext, policy_pack_ext = load_real_split_extensions(args.cutlass_path, args.verbose_compile)
        model = load_model_for_layer(args.model, torch.float16)
        hidden_size = infer_hidden_size(model) if hidden_size is None else hidden_size
        layers = get_layers(model)
        num_layers = len(layers) if num_layers is None else num_layers
        layer = layers[args.layer_idx].to(device=device, dtype=torch.float16).eval()
        patch_info = patch_layer_with_real_split_no_gptq(
            model=model,
            layer_idx=args.layer_idx,
            ratio=args.split_ratio,
            eps=args.eps,
            device=device,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
        )
        layer = get_layers(model)[args.layer_idx].eval()
        log(f"[SPLIT_LAYER_FORWARD_SIGNATURE] {inspect.signature(layer.forward)}")
        split_ms = bench_cases(
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
            "split",
        )
        cleanup_model(model, layer)

    rows: List[dict] = []
    for batch in batches:
        b_ms = bf16_ms.get(batch)
        s_ms = split_ms.get(batch)
        speedup = None if b_ms is None or s_ms is None else float(b_ms / s_ms)
        norm = None if b_ms is None or s_ms is None else float(s_ms / b_ms)
        rows.append({
            "model": args.model,
            "layer_idx": args.layer_idx,
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            "split_ratio": args.split_ratio,
            "bf16_ms": b_ms,
            "split_ms": s_ms,
            "speedup_bf16_over_split": speedup,
            "normalized_split_latency": norm,
            "timing": args.timing,
            "weight_scale_mode": "rtn_maxabs",
            "note": NOTE,
        })

    metadata = {
        "model": args.model,
        "layer_idx": args.layer_idx,
        "seq_len": args.seq_len,
        "batches": batches,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "split_ratio": args.split_ratio,
        "timing": args.timing,
        "warmup": args.warmup,
        "iters": args.iters,
        "weight_scale_mode": "rtn_maxabs_per_output_maxabs_div_7",
        "split_runtime": "RealPolicyLinear_dual_policy",
        "storage_mode": "dual",
        "rotate_online": False,
        "prefetch_workspace": None,
        "note": NOTE,
    }
    write_outputs(out_dir, rows, patch_info, metadata)
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
