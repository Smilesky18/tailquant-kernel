#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RoMeO-style rotation split PPL eval with offline SmoothQuant.

Standalone wrapper. It reuses the existing RoMeO-style rotation split eval and
adds a SmoothQuant-style offline weight/input scale transform without modifying
the original evaluator or model code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
for item in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import eval_policy_v6_weightmode_fp16_romeorot_smoke_v1 as RR  # noqa: E402


BASE = RR.BASE
H = BASE.H


_ORIGINAL_PARSE_ARGS = BASE.parse_args
_ORIGINAL_GET_MODEL = H.model_utils.get_model
_ORIGINAL_OFFLINE_ROTATION = H.apply_offline_weight_rotation
_ORIGINAL_ONLINE_HOOKS = H.register_online_rotation_hooks

_CURRENT_ARGS = None
_HAD_CACHE: Dict[int, Tuple[torch.Tensor, int]] = {}
_SQ_SCALE_BY_NAME: Dict[str, torch.Tensor] = {}
_SQ_SUMMARY: Dict[str, object] = {}


def parse_args_with_smoothquant():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smooth_quant_alpha", type=float, default=float(os.environ.get("ROMEO_SQ_ALPHA", "0.5")))
    parser.add_argument("--smooth_quant_nsamples", type=int, default=int(os.environ.get("ROMEO_SQ_NSAMPLES", "128")))
    parser.add_argument("--smooth_quant_seqlen", type=int, default=int(os.environ.get("ROMEO_SQ_SEQLEN", "512")))
    parser.add_argument("--smooth_quant_min_scale", type=float, default=float(os.environ.get("ROMEO_SQ_MIN_SCALE", "1e-5")))
    parser.add_argument("--smooth_quant_dataset", default=os.environ.get("ROMEO_SQ_DATASET", None))
    sq_args, remaining = parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = _ORIGINAL_PARSE_ARGS()
    finally:
        sys.argv = old_argv
    args.smooth_quant = True
    args.smooth_quant_alpha = sq_args.smooth_quant_alpha
    args.smooth_quant_nsamples = sq_args.smooth_quant_nsamples
    args.smooth_quant_seqlen = sq_args.smooth_quant_seqlen
    args.smooth_quant_min_scale = sq_args.smooth_quant_min_scale
    args.smooth_quant_dataset = sq_args.smooth_quant_dataset or args.cal_dataset
    global _CURRENT_ARGS
    _CURRENT_ARGS = args
    return args


def get_model_fp16(*args, **kwargs):
    return RR.cast_floating_parameters_to_fp16(_ORIGINAL_GET_MODEL(*args, **kwargs))


def _input_ids(value) -> torch.Tensor:
    if torch.is_tensor(value):
        t = value
    elif hasattr(value, "input_ids"):
        return _input_ids(value.input_ids)
    elif isinstance(value, dict) and "input_ids" in value:
        return _input_ids(value["input_ids"])
    elif isinstance(value, (list, tuple)):
        return _input_ids(value[0])
    else:
        t = torch.as_tensor(value)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def _max_abs_last_dim(x: torch.Tensor, width: int) -> torch.Tensor:
    return x.detach().reshape(-1, width).abs().amax(dim=0).float()


def _build_smoothquant_loader(args):
    dataset = str(args.smooth_quant_dataset)
    path = Path(dataset)
    if path.exists() and path.is_file():
        tokenizer_kwargs = {"use_fast": False}
        if args.hf_token is not None:
            tokenizer_kwargs["token"] = args.hf_token
        tokenizer = H.transformers.AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)
        raw = H.load_dataset("json", data_files=str(path), split="train")
        raw = raw.shuffle(seed=args.seed)
        count = min(int(args.smooth_quant_nsamples), len(raw))
        samples = []
        for idx in range(count):
            text = str(raw[idx].get("text", ""))
            encoded = tokenizer(
                text,
                return_tensors="pt",
                max_length=int(args.smooth_quant_seqlen),
                truncation=True,
            )
            ids = encoded["input_ids"]
            if ids.numel() == 0:
                continue
            samples.append(ids)
        if not samples:
            raise RuntimeError(f"No usable SmoothQuant calibration samples in {path}")
        print(f"[RoMeoSmoothQuant] json calibration path={path} samples={len(samples)}", flush=True)
        return samples

    return H.data_utils.get_loaders(
        dataset,
        nsamples=args.smooth_quant_nsamples,
        seed=args.seed,
        seqlen=args.smooth_quant_seqlen,
        model=args.model,
        hf_token=args.hf_token,
        eval_mode=False,
    )


def calibrate_smoothquant_inputs(model, args, rot_flags, had_cache) -> Dict[str, torch.Tensor]:
    device = torch.device("cuda:0")
    model.to(device).eval()
    rotation_handles = RR.register_romeo_online_rotation_hooks(model, rot_flags, had_cache)
    targets = H.collect_target_modules(model)
    max_by_name: Dict[str, torch.Tensor] = {}
    collect_handles: List[torch.utils.hooks.RemovableHandle] = []

    for name, module in targets.items():
        width = int(module.weight.data.shape[1])
        max_by_name[name] = torch.full((width,), 1e-5, dtype=torch.float32, device=device)

        def make_hook(local_name: str, local_width: int):
            def _hook(_module, inputs):
                current = _max_abs_last_dim(inputs[0], local_width).to(device)
                max_by_name[local_name] = torch.maximum(max_by_name[local_name], current)
                return None

            return _hook

        collect_handles.append(module.register_forward_pre_hook(make_hook(name, width)))

    loader = _build_smoothquant_loader(args)

    old_seqlen = getattr(model, "seqlen", None)
    model.seqlen = int(args.smooth_quant_seqlen)
    with torch.inference_mode():
        for sample_idx, sample in enumerate(loader):
            ids = _input_ids(sample).to(device)
            model(ids)
            if (sample_idx + 1) % 16 == 0 or sample_idx + 1 == len(loader):
                print(f"[RoMeoSmoothQuant] calibration {sample_idx + 1}/{len(loader)}", flush=True)
    if old_seqlen is not None:
        model.seqlen = old_seqlen

    for handle in collect_handles + rotation_handles:
        handle.remove()
    return {name: value.detach().cpu() for name, value in max_by_name.items()}


def apply_smoothquant_weight_scale_(model, act_max_by_name: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    if not 0.0 <= float(args.smooth_quant_alpha) <= 1.0:
        raise ValueError("--smooth_quant_alpha must be between 0 and 1")

    targets = H.collect_target_modules(model)
    scale_by_name: Dict[str, torch.Tensor] = {}
    alpha = float(args.smooth_quant_alpha)
    min_scale = float(args.smooth_quant_min_scale)

    with torch.no_grad():
        for name, module in targets.items():
            weight = module.weight.data
            act_max = act_max_by_name[name].to(device=weight.device, dtype=torch.float32)
            act_max = torch.nan_to_num(act_max, nan=1e-5, posinf=1e5, neginf=1e-5).clamp_min(1e-5)
            weight_max = weight.detach().abs().float().amax(dim=0)
            weight_max = torch.nan_to_num(weight_max, nan=1e-5, posinf=1e5, neginf=1e-5).clamp_min(1e-5)
            scale = (act_max.pow(alpha) / weight_max.pow(1.0 - alpha)).clamp_min(min_scale)
            scale = torch.nan_to_num(scale, nan=1.0, posinf=1e4, neginf=min_scale).clamp_min(min_scale)
            module.weight.data = (weight.float() * scale.unsqueeze(0)).to(dtype=weight.dtype)
            scale_by_name[name] = scale.detach().cpu()
    return scale_by_name


def save_smoothquant_summary(args, act_max_by_name: Dict[str, torch.Tensor], scale_by_name: Dict[str, torch.Tensor]):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale_path = out_dir / "smoothquant_scales.pt"
    torch.save(scale_by_name, scale_path)

    mins, maxs, means = [], [], []
    for scale in scale_by_name.values():
        s = scale.float()
        mins.append(float(s.min().item()))
        maxs.append(float(s.max().item()))
        means.append(float(s.mean().item()))

    summary = {
        "enabled": True,
        "alpha": float(args.smooth_quant_alpha),
        "dataset": args.smooth_quant_dataset,
        "nsamples": int(args.smooth_quant_nsamples),
        "seqlen": int(args.smooth_quant_seqlen),
        "min_scale": float(args.smooth_quant_min_scale),
        "num_modules": len(scale_by_name),
        "scale_file": str(scale_path),
        "scale_min": min(mins) if mins else None,
        "scale_max": max(maxs) if maxs else None,
        "scale_mean_avg": sum(means) / len(means) if means else None,
        "act_max_modules": len(act_max_by_name),
    }
    (out_dir / "smoothquant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    global _SQ_SUMMARY
    _SQ_SUMMARY = summary
    print(f"[RoMeoSmoothQuant] {json.dumps(summary, ensure_ascii=False)}", flush=True)


def apply_offline_weight_rotation_romeorot_smoothquant(model, rot_flags):
    had_cache = RR.apply_romeo_offline_weight_rotation(model, rot_flags)
    global _HAD_CACHE, _SQ_SCALE_BY_NAME
    _HAD_CACHE = had_cache
    args = _CURRENT_ARGS
    if args is None:
        raise RuntimeError("SmoothQuant args were not initialized")

    act_max_by_name = calibrate_smoothquant_inputs(model, args, rot_flags, had_cache)
    _SQ_SCALE_BY_NAME = apply_smoothquant_weight_scale_(model, act_max_by_name, args)
    save_smoothquant_summary(args, act_max_by_name, _SQ_SCALE_BY_NAME)
    model.cpu()
    torch.cuda.empty_cache()
    return had_cache


def register_romeo_smoothquant_online_hooks(model, rot_flags, had_cache):
    targets = H.collect_target_modules(model)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    hidden_size = int(model.config.hidden_size)
    hidden_had, hidden_k = RR.get_had_cached(had_cache, hidden_size)
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    if embed_tokens is None or final_norm is None:
        raise RuntimeError("RoMeO-style smoothquant wrapper expects model.model.embed_tokens and model.model.norm")

    def embed_hook(_module, _inputs, output):
        return H.apply_hadamard_last_dim(output, hidden_had, hidden_k)

    def final_norm_pre_hook(_module, inputs):
        x = H.apply_hadamard_last_dim(inputs[0], hidden_had, hidden_k)
        return (x,) + inputs[1:]

    handles.append(embed_tokens.register_forward_hook(embed_hook))
    handles.append(final_norm.register_forward_pre_hook(final_norm_pre_hook))

    selected = 0
    smoothed = 0
    for name, module in targets.items():
        rotate_selected = bool(rot_flags.get(name, False))
        sq_scale = _SQ_SCALE_BY_NAME.get(name)
        if sq_scale is None and not rotate_selected:
            continue
        if sq_scale is not None:
            smoothed += 1
        if rotate_selected:
            selected += 1

        if rotate_selected and RR.is_up_type(name):
            norm_scale = RR.stabilize_norm_scale(RR.get_norm_scale_for_module(model, name).detach())

            def make_up_hook(local_norm_scale: torch.Tensor, local_sq_scale: torch.Tensor | None):
                def _pre_hook(_m, inputs):
                    x = inputs[0]
                    norm = local_norm_scale.to(device=x.device, dtype=x.dtype)
                    x = x / norm
                    if local_sq_scale is not None:
                        sq = local_sq_scale.to(device=x.device, dtype=x.dtype)
                        x = x / sq
                    return (x,) + inputs[1:]

                return _pre_hook

            handles.append(module.register_forward_pre_hook(make_up_hook(norm_scale, sq_scale)))
            continue

        if rotate_selected and RR.is_down_type(name):
            in_dim = int(module.weight.data.shape[1])
            had_k, k = RR.get_had_cached(had_cache, in_dim)

            def make_down_hook(local_had_k, local_k, local_sq_scale: torch.Tensor | None):
                def _pre_hook(_m, inputs):
                    x = H.apply_hadamard_last_dim(inputs[0], local_had_k, local_k)
                    if local_sq_scale is not None:
                        sq = local_sq_scale.to(device=x.device, dtype=x.dtype)
                        x = x / sq
                    return (x,) + inputs[1:]

                return _pre_hook

            handles.append(module.register_forward_pre_hook(make_down_hook(had_k, k, sq_scale)))
            continue

        if sq_scale is not None:
            def make_sq_only_hook(local_sq_scale: torch.Tensor):
                def _pre_hook(_m, inputs):
                    sq = local_sq_scale.to(device=inputs[0].device, dtype=inputs[0].dtype)
                    return (inputs[0] / sq,) + inputs[1:]

                return _pre_hook

            handles.append(module.register_forward_pre_hook(make_sq_only_hook(sq_scale)))

    print(
        f"[RoMeoSmoothQuantHooks] rotate_selected={selected}/{len(rot_flags)} "
        f"smoothed_modules={smoothed} embed_final_hooks=2",
        flush=True,
    )
    return handles


def main():
    BASE.parse_args = parse_args_with_smoothquant
    H.model_utils.get_model = get_model_fp16
    H.apply_offline_weight_rotation = apply_offline_weight_rotation_romeorot_smoothquant
    H.register_online_rotation_hooks = register_romeo_smoothquant_online_hooks
    try:
        BASE.main()
    finally:
        BASE.parse_args = _ORIGINAL_PARSE_ARGS
        H.model_utils.get_model = _ORIGINAL_GET_MODEL
        H.apply_offline_weight_rotation = _ORIGINAL_OFFLINE_ROTATION
        H.register_online_rotation_hooks = _ORIGINAL_ONLINE_HOOKS


if __name__ == "__main__":
    main()
