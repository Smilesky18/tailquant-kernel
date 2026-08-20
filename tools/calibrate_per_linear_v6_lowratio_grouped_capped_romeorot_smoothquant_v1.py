#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped-capped low-ratio search with RoMeO-style rotation + SmoothQuant.

Standalone wrapper. It keeps original search files unchanged, applies
RoMeO-style offline rotation, calibrates SmoothQuant scales on the rotated FP16
graph, absorbs the scales into weights, and then runs the grouped-capped search.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts"):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import calibrate_per_linear_v6_lowratio_grouped_capped_romeorot_v1 as RRS  # noqa: E402
import eval_policy_v6_weightmode_fp16_romeorot_smoothquant_v1 as SQR  # noqa: E402
import eval_policy_v6_weightmode_fp16_romeorot_smoke_v1 as RR  # noqa: E402


V3 = RRS.V3
V6 = RRS.V6
H = RRS.H

_ORIGINAL_V3_PARSE_ARGS = V3.parse_args
_SQ_SCALE_BY_FULL_NAME: Dict[str, torch.Tensor] = {}


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
        args = _ORIGINAL_V3_PARSE_ARGS()
    finally:
        sys.argv = old_argv
    args.smooth_quant = True
    args.smooth_quant_alpha = sq_args.smooth_quant_alpha
    args.smooth_quant_nsamples = sq_args.smooth_quant_nsamples
    args.smooth_quant_seqlen = sq_args.smooth_quant_seqlen
    args.smooth_quant_min_scale = sq_args.smooth_quant_min_scale
    args.smooth_quant_dataset = sq_args.smooth_quant_dataset or args.dataset
    return args


def apply_offline_weight_rotation_romeorot_smoothquant(model, rot_flags):
    had_cache = RR.apply_romeo_offline_weight_rotation(model, rot_flags)
    setattr(model, "_romeorot_had_cache", had_cache)
    if V3.LAST_ARGS is None:
        raise RuntimeError("Search args were not initialized before SmoothQuant")
    args = V3.LAST_ARGS
    act_max_by_name = SQR.calibrate_smoothquant_inputs(model, args, rot_flags, had_cache)
    global _SQ_SCALE_BY_FULL_NAME
    _SQ_SCALE_BY_FULL_NAME = SQR.apply_smoothquant_weight_scale_(model, act_max_by_name, args)
    SQR.save_smoothquant_summary(args, act_max_by_name, _SQ_SCALE_BY_FULL_NAME)
    model.cpu()
    torch.cuda.empty_cache()

    RRS.build_norm_scale_table(model, rot_flags)
    print(
        f"[RoMeoSmoothQuantSearch] smoothed_modules={len(_SQ_SCALE_BY_FULL_NAME)} "
        f"alpha={args.smooth_quant_alpha} dataset={args.smooth_quant_dataset} "
        f"nsamples={args.smooth_quant_nsamples} seqlen={args.smooth_quant_seqlen}",
        flush=True,
    )
    return had_cache


def register_romeorot_smoothquant_hooks_for_layer(
    linears: dict,
    layer_id: int,
    rot_flags: dict,
    had_cache: Dict[int, Tuple[torch.Tensor, int]],
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for local_name, module in linears.items():
        full_name = f"model.layers.{layer_id}.{local_name}"
        rotate_selected = bool(rot_flags.get(full_name, False))
        sq_scale = _SQ_SCALE_BY_FULL_NAME.get(full_name)
        if not rotate_selected and sq_scale is None:
            continue

        if rotate_selected and RR.is_up_type(full_name):
            norm_scale = RRS._NORM_SCALE_BY_FULL_NAME[full_name]

            def make_up_hook(local_norm_scale: torch.Tensor, local_sq_scale: torch.Tensor | None):
                def hook(_module, inputs):
                    x = inputs[0]
                    norm = local_norm_scale.to(device=x.device, dtype=x.dtype)
                    x = x / norm
                    if local_sq_scale is not None:
                        sq = local_sq_scale.to(device=x.device, dtype=x.dtype)
                        x = x / sq
                    return (x,) + inputs[1:]

                return hook

            handles.append(module.register_forward_pre_hook(make_up_hook(norm_scale, sq_scale)))
            continue

        if rotate_selected and RR.is_down_type(full_name):
            in_dim = int(module.weight.shape[1])
            had_k, k = RR.get_had_cached(had_cache, in_dim)

            def make_down_hook(local_had_k, local_k, local_sq_scale: torch.Tensor | None):
                def hook(_module, inputs):
                    x = H.apply_hadamard_last_dim(inputs[0], local_had_k, local_k)
                    if local_sq_scale is not None:
                        sq = local_sq_scale.to(device=x.device, dtype=x.dtype)
                        x = x / sq
                    return (x,) + inputs[1:]

                return hook

            handles.append(module.register_forward_pre_hook(make_down_hook(had_k, k, sq_scale)))
            continue

        if sq_scale is not None:
            def make_sq_hook(local_sq_scale: torch.Tensor):
                def hook(_module, inputs):
                    sq = local_sq_scale.to(device=inputs[0].device, dtype=inputs[0].dtype)
                    return (inputs[0] / sq,) + inputs[1:]

                return hook

            handles.append(module.register_forward_pre_hook(make_sq_hook(sq_scale)))
    return handles


def main():
    V3.parse_args = parse_args_with_smoothquant
    RRS.apply_offline_weight_rotation_romeorot = apply_offline_weight_rotation_romeorot_smoothquant
    RRS.register_romeo_rotation_hooks_for_layer = register_romeorot_smoothquant_hooks_for_layer
    try:
        RRS.main()
    finally:
        V3.parse_args = _ORIGINAL_V3_PARSE_ARGS


if __name__ == "__main__":
    main()
