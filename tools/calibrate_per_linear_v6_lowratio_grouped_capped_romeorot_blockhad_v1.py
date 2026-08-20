#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grouped-capped low-ratio search with unsmoothed RoMeO block-Hadamard rotation."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts"):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import calibrate_per_linear_v6_lowratio_grouped_capped_v3 as V3  # noqa: E402
import eval_policy_v6_weightmode_fp16_romeorot_smoke_v1 as RR  # noqa: E402
from eval_policy_v6_weightmode_fp16_romeorot_blockhad_v1 import (  # noqa: E402
    apply_romeo_offline_weight_rotation_blockhad,
)
from romeorot_blockhad_utils_v1 import HadPlan, apply_had_plan, get_had_plan  # noqa: E402

V6 = V3.V2.V1.V6
H = V6.H
_NORM_SCALE_BY_FULL_NAME: Dict[str, torch.Tensor] = {}


def build_norm_scale_table(model, rot_flags: dict):
    _NORM_SCALE_BY_FULL_NAME.clear()
    for name in H.collect_target_modules(model):
        if not rot_flags.get(name, False) or not RR.is_up_type(name):
            continue
        _NORM_SCALE_BY_FULL_NAME[name] = RR.stabilize_norm_scale(
            RR.get_norm_scale_for_module(model, name).detach()
        )


def maybe_limit_rot_flags(rot_flags: dict) -> dict:
    args = getattr(V3, "LAST_ARGS", None)
    max_layers = int(getattr(args, "max_layers", -1)) if args is not None else -1
    if max_layers < 0:
        return rot_flags
    limited = {}
    for name, enabled in rot_flags.items():
        try:
            layer_id = H.get_layer_id_from_module_name(name)
        except Exception:
            layer_id = -1
        limited[name] = bool(enabled and 0 <= layer_id < max_layers)
    return limited


def apply_offline_weight_rotation_search(model, rot_flags):
    rot_flags = maybe_limit_rot_flags(rot_flags)
    had_cache = apply_romeo_offline_weight_rotation_blockhad(model, rot_flags)
    setattr(model, "_romeorot_had_cache", had_cache)
    build_norm_scale_table(model, rot_flags)
    print(
        f"[RoMeoRotateOptSplitCappedSearchBlockHad] up_norm_scales={len(_NORM_SCALE_BY_FULL_NAME)} "
        f"block_fallback_dims={sorted(k for k, v in had_cache.items() if v.is_block)}",
        flush=True,
    )
    return had_cache


def register_romeo_rotation_hooks_for_layer(
    linears: dict,
    layer_id: int,
    rot_flags: dict,
    had_cache: Dict[int, HadPlan],
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for local_name, module in linears.items():
        full_name = f"model.layers.{layer_id}.{local_name}"
        if not rot_flags.get(full_name, False):
            continue
        if RR.is_up_type(full_name):
            norm_scale = _NORM_SCALE_BY_FULL_NAME[full_name]

            def make_norm_hook(local_norm_scale: torch.Tensor):
                def hook(_module, inputs):
                    scale = local_norm_scale.to(device=inputs[0].device, dtype=inputs[0].dtype)
                    return (inputs[0] / scale,) + inputs[1:]
                return hook

            handles.append(module.register_forward_pre_hook(make_norm_hook(norm_scale)))
            continue
        plan = get_had_plan(H, had_cache, int(module.weight.shape[1]))

        def make_had_hook(local_plan: HadPlan):
            def hook(_module, inputs):
                return (apply_had_plan(H, inputs[0], local_plan),) + inputs[1:]
            return hook

        handles.append(module.register_forward_pre_hook(make_had_hook(plan)))
    return handles


def capture_first_inputs_romeorot(model, loader, device, nsamples, seqlen):
    had_cache: Dict[int, HadPlan] = getattr(model, "_romeorot_had_cache", {})
    hidden_plan = get_had_plan(H, had_cache, int(model.config.hidden_size))
    setattr(model, "_romeorot_had_cache", had_cache)
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    if embed_tokens is None:
        raise RuntimeError("RoMeO-style capture expects model.model.embed_tokens")

    def embed_hook(_module, _inputs, output):
        return apply_had_plan(H, output, hidden_plan)

    handle = embed_tokens.register_forward_hook(embed_hook)
    try:
        return _ORIGINAL_CAPTURE_FIRST_INPUTS(model, loader, device, nsamples, seqlen)
    finally:
        handle.remove()


_ORIGINAL_GET_MODEL = H.model_utils.get_model
_ORIGINAL_OFFLINE_ROTATION = H.apply_offline_weight_rotation
_ORIGINAL_CAPTURE_FIRST_INPUTS = V6.capture_first_inputs
_ORIGINAL_REGISTER_LAYER_ROTATION = V6.register_rotation_hooks_for_layer


def get_model_maybe_fp16(*args, **kwargs):
    model = _ORIGINAL_GET_MODEL(*args, **kwargs)
    if os.environ.get("ROMEO_ROT_SKIP_FP16_CAST", "0") == "1":
        return model
    return RR.cast_floating_parameters_to_fp16(model)


def main():
    H.model_utils.get_model = get_model_maybe_fp16
    H.apply_offline_weight_rotation = apply_offline_weight_rotation_search
    V6.capture_first_inputs = capture_first_inputs_romeorot
    V6.register_rotation_hooks_for_layer = register_romeo_rotation_hooks_for_layer
    try:
        V3.main()
    finally:
        H.model_utils.get_model = _ORIGINAL_GET_MODEL
        H.apply_offline_weight_rotation = _ORIGINAL_OFFLINE_ROTATION
        V6.capture_first_inputs = _ORIGINAL_CAPTURE_FIRST_INPUTS
        V6.register_rotation_hooks_for_layer = _ORIGINAL_REGISTER_LAYER_ROTATION


if __name__ == "__main__":
    main()
