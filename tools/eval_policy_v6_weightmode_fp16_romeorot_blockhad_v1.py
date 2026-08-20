#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unsmoothed RoMeO-style split eval with block-Hadamard fallback."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
for item in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import eval_policy_v6_weightmode_v1 as BASE  # noqa: E402
import eval_policy_v6_weightmode_fp16_romeorot_smoke_v1 as RR  # noqa: E402
from romeorot_blockhad_utils_v1 import HadPlan, apply_had_plan, get_had_plan  # noqa: E402


def apply_romeo_offline_weight_rotation_blockhad(model, rot_flags: Dict[str, bool]) -> Dict[int, HadPlan]:
    targets = BASE.H.collect_target_modules(model)
    had_cache: Dict[int, HadPlan] = {}
    for name, module in targets.items():
        if not rot_flags.get(name, False):
            continue
        out_dim, in_dim = map(int, module.weight.data.shape)
        in_plan = get_had_plan(BASE.H, had_cache, in_dim)
        w_dtype = module.weight.data.dtype
        w = module.weight.data.to(dtype=torch.float32, device=BASE.H.utils.DEV)
        if RR.is_up_type(name):
            norm_scale = RR.stabilize_norm_scale(
                RR.get_norm_scale_for_module(model, name)
            ).to(dtype=torch.float32, device=BASE.H.utils.DEV)
            w = w * norm_scale.unsqueeze(0)
        w = apply_had_plan(BASE.H, w, in_plan)
        if RR.is_down_type(name):
            out_plan = get_had_plan(BASE.H, had_cache, out_dim)
            w = apply_had_plan(BASE.H, w.transpose(-1, -2), out_plan).transpose(-1, -2)
        module.weight.data = w.to(dtype=w_dtype, device="cpu")
    return had_cache


def register_romeo_online_rotation_hooks_blockhad(
    model,
    rot_flags: Dict[str, bool],
    had_cache: Dict[int, HadPlan],
) -> List[torch.utils.hooks.RemovableHandle]:
    targets = BASE.H.collect_target_modules(model)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    hidden_plan = get_had_plan(BASE.H, had_cache, int(model.config.hidden_size))
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    if embed_tokens is None or final_norm is None:
        raise RuntimeError("RoMeO block-Hadamard wrapper expects model.model.embed_tokens and model.model.norm")

    def embed_hook(_module, _inputs, output):
        return apply_had_plan(BASE.H, output, hidden_plan)

    def final_norm_pre_hook(_module, inputs):
        x = apply_had_plan(BASE.H, inputs[0], hidden_plan)
        return (x,) + inputs[1:]

    handles.append(embed_tokens.register_forward_hook(embed_hook))
    handles.append(final_norm.register_forward_pre_hook(final_norm_pre_hook))

    selected = 0
    block_dims = set()
    for name, module in targets.items():
        if not rot_flags.get(name, False):
            continue
        selected += 1
        if RR.is_up_type(name):
            norm_scale = RR.stabilize_norm_scale(RR.get_norm_scale_for_module(model, name).detach())

            def make_norm_hook(local_norm_scale: torch.Tensor):
                def _pre_hook(_m, inp):
                    scale = local_norm_scale.to(device=inp[0].device, dtype=inp[0].dtype)
                    return (inp[0] / scale,) + inp[1:]
                return _pre_hook

            handles.append(module.register_forward_pre_hook(make_norm_hook(norm_scale)))
            continue
        plan = get_had_plan(BASE.H, had_cache, int(module.weight.data.shape[1]))
        if plan.is_block:
            block_dims.add(plan.dim)

        def make_hook(local_plan: HadPlan):
            def _pre_hook(_m, inp):
                return (apply_had_plan(BASE.H, inp[0], local_plan),) + inp[1:]
            return _pre_hook

        handles.append(module.register_forward_pre_hook(make_hook(plan)))
    print(
        f"[RoMeoRotateOptSplitBlockHad] selected={selected}/{len(rot_flags)} "
        f"block_fallback_dims={sorted(block_dims)} embed_final_hooks=2",
        flush=True,
    )
    return handles


def main():
    original_get_model = BASE.H.model_utils.get_model
    original_offline_rotation = BASE.H.apply_offline_weight_rotation
    original_online_hooks = BASE.H.register_online_rotation_hooks

    def get_model_fp16(*args, **kwargs):
        model = original_get_model(*args, **kwargs)
        if os.environ.get("ROMEO_ROT_SKIP_FP16_CAST", "0") == "1":
            return model
        return RR.cast_floating_parameters_to_fp16(model)

    BASE.H.model_utils.get_model = get_model_fp16
    BASE.H.apply_offline_weight_rotation = apply_romeo_offline_weight_rotation_blockhad
    BASE.H.register_online_rotation_hooks = register_romeo_online_rotation_hooks_blockhad
    try:
        BASE.main()
    finally:
        BASE.H.model_utils.get_model = original_get_model
        BASE.H.apply_offline_weight_rotation = original_offline_rotation
        BASE.H.register_online_rotation_hooks = original_online_hooks


if __name__ == "__main__":
    main()
