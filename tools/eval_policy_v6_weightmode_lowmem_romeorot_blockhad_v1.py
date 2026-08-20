#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-memory GPTQ/PPL eval with unsmoothed RoMeO block-Hadamard rotation."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
LOWMEM = ROOT / "kernel_quant/lowmem70b"
for item in (TOOLS, LOWMEM, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts"):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import eval_policy_v6_weightmode_lowmem70b_isolated_v12_twostage as BASE  # noqa: E402
from romeorot_blockhad_utils_v1 import HadPlan, apply_had_plan, get_had_plan  # noqa: E402

H = BASE.H
NORM_SCALE_EPS = float(os.environ.get("ROMEO_ROT_NORM_SCALE_EPS", "1e-3"))


def is_down_type(name: str) -> bool:
    return name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj")


def is_up_type(name: str) -> bool:
    return not is_down_type(name)


def get_norm_scale_for_module(model, name: str) -> torch.Tensor:
    layer_id = H.get_layer_id_from_module_name(name)
    layer = model.model.layers[layer_id]
    if ".self_attn." in name:
        return layer.input_layernorm.weight.data
    if ".mlp." in name:
        return layer.post_attention_layernorm.weight.data
    raise ValueError(f"Cannot infer norm scale for module: {name}")


def stabilize_norm_scale(norm_scale: torch.Tensor) -> torch.Tensor:
    scale = norm_scale.to(dtype=torch.float32)
    sign = torch.where(scale < 0, -torch.ones_like(scale), torch.ones_like(scale))
    return sign * scale.abs().clamp_min(NORM_SCALE_EPS)


def apply_romeo_offline_weight_rotation_blockhad_lowmem(model, rot_flags: Dict[str, bool]) -> Dict[int, HadPlan]:
    targets = H.collect_target_modules(model)
    had_cache: Dict[int, HadPlan] = {}
    for name, module in targets.items():
        if not rot_flags.get(name, False):
            continue
        out_dim, in_dim = map(int, module.weight.data.shape)
        in_plan = get_had_plan(H, had_cache, in_dim)
        w_dtype = module.weight.data.dtype
        w = module.weight.data.to(dtype=torch.float32, device=H.utils.DEV)
        if is_up_type(name):
            norm_scale = stabilize_norm_scale(get_norm_scale_for_module(model, name)).to(dtype=torch.float32, device=H.utils.DEV)
            w = w * norm_scale.unsqueeze(0)
        w = apply_had_plan(H, w, in_plan)
        if is_down_type(name):
            out_plan = get_had_plan(H, had_cache, out_dim)
            w = apply_had_plan(H, w.transpose(-1, -2), out_plan).transpose(-1, -2)
        module.weight.data = w.to(dtype=w_dtype, device="cpu")
    print(
        f"[LOWMEM-RoMeOBlockHad] offline selected={sum(rot_flags.values())}/{len(rot_flags)} "
        f"block_fallback_dims={sorted(k for k, v in had_cache.items() if v.is_block)}",
        flush=True,
    )
    return had_cache


def register_romeo_online_rotation_hooks_blockhad_lowmem(
    model,
    rot_flags: Dict[str, bool],
    had_cache: Dict[int, HadPlan],
) -> List[torch.utils.hooks.RemovableHandle]:
    targets = H.collect_target_modules(model)
    handles: List[torch.utils.hooks.RemovableHandle] = []
    hidden_plan = get_had_plan(H, had_cache, int(model.config.hidden_size))
    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    final_norm = getattr(getattr(model, "model", None), "norm", None)
    if embed_tokens is None or final_norm is None:
        raise RuntimeError("Lowmem RoMeO block-Hadamard wrapper expects model.model.embed_tokens and model.model.norm")

    def embed_hook(_module, _inputs, output):
        return apply_had_plan(H, output, hidden_plan)

    def final_norm_pre_hook(_module, inputs):
        return (apply_had_plan(H, inputs[0], hidden_plan),) + inputs[1:]

    handles.append(embed_tokens.register_forward_hook(embed_hook))
    handles.append(final_norm.register_forward_pre_hook(final_norm_pre_hook))
    selected = 0
    for name, module in targets.items():
        if not rot_flags.get(name, False):
            continue
        selected += 1
        if is_up_type(name):
            norm_scale = stabilize_norm_scale(get_norm_scale_for_module(model, name).detach())

            def make_norm_hook(local_norm_scale: torch.Tensor):
                def _pre_hook(_m, inp):
                    scale = local_norm_scale.to(device=inp[0].device, dtype=inp[0].dtype)
                    return (inp[0] / scale,) + inp[1:]
                return _pre_hook

            handles.append(module.register_forward_pre_hook(make_norm_hook(norm_scale)))
            continue
        plan = get_had_plan(H, had_cache, int(module.weight.data.shape[1]))

        def make_hook(local_plan: HadPlan):
            def _pre_hook(_m, inp):
                return (apply_had_plan(H, inp[0], local_plan),) + inp[1:]
            return _pre_hook

        handles.append(module.register_forward_pre_hook(make_hook(plan)))
    print(f"[LOWMEM-RoMeOBlockHad] online selected={selected}/{len(rot_flags)} embed_final_hooks=2", flush=True)
    return handles


_ORIGINAL_OFFLINE_ROTATION = H.apply_offline_weight_rotation
_ORIGINAL_ONLINE_HOOKS = H.register_online_rotation_hooks


def main():
    H.apply_offline_weight_rotation = apply_romeo_offline_weight_rotation_blockhad_lowmem
    H.register_online_rotation_hooks = register_romeo_online_rotation_hooks_blockhad_lowmem
    try:
        BASE.main()
    finally:
        H.apply_offline_weight_rotation = _ORIGINAL_OFFLINE_ROTATION
        H.register_online_rotation_hooks = _ORIGINAL_ONLINE_HOOKS


if __name__ == "__main__":
    main()
