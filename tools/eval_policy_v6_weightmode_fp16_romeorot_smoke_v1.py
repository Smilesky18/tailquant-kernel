#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FP16 PPL eval with RoMeo-style rotate_opt applied to split policy eval.

This is a standalone wrapper. It monkey-patches the v6 evaluator at runtime and
does not modify the original evaluator or model code.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import eval_policy_v6_weightmode_v1 as BASE  # noqa: E402


NORM_SCALE_EPS = float(os.environ.get("ROMEO_ROT_NORM_SCALE_EPS", "1e-3"))


def cast_floating_parameters_to_fp16(model):
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float16:
                parameter.data = parameter.data.to(dtype=torch.float16)
    return model


def is_down_type(name: str) -> bool:
    return name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj")


def is_up_type(name: str) -> bool:
    return not is_down_type(name)


def get_norm_scale_for_module(model, name: str) -> torch.Tensor:
    layer_id = BASE.H.get_layer_id_from_module_name(name)
    layer = model.model.layers[layer_id]
    if ".self_attn." in name:
        return layer.input_layernorm.weight.data
    if ".mlp." in name:
        return layer.post_attention_layernorm.weight.data
    raise ValueError(f"Cannot infer norm scale for module: {name}")


def stabilize_norm_scale(norm_scale: torch.Tensor) -> torch.Tensor:
    """Signed floor for RoMeo-style norm division in fp16 eval.

    Some Llama RMSNorm gamma entries are tiny. Dividing fp16 activations by the
    raw gamma can overflow before the following Linear has a chance to multiply
    by the offline-absorbed gamma. Use the same stabilized scale on the offline
    weight side and online activation side.
    """
    scale = norm_scale.to(dtype=torch.float32)
    sign = torch.where(scale < 0, -torch.ones_like(scale), torch.ones_like(scale))
    return sign * scale.abs().clamp_min(NORM_SCALE_EPS)


def get_had_cached(
    had_cache: Dict[int, Tuple[torch.Tensor, int]],
    dim: int,
) -> Tuple[torch.Tensor, int]:
    if dim not in had_cache:
        had_cache[dim] = BASE.H.hadamard_utils.get_hadK(dim)
    return had_cache[dim]


def apply_last_dim_hadamard_on_device(x: torch.Tensor, had_k, k: int, device) -> torch.Tensor:
    x_dtype = x.dtype
    y = x.to(dtype=torch.float32, device=device)
    y = BASE.H.apply_hadamard_last_dim(y, had_k, k)
    return y.to(dtype=x_dtype, device=x.device)


def apply_romeo_offline_weight_rotation(
    model,
    rot_flags: Dict[str, bool],
) -> Dict[int, Tuple[torch.Tensor, int]]:
    targets = BASE.H.collect_target_modules(model)
    had_cache: Dict[int, Tuple[torch.Tensor, int]] = {}

    for name, module in targets.items():
        if not rot_flags.get(name, False):
            continue

        out_dim, in_dim = map(int, module.weight.data.shape)
        in_had, in_k = get_had_cached(had_cache, in_dim)
        w_dtype = module.weight.data.dtype
        w = module.weight.data.to(dtype=torch.float32, device=BASE.H.utils.DEV)

        if is_up_type(name):
            norm_scale = stabilize_norm_scale(
                get_norm_scale_for_module(model, name)
            ).to(dtype=torch.float32, device=BASE.H.utils.DEV)
            w = w * norm_scale.unsqueeze(0)

        # All selected linears absorb the input-basis rotation into W.
        w = BASE.H.apply_hadamard_last_dim(w, in_had, in_k)

        # Down-type linears also emit the rotated residual-stream basis.
        if is_down_type(name):
            out_had, out_k = get_had_cached(had_cache, out_dim)
            w = BASE.H.apply_hadamard_last_dim(w.transpose(-1, -2), out_had, out_k).transpose(-1, -2)

        module.weight.data = w.to(dtype=w_dtype, device="cpu")

    return had_cache


def register_romeo_online_rotation_hooks(
    model,
    rot_flags: Dict[str, bool],
    had_cache: Dict[int, Tuple[torch.Tensor, int]],
) -> List[torch.utils.hooks.RemovableHandle]:
    targets = BASE.H.collect_target_modules(model)
    handles: List[torch.utils.hooks.RemovableHandle] = []

    hidden_size = int(model.config.hidden_size)
    hidden_had, hidden_k = get_had_cached(had_cache, hidden_size)

    embed_tokens = getattr(getattr(model, "model", None), "embed_tokens", None)
    final_norm = getattr(getattr(model, "model", None), "norm", None)

    if embed_tokens is None or final_norm is None:
        raise RuntimeError("RoMeo-style rotate_opt wrapper expects model.model.embed_tokens and model.model.norm")

    def embed_hook(_module, _inputs, output):
        return BASE.H.apply_hadamard_last_dim(output, hidden_had, hidden_k)

    def final_norm_pre_hook(_module, inputs):
        x = BASE.H.apply_hadamard_last_dim(inputs[0], hidden_had, hidden_k)
        return (x,) + inputs[1:]

    handles.append(embed_tokens.register_forward_hook(embed_hook))
    handles.append(final_norm.register_forward_pre_hook(final_norm_pre_hook))

    selected = 0
    down_online = 0
    up_norm_div = 0
    stabilized_entries = 0
    for name, module in targets.items():
        if not rot_flags.get(name, False):
            continue
        selected += 1
        if is_up_type(name):
            up_norm_div += 1
            raw_norm_scale = get_norm_scale_for_module(model, name).detach()
            stabilized_entries += int((raw_norm_scale.float().abs() < NORM_SCALE_EPS).sum().item())
            norm_scale = stabilize_norm_scale(raw_norm_scale)

            def make_norm_hook(local_norm_scale: torch.Tensor):
                def _pre_hook(_m, inp):
                    scale = local_norm_scale.to(device=inp[0].device, dtype=inp[0].dtype)
                    return (inp[0] / scale,) + inp[1:]

                return _pre_hook

            handles.append(module.register_forward_pre_hook(make_norm_hook(norm_scale)))
            continue

        down_online += 1
        in_dim = int(module.weight.data.shape[1])
        had_k, k = get_had_cached(had_cache, in_dim)

        def make_hook(local_had_k, local_k):
            def _pre_hook(_m, inp):
                x = BASE.H.apply_hadamard_last_dim(inp[0], local_had_k, local_k)
                return (x,) + inp[1:]

            return _pre_hook

        handles.append(module.register_forward_pre_hook(make_hook(had_k, k)))

    print(
        f"[RoMeoRotateOptSplit] selected={selected}/{len(rot_flags)} "
        f"up_norm_div={up_norm_div} down_online={down_online} "
        f"norm_scale_eps={NORM_SCALE_EPS:g} stabilized_entries={stabilized_entries} "
        f"embed_final_hooks=2"
    )
    return handles


def main():
    original_get_model = BASE.H.model_utils.get_model
    original_offline_rotation = BASE.H.apply_offline_weight_rotation
    original_online_hooks = BASE.H.register_online_rotation_hooks

    def get_model_fp16(*args, **kwargs):
        return cast_floating_parameters_to_fp16(original_get_model(*args, **kwargs))

    BASE.H.model_utils.get_model = get_model_fp16
    BASE.H.apply_offline_weight_rotation = apply_romeo_offline_weight_rotation
    BASE.H.register_online_rotation_hooks = register_romeo_online_rotation_hooks
    try:
        BASE.main()
    finally:
        BASE.H.model_utils.get_model = original_get_model
        BASE.H.apply_offline_weight_rotation = original_offline_rotation
        BASE.H.register_online_rotation_hooks = original_online_hooks


if __name__ == "__main__":
    main()
