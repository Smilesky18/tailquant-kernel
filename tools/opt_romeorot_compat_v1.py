#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPT compatibility patches for split-policy calibration/eval wrappers.

The old v6 split scripts are written around Llama-style names. This module
exposes OPT linears through the same virtual names so the policy format and
grouped/shared-lambda postprocess can be reused without editing original files.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


OPT_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "fc1",
    "fc2",
)


def opt_layers(model):
    return model.model.decoder.layers


def virtual_local_name(real_local: str) -> str:
    if real_local == "self_attn.out_proj":
        return "self_attn.o_proj"
    if real_local == "fc1":
        return "mlp.up_proj"
    if real_local == "fc2":
        return "mlp.down_proj"
    return real_local


def real_local_from_virtual(virtual: str) -> str:
    if virtual == "self_attn.o_proj":
        return "self_attn.out_proj"
    if virtual == "mlp.up_proj":
        return "fc1"
    if virtual == "mlp.down_proj":
        return "fc2"
    return virtual


def opt_collect_target_modules(model) -> Dict[str, nn.Module]:
    out: Dict[str, nn.Module] = {}
    for layer_id, layer in enumerate(opt_layers(model)):
        for real_name, module in layer.named_modules():
            if real_name.endswith(OPT_SUFFIXES) and (
                isinstance(module, nn.Linear) or module.__class__.__name__ == "ActQuantWrapper"
            ):
                out[f"model.layers.{layer_id}.{virtual_local_name(real_name)}"] = module
    return out


def opt_local_linears(layer) -> Dict[str, nn.Linear]:
    out: Dict[str, nn.Linear] = {}
    for real_name, module in layer.named_modules():
        if isinstance(module, nn.Linear) and real_name.endswith(OPT_SUFFIXES):
            out[virtual_local_name(real_name)] = module
    return out


def opt_module_name_to_cfg_key(name: str) -> str:
    if name.endswith("mlp.down_proj"):
        return "mlp.down"
    if name.endswith("mlp.up_proj"):
        return "mlp.up"
    if name.endswith("self_attn.k_proj"):
        return "k"
    if name.endswith("self_attn.o_proj"):
        return "o"
    if name.endswith("self_attn.q_proj"):
        return "q"
    if name.endswith("self_attn.v_proj"):
        return "v"
    if name.endswith("mlp.gate_proj"):
        return "mlp.gate"
    raise ValueError(f"Unsupported OPT virtual module name: {name}")


def opt_get_layer_id_from_module_name(name: str) -> int:
    parts = name.split(".")
    for i, item in enumerate(parts):
        if item == "layers" and i + 1 < len(parts):
            return int(parts[i + 1])
    raise ValueError(f"Cannot parse OPT virtual layer id: {name}")


def is_down_type(name: str) -> bool:
    return name.endswith("self_attn.o_proj") or name.endswith("mlp.down_proj")


def is_up_type(name: str) -> bool:
    return not is_down_type(name)


def get_norm_module_for_virtual(model, name: str):
    layer = opt_layers(model)[opt_get_layer_id_from_module_name(name)]
    if ".self_attn." in name:
        return layer.self_attn_layer_norm
    if ".mlp." in name:
        return layer.final_layer_norm
    raise ValueError(f"Cannot infer OPT LayerNorm for {name}")


def stabilize_scale(scale: torch.Tensor, eps: float) -> torch.Tensor:
    scale = scale.to(dtype=torch.float32)
    sign = torch.where(scale < 0, -torch.ones_like(scale), torch.ones_like(scale))
    return sign * scale.abs().clamp_min(float(eps))


def get_had_cached(base_h, cache: Dict[int, Tuple[torch.Tensor, int]], dim: int):
    dim = int(dim)
    if dim not in cache:
        cache[dim] = base_h.hadamard_utils.get_hadK(dim)
    return cache[dim]


def apply_romeo_offline_weight_rotation_opt(model, base_h, rot_flags: Dict[str, bool], eps: float):
    had_cache: Dict[int, Tuple[torch.Tensor, int]] = {}
    targets = opt_collect_target_modules(model)
    for name, module in targets.items():
        if not rot_flags.get(name, False):
            continue
        out_dim, in_dim = map(int, module.weight.data.shape)
        in_had, in_k = get_had_cached(base_h, had_cache, in_dim)
        w_dtype = module.weight.data.dtype
        b_dtype = module.bias.data.dtype if module.bias is not None else None
        w_orig = module.weight.data.to(dtype=torch.float32, device=base_h.utils.DEV)
        w = w_orig
        if is_up_type(name):
            ln = get_norm_module_for_virtual(model, name)
            gamma = stabilize_scale(ln.weight.data, eps).to(dtype=torch.float32, device=base_h.utils.DEV)
            beta = ln.bias.data.to(dtype=torch.float32, device=base_h.utils.DEV) if ln.bias is not None else None
            if beta is not None:
                folded = torch.matmul(w_orig, beta)
                if module.bias is None:
                    module.bias = nn.Parameter(folded.to(dtype=w_dtype, device="cpu"))
                else:
                    module.bias.data = (module.bias.data.to(dtype=torch.float32, device=base_h.utils.DEV) + folded).to(
                        dtype=b_dtype, device="cpu"
                    )
            w = w * gamma.unsqueeze(0)
        w = base_h.apply_hadamard_last_dim(w, in_had, in_k)
        if is_down_type(name):
            out_had, out_k = get_had_cached(base_h, had_cache, out_dim)
            w = base_h.apply_hadamard_last_dim(w.transpose(-1, -2), out_had, out_k).transpose(-1, -2)
        module.weight.data = w.to(dtype=w_dtype, device="cpu")
    return had_cache


def register_romeo_online_rotation_hooks_opt(model, base_h, rot_flags: Dict[str, bool], had_cache, eps: float):
    handles: List[torch.utils.hooks.RemovableHandle] = []
    hidden_size = int(model.config.hidden_size)
    hidden_had, hidden_k = get_had_cached(base_h, had_cache, hidden_size)
    decoder = model.model.decoder

    def embed_hook(_module, _inputs, output):
        return base_h.apply_hadamard_last_dim(output, hidden_had, hidden_k)

    handles.append(decoder.embed_tokens.register_forward_hook(embed_hook))

    if getattr(decoder, "final_layer_norm", None) is not None:
        def final_norm_pre_hook(_module, inputs):
            x = base_h.apply_hadamard_last_dim(inputs[0], hidden_had, hidden_k)
            return (x,) + inputs[1:]
        handles.append(decoder.final_layer_norm.register_forward_pre_hook(final_norm_pre_hook))

    for name, module in opt_collect_target_modules(model).items():
        if not rot_flags.get(name, False):
            continue
        if is_up_type(name):
            ln = get_norm_module_for_virtual(model, name)
            gamma = stabilize_scale(ln.weight.data.detach(), eps)
            beta = ln.bias.data.detach().to(dtype=torch.float32) if ln.bias is not None else None

            def make_ln_hook(local_gamma, local_beta):
                def hook(_module, inputs):
                    x = inputs[0]
                    g = local_gamma.to(device=x.device, dtype=x.dtype)
                    if local_beta is not None:
                        b = local_beta.to(device=x.device, dtype=x.dtype)
                        x = x - b
                    return (x / g,) + inputs[1:]
                return hook

            handles.append(module.register_forward_pre_hook(make_ln_hook(gamma, beta)))
        else:
            in_dim = int(module.weight.data.shape[1])
            had_k, k = get_had_cached(base_h, had_cache, in_dim)

            def make_had_hook(local_had_k, local_k):
                def hook(_module, inputs):
                    x = base_h.apply_hadamard_last_dim(inputs[0], local_had_k, local_k)
                    return (x,) + inputs[1:]
                return hook

            handles.append(module.register_forward_pre_hook(make_had_hook(had_k, k)))
    print(f"[OPT_RoMeoCompat] selected={sum(rot_flags.values())}/{len(rot_flags)} hooks={len(handles)} eps={eps:g}")
    return handles


@torch.no_grad()
def capture_first_inputs_opt(v6, model, loader, device, nsamples, seqlen):
    layers = opt_layers(model)
    model.config.use_cache = False
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
    if getattr(model.model.decoder, "final_layer_norm", None) is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            if name == "module":
                return super().__getattr__(name)
            return getattr(self.module, name)

        def forward(self, inp, **kwargs):
            if cache["i"] >= nsamples:
                raise StopIteration
            inps[cache["i"]].copy_(inp[0] if inp.ndim == 3 else inp)
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in loader:
        if cache["i"] >= nsamples:
            break
        try:
            model(batch[0].to(device))
        except (ValueError, StopIteration):
            pass
    layers[0] = layers[0].module
    if cache["i"] != nsamples:
        raise RuntimeError(f"Captured {cache['i']} samples, expected {nsamples}")

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if getattr(model.model.decoder, "final_layer_norm", None) is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
    torch.cuda.empty_cache()

    kwargs = {}
    if cache["attention_mask"] is not None:
        kwargs["attention_mask"] = cache["attention_mask"].to(device)
    return inps, kwargs


def register_quarot_rotation_hooks_for_layer_opt(base_h, linears, layer_id, rot_flags, had_cache):
    handles = []
    for local_name, module in linears.items():
        full_name = f"model.layers.{layer_id}.{local_name}"
        if not rot_flags.get(full_name, False):
            continue
        in_dim = int(module.weight.shape[1])
        had_k, k = had_cache[in_dim]

        def make_hook(local_had_k, local_k):
            def hook(_module, inputs):
                x = base_h.apply_hadamard_last_dim(inputs[0], local_had_k, local_k)
                return (x,) + inputs[1:]
            return hook

        handles.append(module.register_forward_pre_hook(make_hook(had_k, k)))
    return handles
