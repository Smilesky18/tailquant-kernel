#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPT grouped-capped low-ratio search with RoMeO-style rotation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts"):
    text = str(item)
    if text not in sys.path:
        sys.path.insert(0, text)

import calibrate_per_linear_v6_lowratio_grouped_capped_v3 as V3  # noqa: E402
import opt_romeorot_compat_v1 as OPTC  # noqa: E402


V6 = V3.V2.V1.V6
H = V6.H
NORM_SCALE_EPS = float(os.environ.get("ROMEO_ROT_NORM_SCALE_EPS", "1e-3"))


def cast_fp16_and_alias_layers(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point() and p.dtype != torch.float16:
                p.data = p.data.to(dtype=torch.float16)
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        model.model.layers = model.model.decoder.layers
    return model


_GET_MODEL = H.model_utils.get_model
_COLLECT = H.collect_target_modules
_CFG_KEY = H.module_name_to_cfg_key
_LAYER_ID = H.get_layer_id_from_module_name
_OFFLINE = H.apply_offline_weight_rotation
_CAPTURE = V6.capture_first_inputs
_LOCAL_LINEARS = V6.local_linears
_REGISTER = V6.register_rotation_hooks_for_layer


def get_model(*args, **kwargs):
    return cast_fp16_and_alias_layers(_GET_MODEL(*args, **kwargs))


def offline(model, rot_flags):
    return OPTC.apply_romeo_offline_weight_rotation_opt(model, H, rot_flags, NORM_SCALE_EPS)


def capture(model, loader, device, nsamples, seqlen):
    had_cache = getattr(model, "_opt_romeorot_had_cache", {})
    hidden_had, hidden_k = OPTC.get_had_cached(H, had_cache, int(model.config.hidden_size))
    model._opt_romeorot_had_cache = had_cache

    def embed_hook(_module, _inputs, output):
        return H.apply_hadamard_last_dim(output, hidden_had, hidden_k)

    handle = model.model.decoder.embed_tokens.register_forward_hook(embed_hook)
    try:
        return OPTC.capture_first_inputs_opt(V6, model, loader, device, nsamples, seqlen)
    finally:
        handle.remove()


def register_layer(linears, layer_id, rot_flags, had_cache):
    handles = []
    for local_name, module in linears.items():
        full_name = f"model.layers.{layer_id}.{local_name}"
        if not rot_flags.get(full_name, False):
            continue
        if OPTC.is_up_type(full_name):
            norm_module = OPTC.get_norm_module_for_virtual(_ACTIVE_MODEL, full_name)
            gamma = OPTC.stabilize_scale(norm_module.weight.data.detach(), NORM_SCALE_EPS)
            beta = norm_module.bias.data.detach().to(dtype=torch.float32) if norm_module.bias is not None else None

            def make_hook(local_gamma, local_beta):
                def hook(_module, inputs):
                    x = inputs[0]
                    g = local_gamma.to(device=x.device, dtype=x.dtype)
                    if local_beta is not None:
                        x = x - local_beta.to(device=x.device, dtype=x.dtype)
                    return (x / g,) + inputs[1:]
                return hook

            handles.append(module.register_forward_pre_hook(make_hook(gamma, beta)))
        else:
            in_dim = int(module.weight.shape[1])
            had_k, k = OPTC.get_had_cached(H, had_cache, in_dim)

            def make_had(local_had_k, local_k):
                def hook(_module, inputs):
                    return (H.apply_hadamard_last_dim(inputs[0], local_had_k, local_k),) + inputs[1:]
                return hook

            handles.append(module.register_forward_pre_hook(make_had(had_k, k)))
    return handles


_ACTIVE_MODEL = None


def main():
    global _ACTIVE_MODEL
    H.model_utils.get_model = get_model
    H.collect_target_modules = OPTC.opt_collect_target_modules
    H.module_name_to_cfg_key = OPTC.opt_module_name_to_cfg_key
    H.get_layer_id_from_module_name = OPTC.opt_get_layer_id_from_module_name
    H.apply_offline_weight_rotation = offline
    V6.capture_first_inputs = capture
    V6.local_linears = OPTC.opt_local_linears
    V6.register_rotation_hooks_for_layer = register_layer
    old_get_model = H.model_utils.get_model

    def tracked_get_model(*args, **kwargs):
        global _ACTIVE_MODEL
        _ACTIVE_MODEL = old_get_model(*args, **kwargs)
        return _ACTIVE_MODEL

    H.model_utils.get_model = tracked_get_model
    try:
        V3.main()
    finally:
        H.model_utils.get_model = _GET_MODEL
        H.collect_target_modules = _COLLECT
        H.module_name_to_cfg_key = _CFG_KEY
        H.get_layer_id_from_module_name = _LAYER_ID
        H.apply_offline_weight_rotation = _OFFLINE
        V6.capture_first_inputs = _CAPTURE
        V6.local_linears = _LOCAL_LINEARS
        V6.register_rotation_hooks_for_layer = _REGISTER


if __name__ == "__main__":
    main()
