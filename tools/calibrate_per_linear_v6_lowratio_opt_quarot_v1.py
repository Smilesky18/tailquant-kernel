#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OPT-1.3B oldv6/QuaRot low-ratio base search wrapper."""
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

import calibrate_per_linear_v6_lowratio_v1 as V1  # noqa: E402
import opt_romeorot_compat_v1 as OPTC  # noqa: E402


V6 = V1.V6
H = V6.H


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
_CAPTURE = V6.capture_first_inputs
_LOCAL_LINEARS = V6.local_linears
_REGISTER = V6.register_rotation_hooks_for_layer


def get_model(*args, **kwargs):
    return cast_fp16_and_alias_layers(_GET_MODEL(*args, **kwargs))


def main():
    H.model_utils.get_model = get_model
    H.collect_target_modules = OPTC.opt_collect_target_modules
    H.module_name_to_cfg_key = OPTC.opt_module_name_to_cfg_key
    H.get_layer_id_from_module_name = OPTC.opt_get_layer_id_from_module_name
    V6.capture_first_inputs = lambda model, loader, device, nsamples, seqlen: OPTC.capture_first_inputs_opt(
        V6, model, loader, device, nsamples, seqlen
    )
    V6.local_linears = OPTC.opt_local_linears
    V6.register_rotation_hooks_for_layer = lambda linears, layer_id, rot_flags, had_cache: (
        OPTC.register_quarot_rotation_hooks_for_layer_opt(H, linears, layer_id, rot_flags, had_cache)
    )
    try:
        V1.main()
    finally:
        H.model_utils.get_model = _GET_MODEL
        H.collect_target_modules = _COLLECT
        H.module_name_to_cfg_key = _CFG_KEY
        H.get_layer_id_from_module_name = _LAYER_ID
        V6.capture_first_inputs = _CAPTURE
        V6.local_linears = _LOCAL_LINEARS
        V6.register_rotation_hooks_for_layer = _REGISTER


if __name__ == "__main__":
    main()
