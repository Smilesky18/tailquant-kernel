#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FP16 fake-quant smoke wrapper for eval_policy_v6_weightmode_v1.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import eval_policy_v6_weightmode_v1 as BASE  # noqa: E402


def cast_floating_parameters_to_fp16(model):
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float16:
                parameter.data = parameter.data.to(dtype=torch.float16)
    return model


def main():
    original_get_model = BASE.H.model_utils.get_model

    def get_model_fp16(*args, **kwargs):
        return cast_floating_parameters_to_fp16(original_get_model(*args, **kwargs))

    BASE.H.model_utils.get_model = get_model_fp16
    try:
        BASE.main()
    finally:
        BASE.H.model_utils.get_model = original_get_model


if __name__ == "__main__":
    main()
