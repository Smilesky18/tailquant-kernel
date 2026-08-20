#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lm-eval accuracy wrapper using the RoMeO-style rotation split path.

This mirrors the RoMeO-rotation PPL wrapper by monkey-patching only the
rotation/model-load hooks of the existing accuracy evaluator. It intentionally
does not modify the original evaluator or helper files.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
TOOL_DIR = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR, TOOL_DIR):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import eval_accuracy_lm_eval_v73_weightmode_v2 as BASE_ACC  # noqa: E402
import eval_policy_v6_weightmode_fp16_romeorot_smoke_v1 as ROMEO  # noqa: E402


def main() -> None:
    original_get_model = BASE_ACC.H.model_utils.get_model
    original_offline_rotation = BASE_ACC.H.apply_offline_weight_rotation
    original_online_hooks = BASE_ACC.H.register_online_rotation_hooks

    def get_model_fp16(*args, **kwargs):
        return ROMEO.cast_floating_parameters_to_fp16(
            original_get_model(*args, **kwargs)
        )

    BASE_ACC.H.model_utils.get_model = get_model_fp16
    BASE_ACC.H.apply_offline_weight_rotation = ROMEO.apply_romeo_offline_weight_rotation
    BASE_ACC.H.register_online_rotation_hooks = ROMEO.register_romeo_online_rotation_hooks
    try:
        BASE_ACC.main()
    finally:
        BASE_ACC.H.model_utils.get_model = original_get_model
        BASE_ACC.H.apply_offline_weight_rotation = original_offline_rotation
        BASE_ACC.H.register_online_rotation_hooks = original_online_hooks


if __name__ == "__main__":
    main()
