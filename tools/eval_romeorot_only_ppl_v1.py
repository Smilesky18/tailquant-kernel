#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate RoMeO-style rotation only, without split ratio or weight quant.

This is intentionally separate from the policy evaluator because the policy
path always registers activation split hooks. Here we only apply the
RoMeO-style offline/online rotations and then run full-model PPL.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts"):
    text = str(item)
    if text not in sys.path:
        sys.path.insert(0, text)

import eval_policy_v6_weightmode_v1 as BASE  # noqa: E402
import eval_policy_v6_weightmode_fp16_romeorot_smoke_v1 as RR  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext103")
    parser.add_argument("--window_start", type=int, default=0)
    parser.add_argument("--n_windows", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rotation_config", required=True)
    parser.add_argument("--hf_token", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    args.eval_dataset = args.dataset
    args.eval_n_samples = args.window_start + args.n_windows
    args.bsz = 1

    started = time.time()
    model = BASE.H.model_utils.get_model(args.model, args.hf_token)
    RR.cast_floating_parameters_to_fp16(model)
    max_len = int(getattr(model.config, "max_position_embeddings", args.seqlen))
    model.seqlen = min(args.seqlen, max_len)

    rot_flags = BASE.H.build_rotation_flags(model, args.rotation_config)
    had_cache = RR.apply_romeo_offline_weight_rotation(model, rot_flags)

    device = torch.device("cuda:0")
    model.to(device).eval()
    handles = RR.register_romeo_online_rotation_hooks(model, rot_flags, had_cache)

    encoded = BASE.H.get_eval_dataset_encoding(args, model_seqlen=model.seqlen)
    window_ce, token_counts, mean_ce = BASE.evaluate(
        model, encoded, device, model.seqlen, args.window_start, args.n_windows
    )
    result = {
        "mode": "romeorot_only_no_ratio_no_weight_quant",
        "model": args.model,
        "dataset": args.dataset,
        "window_start": args.window_start,
        "n_windows": args.n_windows,
        "seqlen": model.seqlen,
        "rotation_config": str(Path(args.rotation_config).resolve()),
        "selected_rotation_modules": int(sum(bool(v) for v in rot_flags.values())),
        "window_ce": window_ce,
        "token_counts": token_counts,
        "mean_ce": mean_ce,
        "ppl": float(math.exp(mean_ce)),
        "elapsed_seconds": time.time() - started,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ppl": result["ppl"], "mean_ce": result["mean_ce"]}, indent=2))

    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
