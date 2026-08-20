#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RTN-weight variant of the real Split full-stack PPL benchmark.

This wrapper keeps bench_real_split_fullstack_v1.py unchanged, but replaces its
GPTQ preparation step with policy-aware RTN weight quantization. It is intended
for fake-vs-real-kernel PPL consistency smoke tests.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
SCRIPT_DIR = ROOT / "kernel_quant/scripts"
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", SCRIPT_DIR, TOOLS):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import bench_real_split_fullstack_v1 as BASE  # noqa: E402
import quant_utils  # noqa: E402
from policy_runtime_v6 import make_weight_scale_stat_fn, normalize_name  # noqa: E402


def build_rtn_args(cli, policy: dict) -> SimpleNamespace:
    args = SimpleNamespace()
    args.model = cli.model
    args.hf_token = cli.hf_token
    args.cal_dataset = cli.cal_dataset
    args.nsamples = cli.nsamples
    args.seed = cli.seed
    args.percdamp = cli.percdamp
    args.gptq_percdamp = cli.percdamp
    args.damp_percent = cli.percdamp
    args.act_order = False
    args.w_bits = 4
    args.w_groupsize = -1
    args.w_asym = False
    args.w_clip = False
    args.int8_down_proj = False
    args.rotate = True
    args.rotation = True
    args.no_rotate = False
    args.apply_rotation = True
    args.quant_method = "rtn"
    args.w_rtn = True
    args.scale_eps = cli.eps
    args.enable_scale_experiment = True
    args.scale_apply_to = "weight"
    args.weight_scale_enabled_modules = None
    args.weight_scale_stat_fn = make_weight_scale_stat_fn(policy)
    return args


def rtn_quantize_with_scale(w: torch.Tensor, *, bits: int, groupsize: int, args) -> tuple[torch.Tensor, torch.Tensor]:
    if bits >= 16:
        return w, torch.ones(w.shape[0], device=w.device, dtype=torch.float32)
    if bool(args.w_asym):
        raise ValueError("RTN smoke only supports symmetric W4, matching fake eval.")

    w_dtype = w.dtype
    w = w.float()
    out_f, in_f = w.shape
    if groupsize is None or groupsize <= 0:
        groupsize = in_f
    if in_f % groupsize != 0:
        raise ValueError(f"RTN groupsize must divide in_features: {in_f} % {groupsize}")

    wg = w.view(out_f, in_f // groupsize, groupsize)
    abs_vals = wg.abs()
    qmax = 2 ** (int(bits) - 1) - 1
    scale_stat = args.weight_scale_stat_fn(abs_vals, args)
    scale = (scale_stat / max(qmax, 1)).clamp(min=float(args.scale_eps))
    q = torch.clamp(torch.round(wg / scale), -qmax - 1, qmax)
    wq = (q * scale).view(out_f, in_f).to(w_dtype)
    exact_scale = scale.reshape(out_f, -1)
    if exact_scale.shape[1] != 1:
        raise ValueError("RealPolicyLinear currently expects per-output-channel scale for this smoke.")
    return wq, exact_scale[:, 0].float()


def apply_offline_weight_rotation_on_device(model, rot_flags: dict, device: torch.device) -> dict:
    targets = BASE.H.collect_target_modules(model)
    had_cache = {}
    for name, module in targets.items():
        if not rot_flags.get(name, False):
            continue
        in_dim = int(module.weight.data.shape[1])
        if in_dim not in had_cache:
            had_k, k = BASE.H.hadamard_utils.get_hadK(in_dim)
            if had_k is not None:
                had_k = had_k.to(device=device)
            had_cache[in_dim] = (had_k, k)
        had_k, k = had_cache[in_dim]

        w_dtype = module.weight.data.dtype
        w = module.weight.data.to(dtype=torch.float32, device=device)
        w = BASE.H.apply_hadamard_last_dim(w, had_k, k)
        module.weight.data = w.to(dtype=w_dtype, device="cpu")
    return had_cache


def install_nozero_split_prepare():
    original = BASE.RealPolicyLinear._prepare_split

    def _prepare_split_nozero(self, A: torch.Tensor, scratch: dict):
        M, K = int(A.shape[0]), int(A.shape[1])
        R = self.R
        if R <= 0:
            raise RuntimeError(f"{self.name}: RTN no-zero smoke requires R>0")

        percentile = min(max(float(self.activation_percentile), 0.0), 100.0)
        body_len = K - R
        body_kth = min(K, max(1, int(math.ceil(body_len * percentile / 100.0))))
        descending_rank = K - body_kth + 1
        select_k = max(R, descending_rank)

        abs_A = A.abs().float()
        top_values, top_indices = torch.topk(
            abs_A,
            k=select_k,
            dim=1,
            largest=True,
            sorted=True,
        )
        body_threshold = top_values[:, descending_rank - 1].contiguous()
        tail_threshold = top_values[:, 0].contiguous()

        tail_indices = top_indices[:, :R]
        tail_indices, _ = torch.sort(tail_indices, dim=1)
        tail_indices = tail_indices.to(torch.int32).contiguous()

        quant_buffers = self.scratch_pool.get_quant_buffers(M, self.K, self.N, R)
        tail_q = quant_buffers["top_q"]
        body_q_top = quant_buffers["body_q_top"]

        self.policy_pack_ext.pack_policy_split(
            A,
            tail_indices,
            body_threshold,
            tail_threshold,
            scratch["A_pack"],
            scratch["body_scale"],
            scratch["top_scale"],
            tail_q,
            self.eps,
        )
        # v61/no-zero semantics: sparse epilogue receives body_q_top so it can
        # replace body quantization at top indices instead of adding tail twice.
        return tail_indices, tail_q, body_q_top

    BASE.RealPolicyLinear._prepare_split = _prepare_split_nozero
    return original


def prepare_rotated_rtn_model(cli, policy: dict):
    started = time.time()
    args = build_rtn_args(cli, policy)
    device = torch.device("cuda:0")
    BASE.H.utils.DEV = device

    model = BASE.H.model_utils.get_model(cli.model, cli.hf_token)
    max_len = int(getattr(model.config, "max_position_embeddings", cli.gptq_seqlen))
    model.seqlen = min(cli.gptq_seqlen, max_len)

    rot_flags = BASE.H.build_rotation_flags(model, cli.rotation_config)
    had_cache = apply_offline_weight_rotation_on_device(model, rot_flags, device)

    quantizers = {}
    layers = model.model.layers
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(device)
        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])
        for local_name, linear in subset.items():
            if "lm_head" in local_name:
                continue
            full_name = normalize_name(f"model.layers.{layer_idx}.{local_name}")
            if full_name not in policy["modules"]:
                continue
            bits = int(args.w_bits)
            if args.int8_down_proj and "down_proj" in local_name:
                bits = 8
            args.weight_scale_current_module = full_name
            wq, scale = rtn_quantize_with_scale(
                linear.weight.data,
                bits=bits,
                groupsize=int(args.w_groupsize),
                args=args,
            )
            linear.weight.data = wq
            quantizers[full_name] = SimpleNamespace(scale=scale.detach().cpu())
        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    return {
        "model": model,
        "quantizers": quantizers,
        "rot_flags": {normalize_name(name): bool(value) for name, value in rot_flags.items()},
        "had_cache": had_cache,
        "max_len": max_len,
        "prepare_seconds": time.time() - started,
    }


def _out_dir_from_argv() -> Path | None:
    for idx, value in enumerate(sys.argv):
        if value == "--out_dir" and idx + 1 < len(sys.argv):
            return Path(sys.argv[idx + 1])
        if value.startswith("--out_dir="):
            return Path(value.split("=", 1)[1])
    return None


def main():
    original_prepare = BASE.prepare_rotated_gptq_model
    original_prepare_split = install_nozero_split_prepare()
    BASE.prepare_rotated_gptq_model = prepare_rotated_rtn_model
    try:
        BASE.main()
    finally:
        BASE.prepare_rotated_gptq_model = original_prepare
        BASE.RealPolicyLinear._prepare_split = original_prepare_split

    out_dir = _out_dir_from_argv()
    if out_dir is not None:
        result_path = out_dir / "result.json"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            data["version"] = "split_real_fullstack_rtn_ppl_v1"
            data.setdefault("quantization", {})["weight"] = "RTN W4 with policy per-Linear weight percentile"
            result_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
