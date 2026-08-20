#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Low-ratio variant of the old v6 per-Linear calibration flow.

This file intentionally lives in the experiment directory and does not modify
the original repository scripts. It reuses the old v6 data/rotation/capture
flow, but changes per-linear model selection:

1. Keep all evaluated optimization states instead of only minimum val_total.
2. Find the best validation reconstruction error.
3. Among states within a configurable reconstruction tolerance, select the
   lowest projected ratio, then the lowest reconstruction.
4. Use a finer ratio projection grid by default.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant" / "scripts"):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def load_v6_module():
    path = ROOT / "kernel_quant/scripts/calibrate_per_linear_v6.py"
    spec = importlib.util.spec_from_file_location("_old_calibrate_per_linear_v6", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import old v6 calibrator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_old_calibrate_per_linear_v6"] = module
    spec.loader.exec_module(module)
    return module


V6 = load_v6_module()


def parse_ratio_grid(text: str) -> tuple[float, ...]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("ratio grid is empty")
    return tuple(sorted(set(values)))


def project_ratio_grid(ratio: float, grid: tuple[float, ...]) -> float:
    return min(grid, key=lambda x: (abs(float(x) - float(ratio)), float(x)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--rotation_config", default="/data/yzy/quarot/qwen3-8B_layer_all.csv")
    p.add_argument("--nsamples", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--capture_rows", type=int, default=1024)
    p.add_argument("--train_rows", type=int, default=128)
    p.add_argument("--val_rows", type=int, default=256)
    p.add_argument("--train_out_channels", type=int, default=256)
    p.add_argument("--val_out_channels", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--ratio_lambda", type=float, default=0.01)
    p.add_argument("--init_ratio", type=float, default=0.04)
    p.add_argument("--max_ratio", type=float, default=0.16)
    p.add_argument("--init_activation_percentile", type=float, default=99.75)
    p.add_argument("--init_weight_percentile", type=float, default=99.9)
    p.add_argument("--min_percentile", type=float, default=0.98)
    p.add_argument("--mask_temp_start", type=float, default=0.02)
    p.add_argument("--mask_temp_end", type=float, default=0.004)
    p.add_argument("--quantile_temp_start", type=float, default=0.003)
    p.add_argument("--quantile_temp_end", type=float, default=0.0008)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--max_layers", type=int, default=-1)
    p.add_argument("--hf_token", default=None)
    p.add_argument(
        "--ratio_grid",
        default="0,0.0025,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.16",
    )
    p.add_argument("--recon_tolerance_rel", type=float, default=0.02)
    p.add_argument("--recon_tolerance_abs", type=float, default=1e-5)
    p.add_argument(
        "--selection_metric",
        choices=["lowest_ratio_within_recon_tolerance", "old_val_total"],
        default="lowest_ratio_within_recon_tolerance",
    )
    return p.parse_args()


def optimize_one_linear(
    *,
    module_name: str,
    x_cpu: torch.Tensor,
    module: torch.nn.Linear,
    args,
    cost_weight: float,
    seed: int,
):
    device = module.weight.device
    x_all = x_cpu.float()
    if len(x_all) < 4:
        raise RuntimeError(f"{module_name}: not enough captured rows: {len(x_all)}")

    ratio_grid = parse_ratio_grid(args.ratio_grid)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    perm = torch.randperm(len(x_all), generator=gen)
    val_count = min(args.val_rows, max(1, len(x_all) // 4))
    val_idx = perm[:val_count]
    train_idx = perm[val_count:]
    if len(train_idx) == 0:
        train_idx = perm

    weight = module.weight.detach()
    bias = module.bias.detach() if module.bias is not None else None
    weight_top = V6.prepare_weight_top_values(weight, args.min_percentile).to(device)

    out_features = int(weight.shape[0])
    val_out_count = min(args.val_out_channels, out_features)
    val_out_idx = torch.linspace(0, out_features - 1, steps=val_out_count, device=device).round().long().unique()

    params = V6.PerLinearQuantParameters(
        init_ratio=args.init_ratio,
        max_ratio=args.max_ratio,
        init_activation_percentile=args.init_activation_percentile / 100.0,
        init_weight_percentile=args.init_weight_percentile / 100.0,
        min_percentile=args.min_percentile,
    ).to(device)
    optimizer = torch.optim.Adam(params.parameters(), lr=args.lr, betas=(0.9, 0.99))

    fixed_val_x = x_all.index_select(0, val_idx).to(device)
    best_total = None
    candidates = []
    history = []
    started = time.time()

    for step in range(args.steps):
        frac = step / max(args.steps - 1, 1)
        mask_temp = args.mask_temp_start * (args.mask_temp_end / args.mask_temp_start) ** frac
        quant_temp = args.quantile_temp_start * (args.quantile_temp_end / args.quantile_temp_start) ** frac

        take = min(args.train_rows, len(train_idx))
        choose = train_idx[torch.randint(0, len(train_idx), (take,), generator=gen)]
        x_batch = x_all.index_select(0, choose).to(device)

        out_take = min(args.train_out_channels, out_features)
        out_idx = torch.randint(0, out_features, (out_take,), generator=gen).to(device).unique()

        optimizer.zero_grad(set_to_none=True)
        total, recon, cost = V6.normalized_linear_reconstruction_loss(
            x_batch,
            weight,
            bias,
            out_idx,
            weight_top,
            params,
            max_ratio=args.max_ratio,
            min_percentile=args.min_percentile,
            mask_temperature=mask_temp,
            quantile_temperature=quant_temp,
            ratio_lambda=args.ratio_lambda,
            cost_weight=cost_weight,
            eps=args.eps,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                val_total, val_recon, val_cost = V6.normalized_linear_reconstruction_loss(
                    fixed_val_x,
                    weight,
                    bias,
                    val_out_idx,
                    weight_top,
                    params,
                    max_ratio=args.max_ratio,
                    min_percentile=args.min_percentile,
                    mask_temperature=args.mask_temp_end,
                    quantile_temperature=args.quantile_temp_end,
                    ratio_lambda=args.ratio_lambda,
                    cost_weight=cost_weight,
                    eps=args.eps,
                )
                values = params.exported()
                projected = project_ratio_grid(values.ratio, ratio_grid)
                record = {
                    "step": step,
                    "train_total": float(total.detach().cpu()),
                    "train_reconstruction": float(recon.cpu()),
                    "train_cost": float(cost.cpu()),
                    "val_total": float(val_total.cpu()),
                    "val_reconstruction": float(val_recon.cpu()),
                    "val_cost": float(val_cost.cpu()),
                    "ratio": values.ratio,
                    "ratio_projected_custom": projected,
                    "activation_percentile": values.activation_percentile * 100.0,
                    "weight_percentile": values.weight_percentile * 100.0,
                }
                state = {k: v.detach().cpu().clone() for k, v in params.state_dict().items()}
                item = {"record": record, "state": state}
                candidates.append(item)
                history.append(record)
                if best_total is None or record["val_total"] < best_total["record"]["val_total"]:
                    best_total = item

    if args.selection_metric == "old_val_total":
        selected = best_total
        selection_reason = "old minimum val_total"
    else:
        best_recon = min(item["record"]["val_reconstruction"] for item in candidates)
        threshold = best_recon * (1.0 + float(args.recon_tolerance_rel)) + float(args.recon_tolerance_abs)
        feasible = [
            item
            for item in candidates
            if item["record"]["val_reconstruction"] <= threshold
        ]
        selected = min(
            feasible,
            key=lambda item: (
                item["record"]["ratio_projected_custom"],
                item["record"]["ratio"],
                item["record"]["val_reconstruction"],
                item["record"]["val_total"],
            ),
        )
        selection_reason = (
            "lowest projected ratio within validation reconstruction tolerance "
            f"(best_recon={best_recon:.8g}, threshold={threshold:.8g})"
        )

    params.load_state_dict(selected["state"])
    learned = params.exported()
    projected = project_ratio_grid(learned.ratio, ratio_grid)
    result = {
        "module_name": module_name,
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "mac_weight": int(module.in_features * module.out_features),
        "cost_weight_relative_to_mean_linear": float(cost_weight),
        "ratio_lambda": float(args.ratio_lambda),
        "ratio_continuous": learned.ratio,
        "ratio_projected": projected,
        "activation_percentile": learned.activation_percentile * 100.0,
        "weight_percentile": learned.weight_percentile * 100.0,
        "best_val_total": selected["record"]["val_total"],
        "best_val_reconstruction": selected["record"]["val_reconstruction"],
        "best_val_cost": selected["record"]["val_cost"],
        "old_best_val_total": best_total["record"]["val_total"],
        "old_best_val_reconstruction": best_total["record"]["val_reconstruction"],
        "old_best_ratio_projected": best_total["record"]["ratio_projected_custom"],
        "selection_reason": selection_reason,
        "selection_metric": args.selection_metric,
        "ratio_grid": list(ratio_grid),
        "recon_tolerance_rel": float(args.recon_tolerance_rel),
        "recon_tolerance_abs": float(args.recon_tolerance_abs),
        "captured_rows": int(len(x_all)),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
    del params, optimizer, weight_top
    torch.cuda.empty_cache()
    return result


def main():
    V6.parse_args = parse_args
    V6.optimize_one_linear = optimize_one_linear
    V6.main()


if __name__ == "__main__":
    main()
