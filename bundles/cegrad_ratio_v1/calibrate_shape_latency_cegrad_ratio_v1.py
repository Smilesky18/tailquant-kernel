#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shape-latency-aware CE-gradient ratio search.

This script is intentionally self-contained and does NOT modify the original
calibrate_per_linear_v74.py or diff_quant_v7.py.  It reuses the same model/data
loading and rotation path as v7.4, but replaces the MSE reconstruction objective
with a candidate-based first-order CE proxy:

    dCE ~= <Y_quant - Y_fp, dCE/dY_fp>

For each Linear, candidates are selected by

    score = (CE_proxy(candidate) - CE_proxy(anchor))
            + latency_lambda * (Latency(shape, ratio) - Latency(shape, 0))

The anchor is single-scale A4 with ratio=0 and percentile=99.5 by default.
The output policy keeps the v7/v7.4-style fields plus extra aliases so downstream
scripts that expect ratio / projected_ratio / R-style fields are less likely to
misinterpret zero-ratio and split-ratio entries.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(
    os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")
).resolve()
for item in (
    ROOT,
    ROOT / "fake_quant",
    ROOT / "kernel_quant",
    ROOT / "kernel_quant" / "scripts",
    ROOT / "experiments" / "kernel_quant" / "layer_latency_split_v1" / "tools",
):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))


def load_helper():
    path = ROOT / "kernel_quant/scripts/eval_split_oldflow_kernel.py"
    spec = importlib.util.spec_from_file_location("_cegrad_helper", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cegrad_helper"] = module
    spec.loader.exec_module(module)
    return module


H = load_helper()
from fake_quant import data_utils  # noqa: E402

try:  # local_mixed_loader_v74 is optional in older copies of the tree.
    from local_mixed_loader_v74 import build_local_mixed_trainloader  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - server-side optional dependency
    build_local_mixed_trainloader = None

try:
    from diff_quant_v7 import project_ratio as _project_ratio  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - robust fallback
    _project_ratio = None


TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


# Keep this fallback explicit.  If diff_quant_v7.project_ratio exists, it is used.
# Otherwise candidates are already discrete, so identity projection is acceptable.
def project_ratio_safe(value: float, mode: str) -> float:
    if mode != "split_v6":
        return 0.0
    if _project_ratio is not None:
        try:
            return float(_project_ratio(float(value)))
        except Exception:
            pass
    return float(max(0.0, value))


def parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for piece in str(text).replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(float(piece))
    if not values:
        raise ValueError(f"Empty float list: {text!r}")
    return sorted(set(values))


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def slice_kwargs(kwargs: Dict[str, Any], sequence_length: int) -> Dict[str, Any]:
    output = dict(kwargs)
    mask = output.get("attention_mask")
    if mask is not None:
        if mask.dim() == 2:
            output["attention_mask"] = mask[:, -sequence_length:]
        elif mask.dim() == 4:
            output["attention_mask"] = mask[
                :, :, -sequence_length:, -sequence_length:
            ]
    position_ids = output.get("position_ids")
    if position_ids is not None and position_ids.dim() == 2:
        output["position_ids"] = position_ids[:, -sequence_length:]
    embeddings = output.get("position_embeddings")
    if embeddings is not None:
        sliced = []
        for value in embeddings:
            if value is None:
                sliced.append(None)
            elif value.dim() >= 2 and value.shape[-2] >= sequence_length:
                sliced.append(value[..., -sequence_length:, :])
            else:
                sliced.append(value)
        output["position_embeddings"] = tuple(sliced)
    return output


@torch.no_grad()
def capture_first_layer_inputs_and_tokens(
    model: nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    nsamples: int,
    seqlen: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Capture first decoder-layer inputs and token ids, v7.4-style."""
    layers = model.model.layers
    model.config.use_cache = False
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inputs = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    token_ids = torch.zeros((nsamples, seqlen), dtype=torch.long, device="cpu")
    cache: Dict[str, Any] = {
        "index": 0,
        "attention_mask": None,
        "position_ids": None,
        "position_embeddings": None,
        "current_tokens": None,
    }

    class Catcher(nn.Module):
        def __init__(self, wrapped: nn.Module):
            super().__init__()
            self.wrapped = wrapped

        def __getattr__(self, name: str) -> Any:
            if name == "wrapped":
                return super().__getattr__(name)
            return getattr(self.wrapped, name)

        def forward(self, value: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            if cache["index"] >= nsamples:
                raise StopIteration
            source = value[0] if value.ndim == 3 else value
            current_tokens = cache.get("current_tokens")
            if current_tokens is None:
                raise RuntimeError("Internal error: missing current token batch")
            tokens = current_tokens[0] if current_tokens.ndim == 2 else current_tokens
            if source.shape[0] < seqlen or tokens.shape[0] < seqlen:
                raise RuntimeError(
                    f"Sequence too short: hidden={tuple(source.shape)}, "
                    f"tokens={tuple(tokens.shape)}, required seqlen={seqlen}"
                )
            inputs[cache["index"]].copy_(source[-seqlen:])
            token_ids[cache["index"]].copy_(tokens[-seqlen:].detach().cpu())
            cache["index"] += 1
            for key in ("attention_mask", "position_ids", "position_embeddings"):
                cache[key] = kwargs.get(key)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in loader:
        if cache["index"] >= nsamples:
            break
        batch_tokens = batch[0]
        cache["current_tokens"] = batch_tokens
        try:
            model(batch_tokens.to(device))
        except (ValueError, StopIteration):
            pass

    layers[0] = layers[0].wrapped
    if cache["index"] != nsamples:
        raise RuntimeError(f"Captured {cache['index']} samples, expected {nsamples}")

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    kwargs = {
        "attention_mask": cache["attention_mask"],
        "position_ids": cache["position_ids"],
    }
    if cache["position_embeddings"] is not None:
        kwargs["position_embeddings"] = cache["position_embeddings"]

    kwargs = {
        key: (
            tuple(item.to(device) for item in value)
            if isinstance(value, tuple)
            else value.to(device)
            if torch.is_tensor(value)
            else value
        )
        for key, value in kwargs.items()
        if value is not None
    }
    return inputs, token_ids, kwargs


def local_linears(layer: nn.Module) -> Dict[str, nn.Linear]:
    return {
        name: module
        for name, module in layer.named_modules()
        if isinstance(module, nn.Linear) and name.endswith(TARGET_SUFFIXES)
    }


def register_rotation_hooks(
    linears: Dict[str, nn.Linear],
    layer_id: int,
    rotation_flags: Dict[str, bool],
    hadamard_cache: Dict[int, Tuple[torch.Tensor, Any]],
) -> List[Any]:
    handles = []
    for local_name, module in linears.items():
        full_name = f"model.layers.{layer_id}.{local_name}"
        if not rotation_flags.get(full_name, False):
            continue
        input_dimension = int(module.weight.shape[1])
        hadamard, k = hadamard_cache[input_dimension]

        def make_hook(local_hadamard: torch.Tensor, local_k: Any):
            def hook(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]):
                rotated = H.apply_hadamard_last_dim(inputs[0], local_hadamard, local_k)
                return (rotated,) + inputs[1:]
            return hook

        handles.append(module.register_forward_pre_hook(make_hook(hadamard, k)))
    return handles


def set_parameter_requires_grad(model: nn.Module, requires_grad: bool = False) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad)


def layer_forward(layer: nn.Module, hidden: torch.Tensor, kwargs: Dict[str, Any]) -> torch.Tensor:
    result = layer(hidden, **kwargs)
    if isinstance(result, tuple):
        return result[0]
    return result


def forward_suffix_from_layer(
    model: nn.Module,
    start_layer_id: int,
    hidden: torch.Tensor,
    kwargs: Dict[str, Any],
) -> torch.Tensor:
    layers = model.model.layers
    for next_layer_id in range(start_layer_id + 1, len(layers)):
        next_layer = layers[next_layer_id].to(hidden.device).eval()
        hidden = layer_forward(next_layer, hidden, kwargs)
    hidden = model.model.norm.to(hidden.device)(hidden)
    logits = model.lm_head.to(hidden.device)(hidden)
    return logits


def compute_lm_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = tokens[:, 1:].contiguous().to(shift_logits.device)
    return F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="mean",
    )


@dataclass(frozen=True)
class Candidate:
    ratio: float
    activation_percentile: float
    weight_percentile: float
    tail_percentile: float


@dataclass
class CandidateMetrics:
    candidate: Candidate
    ratio_projected: float
    R: int
    ce_signed: float
    ce_pos: float
    ce_abs: float
    ce_metric: float
    delta_ce: float
    ce_gain: float
    latency_cost: float
    latency_delta: float
    score: float


class ShapeLatencyTable:
    """Lookup measured shape cost, with linear interpolation over ratio."""

    def __init__(
        self,
        path: Optional[Path],
        batch_column: str,
        missing: str,
        latency_unit_scale: float,
    ) -> None:
        self.path = path
        self.batch_column = batch_column
        self.missing = missing
        self.latency_unit_scale = float(latency_unit_scale)
        self.data: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
        if path is not None:
            self._load(path)

    @staticmethod
    def _first_existing(row: Dict[str, str], names: Sequence[str]) -> Optional[str]:
        lowered = {k.lower(): k for k in row.keys()}
        for name in names:
            if name in row:
                return row[name]
            key = lowered.get(name.lower())
            if key is not None:
                return row[key]
        return None

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError(f"Empty latency table: {path}")
            for row in reader:
                k_text = self._first_existing(row, ("K", "k", "in_features", "in", "input_dim"))
                n_text = self._first_existing(row, ("N", "n", "out_features", "out", "output_dim"))
                ratio_text = self._first_existing(row, ("ratio", "r", "split_ratio", "ratio_projected"))
                cost_text = self._first_existing(
                    row,
                    (
                        self.batch_column,
                        f"latency_{self.batch_column}",
                        f"{self.batch_column}_ms",
                        "latency_ms",
                        "cost",
                        "total_ms",
                        "split_ms",
                    ),
                )
                if k_text is None or n_text is None or ratio_text is None or cost_text is None:
                    continue
                try:
                    key = (int(float(k_text)), int(float(n_text)))
                    ratio = float(ratio_text)
                    cost = float(cost_text) * self.latency_unit_scale
                except ValueError:
                    continue
                self.data.setdefault(key, []).append((ratio, cost))
        for key in list(self.data.keys()):
            merged: Dict[float, List[float]] = {}
            for ratio, cost in self.data[key]:
                merged.setdefault(ratio, []).append(cost)
            self.data[key] = sorted((r, float(np.mean(v))) for r, v in merged.items())
        if not self.data:
            raise RuntimeError(
                f"No usable rows loaded from latency table {path}. "
                f"Expected columns like K,N,ratio,{self.batch_column}."
            )

    def _proxy_cost(self, K: int, N: int, ratio: float) -> float:
        return float(K * N) * max(float(ratio), 0.0)

    def cost(self, K: int, N: int, ratio: float) -> float:
        key = (int(K), int(N))
        ratio = max(float(ratio), 0.0)
        if key not in self.data:
            if self.path is not None and self.missing == "error":
                raise KeyError(
                    f"Missing shape K={K}, N={N} in latency table {self.path}"
                )
            return self._proxy_cost(K, N, ratio)
        points = self.data[key]
        if not points:
            return self._proxy_cost(K, N, ratio)
        if ratio <= points[0][0]:
            if points[0][0] == 0:
                return points[0][1]
            # Interpolate from an implicit ratio=0 baseline at zero incremental cost.
            return points[0][1] * ratio / max(points[0][0], 1e-12)
        if ratio >= points[-1][0]:
            return points[-1][1]
        for (r0, c0), (r1, c1) in zip(points[:-1], points[1:]):
            if r0 <= ratio <= r1:
                alpha = (ratio - r0) / max(r1 - r0, 1e-12)
                return c0 + alpha * (c1 - c0)
        return points[-1][1]

    def delta(self, K: int, N: int, ratio: float) -> float:
        return float(self.cost(K, N, ratio) - self.cost(K, N, 0.0))


def quantize_symmetric(x: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    scale = torch.clamp(scale, min=eps)
    # Use signed symmetric int4 levels.  This proxy is for ranking candidates;
    # evaluator/kernel remains the source of truth for final PPL/latency.
    q = torch.round(x / scale).clamp(-7, 7)
    return q * scale


def row_threshold(x: torch.Tensor, percentile: float, eps: float) -> torch.Tensor:
    # percentile is 0-100 at the script interface.
    q = min(max(float(percentile) / 100.0, 0.0), 1.0)
    return torch.quantile(x.detach().abs().float(), q, dim=-1, keepdim=True).to(x.device).clamp(min=eps)


def channel_threshold(w: torch.Tensor, percentile: float, eps: float) -> torch.Tensor:
    q = min(max(float(percentile) / 100.0, 0.0), 1.0)
    return torch.quantile(w.detach().abs().float(), q, dim=-1, keepdim=True).to(w.device).clamp(min=eps)


def quantize_activation_single(x: torch.Tensor, percentile: float, eps: float) -> torch.Tensor:
    threshold = row_threshold(x, percentile, eps)
    return quantize_symmetric(x.clamp(-threshold, threshold), threshold / 7.0, eps)


def quantize_weight_per_out_channel(w: torch.Tensor, percentile: float, eps: float) -> torch.Tensor:
    threshold = channel_threshold(w, percentile, eps)
    return quantize_symmetric(w.clamp(-threshold, threshold), threshold / 7.0, eps)


def quantize_activation_split(
    x: torch.Tensor,
    ratio: float,
    activation_percentile: float,
    tail_percentile: float,
    eps: float,
) -> torch.Tensor:
    ratio = max(float(ratio), 0.0)
    if ratio <= 0.0:
        return quantize_activation_single(x, activation_percentile, eps)
    K = x.shape[-1]
    R = int(math.ceil(K * ratio))
    R = min(max(R, 0), K)
    if R <= 0:
        return quantize_activation_single(x, activation_percentile, eps)
    abs_x = x.detach().abs()
    top_indices = torch.topk(abs_x, k=R, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask.scatter_(-1, top_indices, True)

    body = x.masked_fill(mask, 0.0)
    tail = x.masked_fill(~mask, 0.0)

    body_threshold = row_threshold(body, activation_percentile, eps)
    q_body = quantize_symmetric(
        body.clamp(-body_threshold, body_threshold),
        body_threshold / 7.0,
        eps,
    )

    if tail_percentile >= 100.0:
        tail_threshold = tail.detach().abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    else:
        tail_threshold = row_threshold(tail, tail_percentile, eps)
    q_tail = quantize_symmetric(
        tail.clamp(-tail_threshold, tail_threshold),
        tail_threshold / 7.0,
        eps,
    )
    return q_body + q_tail


def linear_quant_output(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_indices: torch.Tensor,
    candidate: Candidate,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    w = weight.index_select(0, output_indices)
    b = bias.index_select(0, output_indices) if bias is not None else None
    y_fp = F.linear(x, w, b)
    q_w = quantize_weight_per_out_channel(w, candidate.weight_percentile, eps)
    q_x = quantize_activation_split(
        x,
        ratio=candidate.ratio,
        activation_percentile=candidate.activation_percentile,
        tail_percentile=candidate.tail_percentile,
        eps=eps,
    )
    y_q = F.linear(q_x, q_w, b)
    return y_fp, y_q


def ce_proxy_metrics(
    *,
    x: torch.Tensor,
    grad_y: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_indices: torch.Tensor,
    candidate: Candidate,
    eps: float,
) -> Dict[str, float]:
    y_fp, y_q = linear_quant_output(x, weight, bias, output_indices, candidate, eps)
    g = grad_y.index_select(-1, output_indices).to(y_q.device).float()
    signed_per_row = ((y_q.float() - y_fp.float()) * g).sum(dim=-1)
    ce_signed = signed_per_row.mean()
    ce_pos = torch.relu(signed_per_row).mean()
    ce_abs = signed_per_row.abs().mean()
    return {
        "ce_signed": float(ce_signed.detach().cpu()),
        "ce_pos": float(ce_pos.detach().cpu()),
        "ce_abs": float(ce_abs.detach().cpu()),
    }


def select_metric(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f"Unknown CE metric {metric_name}; available={sorted(metrics)}")
    return float(metrics[metric_name])


def sample_pairs(
    x: torch.Tensor,
    grad: torch.Tensor,
    limit: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = x.reshape(-1, x.shape[-1])
    grad = grad.reshape(-1, grad.shape[-1])
    if x.shape[0] != grad.shape[0]:
        raise RuntimeError(f"x/grad row mismatch: {x.shape} vs {grad.shape}")
    if len(x) <= limit:
        return x, grad
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    indices = torch.randperm(len(x), generator=generator)[:limit]
    return x.index_select(0, indices), grad.index_select(0, indices)


def build_candidates(args: argparse.Namespace) -> List[Candidate]:
    ratios = [0.0] if args.mode == "single_scale" else parse_float_list(args.ratio_candidates)
    activation_percentiles = parse_float_list(args.activation_percentile_candidates)
    weight_percentiles = parse_float_list(args.weight_percentile_candidates)
    tail_percentiles = parse_float_list(args.tail_percentile_candidates)
    candidates: List[Candidate] = []
    for ratio in ratios:
        if ratio <= 0.0:
            for a_p in activation_percentiles:
                for w_p in weight_percentiles:
                    candidates.append(Candidate(0.0, a_p, w_p, 100.0))
        else:
            for a_p in activation_percentiles:
                for w_p in weight_percentiles:
                    for t_p in tail_percentiles:
                        candidates.append(Candidate(ratio, a_p, w_p, t_p))
    # Stable de-duplication.
    unique = {}
    for c in candidates:
        unique[(c.ratio, c.activation_percentile, c.weight_percentile, c.tail_percentile)] = c
    return list(unique.values())


def candidate_key(candidate: Candidate) -> Tuple[float, float, float, float]:
    return (
        float(candidate.ratio),
        float(candidate.activation_percentile),
        float(candidate.weight_percentile),
        float(candidate.tail_percentile),
    )


def optimize_linear_by_candidates(
    *,
    module_name: str,
    x_cpu: torch.Tensor,
    grad_cpu: torch.Tensor,
    module: nn.Linear,
    args: argparse.Namespace,
    latency_table: ShapeLatencyTable,
    seed: int,
) -> Dict[str, Any]:
    started = time.time()
    device = module.weight.device
    K = int(module.in_features)
    N = int(module.out_features)
    x_rows, grad_rows = sample_pairs(x_cpu, grad_cpu, args.capture_rows, seed)
    if len(x_rows) < 4:
        raise RuntimeError(f"{module_name}: insufficient captured rows: {len(x_rows)}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 17)
    output_count = min(args.out_channels, N)
    if args.output_channel_sampling == "linspace":
        output_indices = torch.linspace(0, N - 1, steps=output_count).round().long().unique()
    else:
        output_indices = torch.randperm(N, generator=generator)[:output_count].sort().values
    output_indices = output_indices.to(device)

    x = x_rows.to(device, dtype=torch.float32)
    grad_y = grad_rows.to(device, dtype=torch.float32)
    weight = module.weight.detach().to(device, dtype=torch.float32)
    bias = module.bias.detach().to(device, dtype=torch.float32) if module.bias is not None else None

    anchor = Candidate(
        ratio=0.0,
        activation_percentile=args.anchor_activation_percentile,
        weight_percentile=args.anchor_weight_percentile,
        tail_percentile=100.0,
    )
    candidates = build_candidates(args)
    if candidate_key(anchor) not in {candidate_key(c) for c in candidates}:
        candidates.insert(0, anchor)

    with torch.no_grad():
        anchor_metrics = ce_proxy_metrics(
            x=x,
            grad_y=grad_y,
            weight=weight,
            bias=bias,
            output_indices=output_indices,
            candidate=anchor,
            eps=args.eps,
        )
        anchor_value = select_metric(anchor_metrics, args.ce_metric)

        evaluated: List[CandidateMetrics] = []
        for candidate in candidates:
            ratio_projected = project_ratio_safe(candidate.ratio, args.mode)
            effective_candidate = Candidate(
                ratio=ratio_projected,
                activation_percentile=candidate.activation_percentile,
                weight_percentile=candidate.weight_percentile,
                tail_percentile=candidate.tail_percentile,
            )
            metrics = ce_proxy_metrics(
                x=x,
                grad_y=grad_y,
                weight=weight,
                bias=bias,
                output_indices=output_indices,
                candidate=effective_candidate,
                eps=args.eps,
            )
            ce_metric = select_metric(metrics, args.ce_metric)
            delta_ce = ce_metric - anchor_value
            ce_gain = -delta_ce
            R = int(round(K * ratio_projected)) if args.mode == "split_v6" else 0
            latency_cost = latency_table.cost(K, N, ratio_projected)
            latency_delta = latency_table.delta(K, N, ratio_projected)
            score = delta_ce + args.latency_lambda * latency_delta
            if ratio_projected > 0.0 and ce_gain < args.min_ce_gain:
                score += args.no_gain_penalty
            evaluated.append(
                CandidateMetrics(
                    candidate=effective_candidate,
                    ratio_projected=ratio_projected,
                    R=R,
                    ce_signed=metrics["ce_signed"],
                    ce_pos=metrics["ce_pos"],
                    ce_abs=metrics["ce_abs"],
                    ce_metric=ce_metric,
                    delta_ce=delta_ce,
                    ce_gain=ce_gain,
                    latency_cost=latency_cost,
                    latency_delta=latency_delta,
                    score=score,
                )
            )

    # Always allow the anchor; among equal scores, prefer lower latency, then lower ratio.
    best = min(evaluated, key=lambda m: (m.score, m.latency_delta, m.ratio_projected))
    selected = best.candidate
    selected_ratio = 0.0 if args.mode != "split_v6" else float(best.ratio_projected)
    selected_R = 0 if args.mode != "split_v6" else int(round(K * selected_ratio))

    history = []
    for item in sorted(evaluated, key=lambda m: (m.score, m.latency_delta, m.ratio_projected)):
        history.append({
            "ratio": item.candidate.ratio,
            "ratio_projected": item.ratio_projected,
            "R": item.R,
            "activation_percentile": item.candidate.activation_percentile,
            "weight_percentile": item.candidate.weight_percentile,
            "tail_percentile": item.candidate.tail_percentile,
            "ce_signed": item.ce_signed,
            "ce_pos": item.ce_pos,
            "ce_abs": item.ce_abs,
            "ce_metric": item.ce_metric,
            "delta_ce": item.delta_ce,
            "ce_gain": item.ce_gain,
            "latency_cost": item.latency_cost,
            "latency_delta": item.latency_delta,
            "score": item.score,
        })

    result = {
        "module_name": module_name,
        "mode": args.mode,
        "search_algorithm": "shape_latency_cegrad_candidate_v1",
        "in_features": K,
        "out_features": N,
        "K": K,
        "N": N,
        "shape": f"K{K}_N{N}",
        "mac_weight": int(K * N),
        "ratio_lambda": float(args.latency_lambda),
        "latency_lambda": float(args.latency_lambda),
        "ce_metric_name": args.ce_metric,
        "ratio_continuous": selected_ratio,
        "ratio_projected": selected_ratio,
        "ratio": selected_ratio,
        "projected_ratio": selected_ratio,
        "used_ratio": selected_ratio,
        "split_ratio": selected_ratio,
        "R": selected_R,
        "projected_R": selected_R,
        "used_R": selected_R,
        "activation_percentile": float(selected.activation_percentile),
        "weight_percentile": float(selected.weight_percentile),
        "tail_percentile": float(selected.tail_percentile),
        "anchor_activation_percentile": float(anchor.activation_percentile),
        "anchor_weight_percentile": float(anchor.weight_percentile),
        "anchor_ce_signed": anchor_metrics["ce_signed"],
        "anchor_ce_pos": anchor_metrics["ce_pos"],
        "anchor_ce_abs": anchor_metrics["ce_abs"],
        "anchor_ce_metric": anchor_value,
        "selected_ce_signed": best.ce_signed,
        "selected_ce_pos": best.ce_pos,
        "selected_ce_abs": best.ce_abs,
        "selected_ce_metric": best.ce_metric,
        "delta_ce": best.delta_ce,
        "ce_gain": best.ce_gain,
        "latency_cost": best.latency_cost,
        "latency_delta": best.latency_delta,
        "shape_latency_proxy": best.latency_delta,
        "score": best.score,
        "best_val_total": best.score,
        "best_val_ce": best.ce_metric,
        "best_val_ratio_cost": best.latency_delta,
        "captured_rows": int(len(x_rows)),
        "sampled_output_channels": int(len(output_indices)),
        "elapsed_seconds": time.time() - started,
        "history": history[: args.keep_history_topk] if args.keep_history_topk > 0 else history,
    }
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single_scale", "split_v6"], default="split_v6")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument(
        "--local_source",
        action="append",
        default=[],
        help="NAME=/absolute/path/train.jsonl[@WEIGHT]; repeat for a mixed corpus",
    )
    parser.add_argument("--reservoir_docs_per_source", type=int, default=2048)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rotation_config", required=True)
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capture_rows", type=int, default=1024)
    parser.add_argument("--out_channels", type=int, default=256)
    parser.add_argument("--output_channel_sampling", choices=["random", "linspace"], default="linspace")
    parser.add_argument("--max_layers", type=int, default=-1)
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument(
        "--ratio_candidates",
        default="0,0.00125,0.0025,0.005,0.01,0.02,0.04",
    )
    parser.add_argument("--activation_percentile_candidates", default="99.5,99.75,99.9")
    parser.add_argument("--weight_percentile_candidates", default="99.75,99.9,100")
    parser.add_argument("--tail_percentile_candidates", default="100")
    parser.add_argument("--anchor_activation_percentile", type=float, default=99.5)
    parser.add_argument("--anchor_weight_percentile", type=float, default=99.75)

    parser.add_argument("--ce_metric", choices=["ce_pos", "ce_abs", "ce_signed"], default="ce_pos")
    parser.add_argument("--min_ce_gain", type=float, default=0.0)
    parser.add_argument("--no_gain_penalty", type=float, default=1e9)

    parser.add_argument("--latency_table", default=None)
    parser.add_argument("--latency_batch", default="b16")
    parser.add_argument("--latency_lambda", type=float, default=0.0)
    parser.add_argument(
        "--missing_latency",
        choices=["error", "proxy"],
        default="error",
        help="If latency_table is given and a shape is missing, error by default. "
             "Use proxy for K*N*ratio fallback.",
    )
    parser.add_argument("--latency_unit_scale", type=float, default=1.0)
    parser.add_argument("--keep_history_topk", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_directory = Path(args.out_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    latency_table = ShapeLatencyTable(
        Path(args.latency_table) if args.latency_table else None,
        batch_column=args.latency_batch,
        missing=args.missing_latency,
        latency_unit_scale=args.latency_unit_scale,
    )

    corpus_metadata = None
    if args.local_source:
        if build_local_mixed_trainloader is None:
            raise RuntimeError("--local_source requested, but local_mixed_loader_v74 cannot be imported")
        loader, corpus_metadata = build_local_mixed_trainloader(
            raw_sources=args.local_source,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
            model=args.model,
            hf_token=args.hf_token,
            reservoir_docs_per_source=args.reservoir_docs_per_source,
        )
        args.dataset = "local_mixed_v74"
    else:
        loader = data_utils.get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=args.seqlen,
            eval_mode=False,
            hf_token=args.hf_token,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0")
    model = H.model_utils.get_model(args.model, args.hf_token)
    model.seqlen = args.seqlen
    model.config.use_cache = False
    model.train(False)
    set_parameter_requires_grad(model, False)

    rotation_flags = H.build_rotation_flags(model, args.rotation_config)
    hadamard_cache = H.apply_offline_weight_rotation(model, rotation_flags)

    inputs, token_ids, layer_kwargs = capture_first_layer_inputs_and_tokens(
        model, loader, device, args.nsamples, args.seqlen
    )
    outputs = torch.zeros_like(inputs)

    partial_path = output_directory / "policy_partial.json"
    if partial_path.exists():
        policy = json.loads(partial_path.read_text(encoding="utf-8"))
        existing_mode = policy.get("metadata", {}).get("mode")
        if existing_mode != args.mode:
            raise RuntimeError(
                f"Existing partial policy mode={existing_mode}, requested mode={args.mode}. "
                "Use a fresh out_dir."
            )
    else:
        policy = {
            "metadata": {
                "version": "cegrad_shape_latency_v1",
                "base": "calibrate_per_linear_v74-compatible traversal",
                "mode": args.mode,
                "model": args.model,
                "dataset": args.dataset,
                "local_corpus": corpus_metadata,
                "nsamples": args.nsamples,
                "seqlen": args.seqlen,
                "seed": args.seed,
                "rotation_config": args.rotation_config,
                "surrogate": "first_order_ce_gradient_proxy_plus_shape_latency",
                "ce_metric": args.ce_metric,
                "latency_table": str(args.latency_table) if args.latency_table else None,
                "latency_batch": args.latency_batch,
                "latency_lambda": args.latency_lambda,
                "ratio_candidates": parse_float_list(args.ratio_candidates),
                "activation_percentile_candidates": parse_float_list(args.activation_percentile_candidates),
                "weight_percentile_candidates": parse_float_list(args.weight_percentile_candidates),
                "tail_percentile_candidates": parse_float_list(args.tail_percentile_candidates),
                "anchor_activation_percentile": args.anchor_activation_percentile,
                "anchor_weight_percentile": args.anchor_weight_percentile,
                "activation_semantics": (
                    "single per-token A4 percentile clipping"
                    if args.mode == "single_scale"
                    else "hard top-ratio split proxy; body A4 percentile; tail maxabs/percentile"
                ),
                "weight_semantics": "per-output-channel W4 percentile proxy",
            },
            "modules": {},
        }

    layers = model.model.layers
    number_of_layers = len(layers) if args.max_layers < 0 else min(len(layers), args.max_layers)

    for layer_id in range(number_of_layers):
        print(f"\n[CEGRAD:{args.mode}] layer {layer_id}/{number_of_layers - 1}", flush=True)
        layer = layers[layer_id].to(device).eval()
        linears = local_linears(layer)

        rotation_handles = register_rotation_hooks(linears, layer_id, rotation_flags, hadamard_cache)
        captured_inputs: Dict[str, List[torch.Tensor]] = {name: [] for name in linears}
        captured_outputs: Dict[str, List[torch.Tensor]] = {name: [] for name in linears}
        capture_handles: List[Any] = []

        for local_name, module in linears.items():
            def make_pre_capture(name: str):
                def hook(_module: nn.Module, hook_inputs: Tuple[torch.Tensor, ...]):
                    captured_inputs[name].append(hook_inputs[0].detach().to("cpu", dtype=torch.float16))
                    return None
                return hook

            def make_output_capture(name: str):
                def hook(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor):
                    if not torch.is_tensor(output):
                        raise RuntimeError(f"Unexpected non-tensor Linear output for {name}: {type(output)}")
                    output.retain_grad()
                    captured_outputs[name].append(output)
                    return None
                return hook

            capture_handles.append(module.register_forward_pre_hook(make_pre_capture(local_name)))
            capture_handles.append(module.register_forward_hook(make_output_capture(local_name)))

        for sample_id in range(args.nsamples):
            model.zero_grad(set_to_none=True)
            kwargs = slice_kwargs(layer_kwargs, int(inputs[sample_id].shape[0]))
            hidden_in = inputs[sample_id].unsqueeze(0).to(device).detach().requires_grad_(True)
            tokens = token_ids[sample_id].unsqueeze(0).to(device)
            hidden = layer_forward(layer, hidden_in, kwargs)
            outputs[sample_id].copy_(hidden.detach().squeeze(0))
            logits = forward_suffix_from_layer(model, layer_id, hidden, kwargs)
            loss = compute_lm_loss(logits, tokens)
            loss.backward()

            # Move output gradients to CPU immediately and drop graph references.
            for name, tensors in captured_outputs.items():
                last = tensors[-1]
                if last.grad is None:
                    raise RuntimeError(f"Missing retained grad for {name} at sample {sample_id}")
                tensors[-1] = last.grad.detach().to("cpu", dtype=torch.float16)
            del hidden_in, hidden, logits, loss
            torch.cuda.empty_cache()

        for handle in capture_handles + rotation_handles:
            handle.remove()

        for local_name, module in linears.items():
            full_name = f"model.layers.{layer_id}.{local_name}"
            if full_name in policy["modules"]:
                print(f"[SKIP] {full_name}", flush=True)
                continue
            x_values = torch.cat(captured_inputs[local_name], dim=0)
            grad_values = torch.cat(captured_outputs[local_name], dim=0)
            result = optimize_linear_by_candidates(
                module_name=full_name,
                x_cpu=x_values,
                grad_cpu=grad_values,
                module=module,
                args=args,
                latency_table=latency_table,
                seed=args.seed * 100000 + layer_id * 1000 + len(policy["modules"]),
            )
            policy["modules"][full_name] = result
            atomic_json(partial_path, policy)
            print(
                f"[SELECTED] {full_name} "
                f"shape=K{result['K']}_N{result['N']} "
                f"r={100*result['ratio_projected']:.4f}% "
                f"R={result['R']} "
                f"A-p={result['activation_percentile']:.3f} "
                f"W-p={result['weight_percentile']:.3f} "
                f"ce_gain={result['ce_gain']:.6e} "
                f"lat_d={result['latency_delta']:.6e} "
                f"score={result['score']:.6e}",
                flush=True,
            )

        layers[layer_id] = layer.cpu()
        del layer, captured_inputs, captured_outputs
        inputs, outputs = outputs, inputs
        torch.cuda.empty_cache()

    modules = list(policy["modules"].values())
    ratios = np.asarray([item["ratio_projected"] for item in modules], dtype=float)
    mac = np.asarray([item["mac_weight"] for item in modules], dtype=float)
    latency_delta = np.asarray([item.get("latency_delta", 0.0) for item in modules], dtype=float)
    ce_gain = np.asarray([item.get("ce_gain", 0.0) for item in modules], dtype=float)
    activation_percentiles = np.asarray([item["activation_percentile"] for item in modules], dtype=float)
    weight_percentiles = np.asarray([item["weight_percentile"] for item in modules], dtype=float)
    sum_R = int(sum(int(item.get("R", 0)) for item in modules))

    policy["summary"] = {
        "mode": args.mode,
        "module_count": len(modules),
        "mean_projected_ratio_unweighted": float(ratios.mean()) if len(ratios) else 0.0,
        "mac_weighted_projected_ratio": (
            float(np.sum(mac * ratios) / np.sum(mac)) if len(mac) and np.sum(mac) > 0 else 0.0
        ),
        "zero_ratio_module_count": int((ratios == 0).sum()) if len(ratios) else 0,
        "split_module_count": int((ratios > 0).sum()) if len(ratios) else 0,
        "sum_R": sum_R,
        "shape_latency_proxy_sum": float(latency_delta.sum()) if len(latency_delta) else 0.0,
        "ce_gain_sum": float(ce_gain.sum()) if len(ce_gain) else 0.0,
        "activation_percentile_min": float(activation_percentiles.min()) if len(activation_percentiles) else 0.0,
        "activation_percentile_mean": float(activation_percentiles.mean()) if len(activation_percentiles) else 0.0,
        "activation_percentile_max": float(activation_percentiles.max()) if len(activation_percentiles) else 0.0,
        "weight_percentile_min": float(weight_percentiles.min()) if len(weight_percentiles) else 0.0,
        "weight_percentile_mean": float(weight_percentiles.mean()) if len(weight_percentiles) else 0.0,
        "weight_percentile_max": float(weight_percentiles.max()) if len(weight_percentiles) else 0.0,
    }
    atomic_json(output_directory / "policy.json", policy)

    rows: List[Dict[str, Any]] = []
    for name, config in sorted(policy["modules"].items()):
        rows.append({
            "module_name": name,
            "mode": args.mode,
            "shape": config["shape"],
            "K": config["K"],
            "N": config["N"],
            "ratio_projected": config["ratio_projected"],
            "R": config["R"],
            "activation_percentile": config["activation_percentile"],
            "weight_percentile": config["weight_percentile"],
            "tail_percentile": config["tail_percentile"],
            "ce_gain": config["ce_gain"],
            "delta_ce": config["delta_ce"],
            "latency_delta": config["latency_delta"],
            "score": config["score"],
            "mac_weight": config["mac_weight"],
        })
    write_csv(output_directory / "policy_summary.csv", rows)
    print(json.dumps(policy["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
