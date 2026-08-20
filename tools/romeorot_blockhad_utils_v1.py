#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility helpers for unsmoothed RoMeO-style split rotation.

This file is intentionally standalone. It does not modify the original helper
modules; wrappers import it when a model has a Hadamard dimension unsupported by
the base QuaRot helper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


FALLBACK_BLOCK_DIM = 5120


@dataclass(frozen=True)
class HadPlan:
    dim: int
    had_k: torch.Tensor | None
    k: int
    block_dim: int = 0
    block_had_k: torch.Tensor | None = None
    block_k: int = 0

    @property
    def is_block(self) -> bool:
        return self.block_dim > 0


def get_had_plan(base_h, cache: Dict[int, HadPlan], dim: int) -> HadPlan:
    dim = int(dim)
    if dim in cache:
        return cache[dim]
    try:
        had_k, k = base_h.hadamard_utils.get_hadK(dim)
        plan = HadPlan(dim=dim, had_k=had_k, k=int(k))
    except AssertionError:
        if dim % FALLBACK_BLOCK_DIM != 0:
            raise
        block_had_k, block_k = base_h.hadamard_utils.get_hadK(FALLBACK_BLOCK_DIM)
        plan = HadPlan(
            dim=dim,
            had_k=None,
            k=0,
            block_dim=FALLBACK_BLOCK_DIM,
            block_had_k=block_had_k,
            block_k=int(block_k),
        )
    cache[dim] = plan
    return plan


def apply_had_plan(base_h, x: torch.Tensor, plan: HadPlan) -> torch.Tensor:
    if not plan.is_block:
        return base_h.apply_hadamard_last_dim(x, plan.had_k, plan.k)
    original_shape = x.shape
    blocks = int(original_shape[-1]) // int(plan.block_dim)
    y = x.reshape(*original_shape[:-1], blocks, int(plan.block_dim))
    y = base_h.apply_hadamard_last_dim(y, plan.block_had_k, plan.block_k)
    return y.reshape(original_shape)


def cache_for_legacy(cache: Dict[int, HadPlan]) -> Dict[int, Tuple[torch.Tensor, int]]:
    """Return only native plans for legacy code paths that inspect had_cache."""
    out: Dict[int, Tuple[torch.Tensor, int]] = {}
    for dim, plan in cache.items():
        if not plan.is_block:
            out[dim] = (plan.had_k, plan.k)
    return out
