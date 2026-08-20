"""Total layer latency benchmark using the v57 threshold top-R frontend.

This wrapper intentionally leaves the v53 benchmark untouched. It reuses the
same BF16/RoMeO/Split timing pipeline, but replaces the fused top-R prepare path
with `fused_threshold_topr_pack_ext_v57` (K in {4096, 12288}, R <= 512).
"""
from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for p in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import bench_hadamard_three_schemes_quarot_dense_sharedqkv_inplace_fusedtopr_v43 as V43  # noqa: E402
import bench_prefill_bf16_romeoquarotdense_split_total_v53 as BASE  # noqa: E402
from fused_threshold_topr_pack_ext_v57 import load_threshold_topr_pack_ext  # noqa: E402


def split_select_params(mod):
    K = int(mod.K)
    R = int(getattr(mod, "R", 0))
    percentile = min(max(float(getattr(mod, "activation_percentile", 100.0)), 0.0), 100.0)
    body_len = K - R
    body_kth = min(K, max(1, int(math.ceil(body_len * percentile / 100.0))))
    descending_rank = K - body_kth + 1
    select_k = max(R, descending_rank)
    return R, descending_rank, select_k


def patch_threshold_topr_prepare_v57(mod: nn.Module, threshold_ext) -> bool:
    if getattr(mod, "_threshold_topr_prepare_v57_total", False):
        return False

    orig_prepare = mod._prepare_split
    idx_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

    def threshold_prepare(self, A: torch.Tensor, scratch: Dict[str, torch.Tensor]):
        M, K = int(A.shape[0]), int(A.shape[1])
        R, descending_rank, _ = split_select_params(self)
        if (
            K not in (4096, 12288)
            or R <= 0
            or R > 512
            or A.dtype != torch.float16
            or not A.is_contiguous()
        ):
            return orig_prepare(A, scratch)

        quant_buffers = self.scratch_pool.get_quant_buffers(M, self.K, self.N, R)
        tail_q = quant_buffers["top_q"]
        body_q_top = quant_buffers["body_q_top"]

        device_index = A.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (M, K, R, int(device_index))
        tail_indices = idx_cache.get(key)
        if tail_indices is None:
            tail_indices = torch.empty((M, R), dtype=torch.int32, device=A.device)
            idx_cache[key] = tail_indices

        threshold_ext.threshold_topr_pack(
            A,
            R,
            descending_rank,
            scratch["A_pack"],
            scratch["body_scale"],
            scratch["top_scale"],
            tail_q,
            tail_indices,
            self.eps,
        )
        body_q_top.zero_()
        return tail_indices, tail_q, body_q_top

    mod._prepare_split = types.MethodType(threshold_prepare, mod)
    mod._threshold_topr_prepare_v57_total = True
    mod._threshold_topr_idx_cache_v57_total = idx_cache
    return True


def load_fused_topr_pack_ext(verbose: bool = False):
    ext = load_threshold_topr_pack_ext(verbose=verbose)
    BASE.log("[V57_THRESHOLD_TOPR_ACTIVE] K in {4096,12288}, R<=512")
    return ext


def main():
    V43.patch_fused_topr_prepare = patch_threshold_topr_prepare_v57
    BASE.load_fused_topr_pack_ext = load_fused_topr_pack_ext
    BASE.log("[BENCH_TOTAL_V57] Reusing v53 total pipeline with v57 threshold top-R frontend.")
    BASE.main()


if __name__ == "__main__":
    main()
