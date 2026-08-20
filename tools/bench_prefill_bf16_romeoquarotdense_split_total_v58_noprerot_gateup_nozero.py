"""No-pre-rotation gate/up variant with epilogue-mode v57 prepare.

Compared with `bench_prefill_bf16_romeoquarotdense_split_total_v58_noprerot_gateup.py`,
this version removes the compatibility-only `body_q_top.zero_()` from the v57
threshold prepare path. The v58 oct epilogue only consumes indices/top_q/scales.
"""
from __future__ import annotations

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
import bench_qwen3_l18_split_opts_v57 as V57  # noqa: E402
import bench_qwen3_l18_split_opts_v58 as V58  # noqa: E402
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from fused_threshold_topr_pack_ext_v57 import load_threshold_topr_pack_ext  # noqa: E402


_ORIGINAL_BUILD_SPLIT_LAYER = BASE.build_split_layer
_EPILOGUE_EXT = None


def install_threshold_topr_prepare_v57_nozero():
    def patch_threshold_topr_prepare_v57_nozero(mod: nn.Module, threshold_ext) -> bool:
        if getattr(mod, "_threshold_topr_prepare_v57_nozero", False):
            return False

        orig_prepare = mod._prepare_split
        idx_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

        def threshold_prepare(self, A: torch.Tensor, scratch: Dict[str, torch.Tensor]):
            M, K = int(A.shape[0]), int(A.shape[1])
            R, descending_rank, _ = V57.split_select_params(self)
            if K not in (4096, 12288) or R <= 0 or R > 512 or A.dtype != torch.float16 or not A.is_contiguous():
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
            return tail_indices, tail_q, body_q_top

        mod._prepare_split = types.MethodType(threshold_prepare, mod)
        mod._threshold_topr_prepare_v57_nozero = True
        mod._threshold_topr_idx_cache_v57_nozero = idx_cache
        return True

    V43.patch_fused_topr_prepare = patch_threshold_topr_prepare_v57_nozero
    BASE.log("[INSTALL_THRESHOLD_TOPR_PREPARE_V57_NOZERO] K in {4096,12288}, R<=512")


def load_fused_topr_pack_ext(verbose: bool = False):
    install_threshold_topr_prepare_v57_nozero()
    ext = load_threshold_topr_pack_ext(verbose=verbose)
    BASE.log("[V58_NOPREROT_GATEUP_NOZERO_THRESHOLD_TOPR] v57 nozero prepare")
    return ext


def build_split_layer_no_prerot_gateup_nozero(
    base_layer,
    layer_idx,
    policy,
    rot_flags,
    B,
    main_ext,
    layout_ext,
    policy_pack_ext,
    eps,
    device,
    qext,
    fused_topr_ext,
    qwen_shared: bool,
):
    layer, rec = _ORIGINAL_BUILD_SPLIT_LAYER(
        base_layer,
        layer_idx,
        policy,
        rot_flags,
        B,
        main_ext,
        layout_ext,
        policy_pack_ext,
        eps,
        device,
        qext,
        fused_topr_ext,
        qwen_shared,
    )
    shared = V57.patch_gate_up_shared_hadamard(layer)
    if shared:
        BASE.log("[V58_NOPREROT_GATEUP_NOZERO_SHARED] " + __import__("json").dumps(shared))
    if _EPILOGUE_EXT is not None:
        patched = V58.patch_fused_sparse_epilogue(layer, qext, _EPILOGUE_EXT, "oct")
        BASE.log(f"[V58_NOPREROT_GATEUP_NOZERO_EPILOGUE_OCT] layer={layer_idx} modules={len(patched)}")
    return layer, rec


def main():
    global _EPILOGUE_EXT
    _EPILOGUE_EXT = load_fused_sparse_epilogue_ext(verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0"))))
    BASE.load_fused_topr_pack_ext = load_fused_topr_pack_ext
    BASE.build_split_layer = build_split_layer_no_prerot_gateup_nozero
    BASE.log("[BENCH_TOTAL_V58_NOPREROT_GATEUP_NOZERO] threshold_v57_nozero_gateup_epilogue_oct, no weight pre-rotation.")
    BASE.main()


if __name__ == "__main__":
    main()
