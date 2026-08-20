"""No-pre-rotation gate/up variant with policy-aware sparse epilogue dispatch.

Base path:
  * v57 threshold prepare in epilogue mode (no body_q_top zero)
  * q/k/v shared online Hadamard from V43
  * gate/up shared online Hadamard
  * v58 sparse epilogue with per-module quad/oct dispatch

This is an experimental shape-policy dispatcher. It uses existing v58 quad/oct
epilogues and keeps the original policy file unchanged.
"""
from __future__ import annotations

import json
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
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from fused_threshold_topr_pack_ext_v57 import load_threshold_topr_pack_ext  # noqa: E402


_ORIGINAL_BUILD_SPLIT_LAYER = BASE.build_split_layer
_EPILOGUE_EXT = None
_QUAD_MIN_R = int(os.environ.get("DISPATCH_QUAD_MIN_R", "205"))


def install_threshold_topr_prepare_v57_nozero():
    def patch_threshold_topr_prepare_v57_nozero(mod: nn.Module, threshold_ext) -> bool:
        if getattr(mod, "_threshold_topr_prepare_v60_nozero", False):
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
        mod._threshold_topr_prepare_v60_nozero = True
        mod._threshold_topr_idx_cache_v60_nozero = idx_cache
        return True

    V43.patch_fused_topr_prepare = patch_threshold_topr_prepare_v57_nozero
    BASE.log("[INSTALL_THRESHOLD_TOPR_PREPARE_V60_NOZERO] K in {4096,12288}, R<=512")


def load_fused_topr_pack_ext(verbose: bool = False):
    install_threshold_topr_prepare_v57_nozero()
    ext = load_threshold_topr_pack_ext(verbose=verbose)
    BASE.log("[V60_DISPATCH_THRESHOLD_TOPR] v57 nozero prepare")
    return ext


def choose_epilogue_kind(mod) -> str:
    r = int(getattr(mod, "R", 0))
    n = int(getattr(mod, "N", 0))
    # High-R correction can become register-heavy with oct. Try quad for high R
    # on larger output tiles; keep oct for the many small-R modules.
    if r >= _QUAD_MIN_R and n >= 4096:
        return "quad"
    return "oct"


def patch_fused_sparse_epilogue_dispatch(layer: nn.Module, qext, epilogue_ext):
    patched = []
    for name, mod in layer.named_modules():
        if not V43.is_real_policy_linear(mod) or not bool(getattr(mod, "is_split", False)):
            continue
        if not hasattr(mod, "B_row"):
            continue
        kind = choose_epilogue_kind(mod)

        def make_split_compute(module_name, epilogue_kind):
            def split_compute(self, A, scratch, B_col, dense_ready_event, dense_stream, sparse_stream):
                M = int(A.shape[0])
                device = A.device
                current = torch.cuda.current_stream(device)
                indices, top_q, _ = self._prepare_split(A, scratch)

                dense_stream.wait_stream(current)
                with torch.cuda.stream(dense_stream):
                    C = V43.quarot_dense_gemm(qext, scratch["A_pack"], B_col, M, self.N, self.K)

                sparse_stream.wait_stream(dense_stream)
                with torch.cuda.stream(sparse_stream):
                    if epilogue_kind == "quad":
                        epilogue_ext.scale_sparse_epilogue_quad(
                            C,
                            scratch["body_scale"],
                            top_q,
                            indices,
                            self.B_row,
                            scratch["top_scale"],
                            self.w_scale,
                            scratch["Y_body"],
                            self.K,
                        )
                    elif epilogue_kind == "oct":
                        epilogue_ext.scale_sparse_epilogue_oct(
                            C,
                            scratch["body_scale"],
                            top_q,
                            indices,
                            self.B_row,
                            scratch["top_scale"],
                            self.w_scale,
                            scratch["Y_body"],
                            self.K,
                        )
                    else:
                        raise ValueError(epilogue_kind)

                indices.record_stream(sparse_stream)
                top_q.record_stream(sparse_stream)
                C.record_stream(sparse_stream)
                current.wait_stream(sparse_stream)
                return scratch["Y_body"]

            return split_compute

        mod._split_compute = types.MethodType(make_split_compute(name, kind), mod)
        mod._fused_sparse_epilogue_v60 = kind
        patched.append({
            "name": name,
            "K": int(getattr(mod, "K", 0)),
            "N": int(getattr(mod, "N", 0)),
            "R": int(getattr(mod, "R", 0)),
            "kind": kind,
        })
    BASE.log("[PATCH_FUSED_SPARSE_EPILOGUE_V60_DISPATCH] " + json.dumps(patched, indent=2))
    return patched


def build_split_layer_v60_dispatch(
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
        BASE.log("[V60_DISPATCH_GATEUP_SHARED] " + json.dumps(shared))
    if _EPILOGUE_EXT is not None:
        patched = patch_fused_sparse_epilogue_dispatch(layer, qext, _EPILOGUE_EXT)
        quad = sum(1 for p in patched if p["kind"] == "quad")
        oct_ = sum(1 for p in patched if p["kind"] == "oct")
        BASE.log(f"[V60_DISPATCH_EPILOGUE] layer={layer_idx} quad={quad} oct={oct_}")
    return layer, rec


def main():
    global _EPILOGUE_EXT
    _EPILOGUE_EXT = load_fused_sparse_epilogue_ext(verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0"))))
    BASE.load_fused_topr_pack_ext = load_fused_topr_pack_ext
    BASE.build_split_layer = build_split_layer_v60_dispatch
    BASE.log(f"[BENCH_TOTAL_V60_DISPATCH] nozero + gateup + quad/oct dispatch, quad_min_r={_QUAD_MIN_R}.")
    BASE.main()


if __name__ == "__main__":
    main()
