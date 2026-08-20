"""Hybrid total layer latency benchmark for shared Split prepare smoke tests.

This wrapper keeps v61's q/k/v and gate/up full shared-prepare path, but only
installs the v58 sparse epilogue patch on layers where a shared-prepare group is
actually detected. Layers without a shareable group fall back to the original
Split forward so tiny-R-zero policies do not pay the v61 epilogue patch cost
when there is no q/k/v or gate/up prepare to share.
"""
from __future__ import annotations

import json
import os

import bench_prefill_bf16_romeoquarotdense_split_total_v53 as BASE
import bench_prefill_bf16_romeoquarotdense_split_total_v61_sharedprepare as V61
import bench_qwen3_l18_split_opts_v58 as V58
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext


def build_split_layer_shared_prepare_hybrid(*args, **kwargs):
    layer_idx = args[1] if len(args) > 1 else kwargs.get("layer_idx", -1)
    qext = args[10] if len(args) > 10 else kwargs.get("qext")

    layer, rec = V61._ORIGINAL_BUILD_SPLIT_LAYER(*args, **kwargs)
    patched_qkv = V61.patch_qkv_shared_prepare(layer)
    patched_gateup = V61.patch_gate_up_shared_prepare(layer)
    shared_patches = patched_qkv + patched_gateup

    if shared_patches:
        BASE.log("[V62_HYBRID_SHARED_PREPARE] " + json.dumps(shared_patches))
        if V61._EPILOGUE_EXT is not None:
            patched = V58.patch_fused_sparse_epilogue(layer, qext, V61._EPILOGUE_EXT, "oct")
            BASE.log(f"[V62_HYBRID_EPILOGUE_OCT] layer={layer_idx} modules={len(patched)}")
    else:
        BASE.log(f"[V62_HYBRID_NO_SHARED_PREPARE] layer={layer_idx} epilogue_patch=skipped")

    return layer, rec


def main():
    V61._EPILOGUE_EXT = load_fused_sparse_epilogue_ext(
        verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0")))
    )
    BASE.load_fused_topr_pack_ext = V61.load_fused_topr_pack_ext

    original_build = BASE.build_split_layer

    def build_and_capture_qext(*args, **kwargs):
        V61._QEXT = args[10] if len(args) > 10 else kwargs.get("qext")
        return build_split_layer_shared_prepare_hybrid(*args, **kwargs)

    BASE.build_split_layer = build_and_capture_qext
    BASE.log(
        "[BENCH_TOTAL_V62_SHARED_PREPARE_HYBRID] "
        "v61 shared qkv/gateup prepare; skip v58 epilogue patch when no shared group."
    )
    try:
        BASE.main()
    finally:
        BASE.build_split_layer = original_build


if __name__ == "__main__":
    main()
