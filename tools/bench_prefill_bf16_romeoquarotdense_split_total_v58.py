"""Total layer latency benchmark with the verified v57/v58 best split path.

This leaves the v53 total benchmark untouched. The Split variant reuses the
same policy/model timing loop, but swaps in:
  * v57 threshold top-R prepare
  * up-type weight pre-rotation for q/k/v/gate/up
  * v58 oct fused dense-scale + sparse-correction epilogue
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for p in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import bench_prefill_bf16_romeoquarotdense_split_total_v53 as BASE  # noqa: E402
import bench_qwen3_l18_split_opts_v57 as V57  # noqa: E402
import bench_qwen3_l18_split_opts_v58 as V58  # noqa: E402
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from fused_threshold_topr_pack_ext_v57 import load_threshold_topr_pack_ext  # noqa: E402


_ORIGINAL_BUILD_SPLIT_LAYER = BASE.build_split_layer
_EPILOGUE_EXT = None


def load_fused_topr_pack_ext(verbose: bool = False):
    V57.install_threshold_topr_prepare_v57()
    ext = load_threshold_topr_pack_ext(verbose=verbose)
    BASE.log("[V58_TOTAL_THRESHOLD_TOPR] v57 K in {4096,12288}, R<=512")
    return ext


def build_split_layer_best_combo(
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
    local_base_layer, local_rot_flags, prerot_records = V58.make_up_type_weight_prerotated_layer(
        base_layer,
        int(layer_idx),
        rot_flags,
        B,
        torch.device(device),
    )
    BASE.log(f"[V58_TOTAL_PREROT_UP] layer={layer_idx} modules={len(prerot_records)}")
    layer, rec = _ORIGINAL_BUILD_SPLIT_LAYER(
        local_base_layer,
        layer_idx,
        policy,
        local_rot_flags,
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
    local_base_layer.to("cpu")
    if _EPILOGUE_EXT is not None:
        patched = V58.patch_fused_sparse_epilogue(layer, qext, _EPILOGUE_EXT, "oct")
        BASE.log(f"[V58_TOTAL_EPILOGUE_OCT] layer={layer_idx} modules={len(patched)}")
    return layer, rec


def main():
    global _EPILOGUE_EXT
    _EPILOGUE_EXT = load_fused_sparse_epilogue_ext(verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0"))))
    BASE.load_fused_topr_pack_ext = load_fused_topr_pack_ext
    BASE.build_split_layer = build_split_layer_best_combo
    BASE.log("[BENCH_TOTAL_V58] Split path = threshold_v57_prerot_up_epilogue_oct.")
    BASE.main()


if __name__ == "__main__":
    main()
