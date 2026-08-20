"""v61 shared-prepare latency wrapper with fp16 online Hadamard rotation.

This file is intentionally a non-invasive baseline wrapper. It preserves the
v61 shared-prepare and fused sparse epilogue behavior, but patches online
Hadamard rotation to stay in the input dtype so the current RoMeO faster
Hadamard implementation can run.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for p in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import bench_prefill_bf16_romeoquarotdense_split_total_v61_sharedprepare as V61  # noqa: E402
import bench_prefill_bf16_romeoquarotdense_split_total_v53 as BASE  # noqa: E402


def patch_online_rotate_fp16(layer: nn.Module, B) -> int:
    patched = 0
    for mod in layer.modules():
        if not bool(getattr(mod, "rotate_online", False)) or getattr(mod, "had_k", None) is None:
            continue

        def make_rotate():
            def _rotate(self, x: torch.Tensor) -> torch.Tensor:
                if not self.rotate_online:
                    return x
                had = self.had_k.to(device=x.device, dtype=x.dtype).contiguous()
                return B.H.hadamard_utils.matmul_hadU_cuda(x.contiguous(), had, self.had_factor)

            return _rotate

        mod._rotate = types.MethodType(make_rotate(), mod)
        patched += 1
    return patched


def build_split_layer_v61_fp16rotate(*args, **kwargs):
    layer, rec = V61.build_split_layer_shared_prepare(*args, **kwargs)
    B = args[4] if len(args) > 4 else kwargs["B"]
    layer_idx = args[1] if len(args) > 1 else kwargs.get("layer_idx", "?")
    patched = patch_online_rotate_fp16(layer, B)
    if patched:
        BASE.log(f"[V61_BASELINE_FP16_ROTATE] layer={layer_idx} online_rotate_fp16={patched}")
    return layer, rec


def main():
    V61._EPILOGUE_EXT = V61.load_fused_sparse_epilogue_ext(
        verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0")))
    )
    BASE.load_fused_topr_pack_ext = V61.load_fused_topr_pack_ext

    original_build = BASE.build_split_layer

    def build_and_capture_qext(*args, **kwargs):
        V61._QEXT = args[10] if len(args) > 10 else kwargs.get("qext")
        return build_split_layer_v61_fp16rotate(*args, **kwargs)

    BASE.build_split_layer = build_and_capture_qext
    BASE.log("[BENCH_TOTAL_V61_BASELINE_FP16_ROTATE] v61 semantics with dtype-preserving online Hadamard.")
    try:
        BASE.main()
    finally:
        BASE.build_split_layer = original_build


if __name__ == "__main__":
    main()
