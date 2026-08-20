"""v61 shared-prepare latency with up-type offline Hadamard pre-rotation.

This standalone wrapper leaves all original benchmark/kernel files unchanged.
Compared with v61_sharedprepare:
  * q/k/v/gate/up weights absorb the input Hadamard offline.
  * q/k/v and gate/up shared full prepare run on unrotated activations.
  * runtime online Hadamard is kept only for o_proj/down_proj.
"""
from __future__ import annotations

import copy
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
import bench_hadamard_three_schemes_quarot_dense_sharedqkv_inplace_fusedtopr_v43 as V43  # noqa: E402


UP_TYPE_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
)
DOWN_TYPE_SUFFIXES = (
    "self_attn.o_proj",
    "mlp.down_proj",
)


def is_up_type(local_name: str) -> bool:
    return local_name.endswith(UP_TYPE_SUFFIXES)


def is_down_type(local_name: str) -> bool:
    return local_name.endswith(DOWN_TYPE_SUFFIXES)


def _apply_weight_input_hadamard_(linear: nn.Linear, B, cache: dict[int, tuple[torch.Tensor, int]], device: torch.device):
    k = int(linear.weight.shape[1])
    if k not in cache:
        cache[k] = B.H.hadamard_utils.get_hadK(k)
    had_k, had_factor = cache[k]
    dtype = linear.weight.data.dtype
    w = linear.weight.data.to(device=device, dtype=torch.float16).contiguous()
    had = had_k.to(device=device, dtype=torch.float16).contiguous()
    w = B.H.hadamard_utils.matmul_hadU_cuda(w, had, had_factor)
    linear.weight.data = w.to(device="cpu", dtype=dtype).contiguous()


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


def build_split_layer_romeorot_prepare(
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
    layer_for_build = copy.deepcopy(base_layer).cpu().eval()
    patched = []
    had_cache: dict[int, tuple[torch.Tensor, int]] = {}

    for _parent, _child_name, local_name, linear in V43.iter_target_linears(layer_for_build):
        full_name = f"model.layers.{layer_idx}.{local_name}"
        if not bool(rot_flags.get(full_name, False)):
            continue
        if is_up_type(local_name):
            _apply_weight_input_hadamard_(linear, B, had_cache, device)
            patched.append(full_name)

    patched_rot_flags = dict(rot_flags)
    for name in patched:
        patched_rot_flags[name] = False

    layer, rec = V61._ORIGINAL_BUILD_SPLIT_LAYER(
        layer_for_build,
        layer_idx,
        policy,
        patched_rot_flags,
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

    patched_qkv = V61.patch_qkv_shared_prepare(layer)
    patched_gateup = V61.patch_gate_up_shared_prepare(layer)
    patched_online = patch_online_rotate_fp16(layer, B)
    if patched or patched_qkv or patched_gateup or patched_online:
        BASE.log(
            f"[V61_ROMEO_ROT_PREPARE] layer={layer_idx} "
            f"offline_up_hadamard={len(patched)} shared_patches={len(patched_qkv) + len(patched_gateup)} "
            f"online_rotate_fp16={patched_online}"
        )
    if V61._EPILOGUE_EXT is not None:
        import bench_qwen3_l18_split_opts_v58 as V58
        patched_epi = V58.patch_fused_sparse_epilogue(layer, qext, V61._EPILOGUE_EXT, "oct")
        BASE.log(f"[V61_ROMEO_ROT_PREPARE_EPILOGUE_OCT] layer={layer_idx} modules={len(patched_epi)}")
    return layer, rec


def main():
    V61._EPILOGUE_EXT = V61.load_fused_sparse_epilogue_ext(
        verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0")))
    )
    BASE.load_fused_topr_pack_ext = V61.load_fused_topr_pack_ext

    original_build = BASE.build_split_layer

    def build_and_capture_qext(*args, **kwargs):
        V61._QEXT = args[10] if len(args) > 10 else kwargs.get("qext")
        return build_split_layer_romeorot_prepare(*args, **kwargs)

    BASE.build_split_layer = build_and_capture_qext
    BASE.log("[BENCH_TOTAL_V61_ROMEO_ROT_PREPARE] up_type_offline_hadamard, qkv_gateup_no_online_hadamard_prepare, o_down_online_hadamard_only.")
    try:
        BASE.main()
    finally:
        BASE.build_split_layer = original_build


if __name__ == "__main__":
    main()
