"""Total layer latency benchmark with grouped-policy shared Split prepare.

Split path:
  * v57 threshold top-R prepare without body_q_top.zero_()
  * no up-type weight pre-rotation
  * q/k/v share online Hadamard and full Split prepare
  * gate/up share online Hadamard and full Split prepare
  * v58 oct fused dense-scale + sparse-correction epilogue

This file intentionally leaves prior benchmark/kernel files unchanged.
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
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


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
_QEXT = None


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
    BASE.log("[V61_SHARED_PREPARE_THRESHOLD_TOPR] v57 nozero prepare")
    return ext


def _shareable_split_group(mods: list[nn.Module]) -> bool:
    if len(mods) < 2:
        return False
    if not all(V43.is_real_policy_linear(m) and bool(getattr(m, "is_split", False)) for m in mods):
        return False
    if not all(hasattr(m, "B_col") and hasattr(m, "B_row") for m in mods):
        return False
    k0 = int(mods[0].K)
    r0 = int(mods[0].R)
    p0 = float(getattr(mods[0], "activation_percentile", 100.0))
    if r0 <= 0:
        return False
    for mod in mods[1:]:
        if int(mod.K) != k0 or int(mod.R) != r0:
            return False
        if abs(float(getattr(mod, "activation_percentile", 100.0)) - p0) > 1e-6:
            return False
    return True


def _prepare_shared(mod: nn.Module, A: torch.Tensor) -> dict:
    M = int(A.shape[0])
    scratch = mod.scratch_pool.get(M, int(mod.K), int(mod.N))
    indices, top_q, _ = mod._prepare_split(A, scratch)
    return {
        "M": M,
        "K": int(mod.K),
        "scratch": scratch,
        "indices": indices,
        "top_q": top_q,
    }


def _split_from_shared_prepare(mod: nn.Module, shared: dict) -> torch.Tensor:
    if _QEXT is None or _EPILOGUE_EXT is None:
        raise RuntimeError("shared prepare split path requires qext and epilogue ext")

    M = int(shared["M"])
    K = int(shared["K"])
    if K != int(mod.K):
        raise RuntimeError(f"{getattr(mod, 'name', mod)}: shared K mismatch {K} vs {mod.K}")

    device = shared["top_q"].device
    current = torch.cuda.current_stream(device)
    out_scratch = mod.scratch_pool.get(M, int(mod.K), int(mod.N))
    src_scratch = shared["scratch"]
    indices = shared["indices"]
    top_q = shared["top_q"]

    dense_stream = mod.dense_stream
    sparse_stream = mod.sparse_stream
    dense_stream.wait_stream(current)
    with torch.cuda.stream(dense_stream):
        C = V43.quarot_dense_gemm(_QEXT, src_scratch["A_pack"], mod.B_col, M, int(mod.N), int(mod.K))

    sparse_stream.wait_stream(dense_stream)
    with torch.cuda.stream(sparse_stream):
        _EPILOGUE_EXT.scale_sparse_epilogue_oct(
            C,
            src_scratch["body_scale"],
            top_q,
            indices,
            mod.B_row,
            src_scratch["top_scale"],
            mod.w_scale,
            out_scratch["Y_body"],
            int(mod.K),
        )

    indices.record_stream(sparse_stream)
    top_q.record_stream(sparse_stream)
    C.record_stream(sparse_stream)
    current.wait_stream(sparse_stream)
    return out_scratch["Y_body"]


def _linear_from_shared_prepare(mod: nn.Module, shared: dict, original_shape: tuple[int, ...]) -> torch.Tensor:
    out = _split_from_shared_prepare(mod, shared)
    if getattr(mod, "bias", None) is not None:
        out = out + mod.bias
    return out.view(*original_shape, int(mod.N))


def patch_gate_up_shared_prepare(layer: nn.Module) -> list[str]:
    mlp = getattr(layer, "mlp", None)
    if mlp is None or getattr(mlp, "_gate_up_shared_prepare_v61", False):
        return []

    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if not _shareable_split_group([gate, up]) or not V43.is_real_policy_linear(down):
        return []
    act_fn = getattr(mlp, "act_fn", None)
    if act_fn is None:
        return []

    def shared_mlp_forward(self, hidden_states: torch.Tensor):
        gate_proj = self.gate_proj
        up_proj = self.up_proj
        shared_hidden = hidden_states
        if bool(getattr(gate_proj, "rotate_online", False)) or bool(getattr(up_proj, "rotate_online", False)):
            shared_hidden = gate_proj._rotate(hidden_states)
        original_shape = shared_hidden.shape[:-1]
        A = shared_hidden.reshape(-1, int(gate_proj.K)).contiguous()
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        shared = _prepare_shared(gate_proj, A)
        gate_out = _linear_from_shared_prepare(gate_proj, shared, original_shape)
        up_out = _linear_from_shared_prepare(up_proj, shared, original_shape)
        return self.down_proj(self.act_fn(gate_out) * up_out)

    mlp.forward = types.MethodType(shared_mlp_forward, mlp)
    mlp._gate_up_shared_prepare_v61 = True
    return ["mlp.forward_shared_gate_up_prepare_v61"]


def patch_qkv_shared_prepare(layer: nn.Module) -> list[str]:
    attn = getattr(layer, "self_attn", None)
    if attn is None or getattr(attn, "_qkv_shared_prepare_v61", False):
        return []

    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    if not _shareable_split_group([q_proj, k_proj, v_proj]):
        return []

    def shared_attention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_proj = self.q_proj
        k_proj = self.k_proj
        v_proj = self.v_proj
        shared_hidden = hidden_states
        if (
            bool(getattr(q_proj, "rotate_online", False))
            or bool(getattr(k_proj, "rotate_online", False))
            or bool(getattr(v_proj, "rotate_online", False))
        ):
            shared_hidden = q_proj._rotate(hidden_states)

        A = shared_hidden.reshape(-1, int(q_proj.K)).contiguous()
        if A.dtype != torch.float16:
            A = A.to(torch.float16)
        shared = _prepare_shared(q_proj, A)

        query_states = _linear_from_shared_prepare(q_proj, shared, input_shape).view(hidden_shape)
        key_states = _linear_from_shared_prepare(k_proj, shared, input_shape).view(hidden_shape)
        value_states = _linear_from_shared_prepare(v_proj, shared, input_shape).view(hidden_shape)

        if hasattr(self, "q_norm"):
            query_states = self.q_norm(query_states)
        if hasattr(self, "k_norm"):
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_kwargs = dict(kwargs)
        if hasattr(self, "sliding_window"):
            attn_kwargs["sliding_window"] = self.sliding_window
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **attn_kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = types.MethodType(shared_attention_forward, attn)
    attn._qkv_shared_prepare_v61 = True
    return ["self_attn.forward_shared_qkv_prepare_v61"]


def build_split_layer_shared_prepare(
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
    patched_qkv = patch_qkv_shared_prepare(layer)
    patched_gateup = patch_gate_up_shared_prepare(layer)
    if patched_qkv or patched_gateup:
        BASE.log("[V61_SHARED_PREPARE] " + json.dumps(patched_qkv + patched_gateup))
    if _EPILOGUE_EXT is not None:
        patched = V58.patch_fused_sparse_epilogue(layer, qext, _EPILOGUE_EXT, "oct")
        BASE.log(f"[V61_SHARED_PREPARE_EPILOGUE_OCT] layer={layer_idx} modules={len(patched)}")
    return layer, rec


def main():
    global _EPILOGUE_EXT, _QEXT
    _EPILOGUE_EXT = load_fused_sparse_epilogue_ext(verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0"))))
    BASE.load_fused_topr_pack_ext = load_fused_topr_pack_ext

    original_build = BASE.build_split_layer

    def build_and_capture_qext(*args, **kwargs):
        global _QEXT
        _QEXT = args[10] if len(args) > 10 else kwargs.get("qext")
        return build_split_layer_shared_prepare(*args, **kwargs)

    BASE.build_split_layer = build_and_capture_qext
    BASE.log("[BENCH_TOTAL_V61_SHARED_PREPARE] threshold_v57_nozero_shared_qkv_gateup_prepare_epilogue_oct, no weight pre-rotation.")
    try:
        BASE.main()
    finally:
        BASE.build_split_layer = original_build


if __name__ == "__main__":
    main()
