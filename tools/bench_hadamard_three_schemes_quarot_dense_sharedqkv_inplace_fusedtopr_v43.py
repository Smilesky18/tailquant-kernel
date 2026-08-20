import argparse
import copy
import csv
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from transformers.initialization import no_init_weights
except Exception:
    try:
        from transformers.modeling_utils import no_init_weights
    except Exception:
        from contextlib import nullcontext as no_init_weights

ROMEO_ROOT = Path(os.environ.get("ROMEO_ROOT", "/data/yzy/RoMeo")).resolve()
ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
EXP_TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
_path_order = [EXP_TOOLS, ROMEO_ROOT, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant"]
for p in _path_order:
    sp = str(p)
    while sp in sys.path:
        sys.path.remove(sp)
sys.path[:0] = [str(p) for p in _path_order]
# Keep RoMeo ahead of the project root so qlinear.py imports RoMeo hadamard_utils/rotate.py.

import bench_layer_bf16_pure_split_no_gptq_v8 as V8
import bench_layer_qfactory_raw_backend_v19 as QF19
import bench_multimodel_all_layers_policy_fastqf_v29 as V29
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension
from fused_topr_pack_ext_v42 import load_fused_topr_pack_ext
from qlinear import MixedRQLinear
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, eager_attention_forward


TARGET_SUFFIXES = V29.TARGET_SUFFIXES


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--rotation_config", required=True)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--layers", default="all")
    p.add_argument("--variants", default="split,romeo,quarot")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--force_activation_percentile_100", action="store_true")
    p.add_argument("--romeo_activation_threshold", type=float, default=0.05)
    p.add_argument("--romeo_weight_threshold", type=float, default=0.05)
    p.add_argument("--romeo_multistream", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qfactory_fast_preset", default="qwen3_sm120_v1", choices=["none", "qwen3_sm120_v1"])
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", str(ROOT / "third_party/cutlass")))
    p.add_argument("--fused_topr_pack", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: List[dict]):
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_layer_once(layer, hidden_states, position_ids, position_embeddings):
    try:
        out = layer(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)
    except TypeError:
        try:
            out = layer(hidden_states, position_embeddings=position_embeddings)
        except TypeError:
            out = layer(hidden_states, position_ids=position_ids)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def bench_graph(fn, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    ms = float(start.elapsed_time(end) / iters)
    del graph
    torch.cuda.empty_cache()
    return ms


def iter_target_linears(layer: nn.Module):
    return V29.iter_target_linears(layer)


def policy_key(layer_idx: int, local_name: str) -> str:
    return f"model.layers.{layer_idx}.{local_name}"


def make_hadamard_cache_for_layer(B, layer: nn.Module):
    cache = {}
    for _, _, _, linear in iter_target_linears(layer):
        k = int(linear.in_features)
        if k not in cache:
            cache[k] = B.H.hadamard_utils.get_hadK(k)
    return cache


def build_split_layer_with_hadamard(base_layer, layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, eps, device, quarot_dense_ext=None, fused_topr_ext=None):
    BASE = getattr(B, "BASE")
    RealPolicyLinear = getattr(B, "RealPolicyLinear")
    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    linears = iter_target_linears(layer)
    had_cache = make_hadamard_cache_for_layer(B, layer)

    pure_shapes: Dict[Tuple[int, int], int] = {}
    split_shapes: Dict[Tuple[int, int], int] = {}
    records = []
    cfgs = {}
    for _, _, local_name, linear in linears:
        key = policy_key(layer_idx, local_name)
        cfg = policy["modules"].get(key)
        if cfg is None:
            raise KeyError(f"Policy missing {key}")
        cfgs[local_name] = cfg
        N, K = map(int, linear.weight.shape)
        ratio = float(cfg["ratio"])
        R = BASE.ceil_ratio_count(K, ratio)
        (split_shapes if R > 0 else pure_shapes)[(K, N)] = max((split_shapes if R > 0 else pure_shapes).get((K, N), 0), R)
        records.append({"name": key, "local_name": local_name, "K": K, "N": N, "R": int(R), "ratio": ratio})

    pure_pool = BASE.SharedScratchPool(device=device, max_r_by_shape=pure_shapes, split=False) if pure_shapes else None
    split_pool = BASE.SharedScratchPool(device=device, max_r_by_shape=split_shapes, split=True) if split_shapes else None

    for parent, child_name, local_name, linear in linears:
        cfg = cfgs[local_name]
        weight_cpu = linear.weight.detach().cpu().contiguous()
        bias_cpu = None if linear.bias is None else linear.bias.detach().cpu().contiguous()
        scale_cpu = V29.make_rtn_scale_cpu(weight_cpu, eps)
        ratio = float(cfg["ratio"])
        R = BASE.ceil_ratio_count(int(linear.in_features), ratio)
        mode = "dual_policy" if R > 0 else "pure"
        full_name = policy_key(layer_idx, local_name)
        rotate_online = bool(rot_flags.get(full_name, False))
        had_k, had_factor = had_cache.get(int(linear.in_features), (None, 1))
        repl = RealPolicyLinear(
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            mode=mode,
            weight_cpu=weight_cpu,
            bias_cpu=bias_cpu,
            policy_cfg=cfg,
            gptq_scale_cpu=scale_cpu,
            eps=eps,
            device=device,
            name=full_name,
            scratch_pool=split_pool if mode == "dual_policy" else pure_pool,
            prefetch_workspace=None,
            rotate_online=rotate_online,
            had_k=had_k if rotate_online else None,
            had_factor=had_factor if rotate_online else 1,
        )
        setattr(parent, child_name, repl)

    if quarot_dense_ext is not None:
        patch_quarot_dense_backend(layer, quarot_dense_ext, fused_topr_ext=fused_topr_ext)
    shared = patch_qwen_qkv_shared_preprocess(layer, enable_shared_topk=(fused_topr_ext is None))
    if shared:
        log("[PATCH_SHARED_QKV_V43] " + json.dumps(shared, indent=2))
    return layer.to(device=device).eval(), records


def install_qfactory_mixed_fast_preset():
    try:
        import qfactory.kernels.gemm_w4a4_mixed_precision as mixed
    except Exception as exc:
        log(f"[QFACTORY_MIXED_FAST_PRESET_ERROR] {type(exc).__name__}: {exc}")
        return

    def one_config_tunable_keys():
        return {
            "NStage": [2],
            "TileM": [128],
            "TileN": [128],
            "TileK": [128],
            "WarpM": [64],
            "WarpN": [64],
            "WarpK": [128],
        }

    mixed.generate_tunable_keys = one_config_tunable_keys
    log("[QFACTORY_MIXED_FAST_PRESET] installed one-config mixed kernel tuning space")


def is_real_policy_linear(m: nn.Module) -> bool:
    name = m.__class__.__name__
    return name == "RealPolicyLinear" or "RealPolicyLinear" in name


def quarot_dense_gemm(qext, A_pack: torch.Tensor, B_col: torch.Tensor, M: int, N: int, K: int) -> torch.Tensor:
    return qext.matmul(
        A_pack.view(M, K // 2).contiguous(),
        B_col.view(N, K // 2).contiguous(),
    )


def patch_quarot_dense_backend(layer: nn.Module, qext, fused_topr_ext=None):
    import types

    patched = []
    for name, mod in layer.named_modules():
        if not is_real_policy_linear(mod) or not hasattr(mod, "B_col"):
            continue

        def make_quarot_pure(module_name):
            def quarot_pure(self, A, scratch):
                M = int(A.shape[0])
                ext = getattr(self, "ext", None)
                if ext is None:
                    ext = getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError("cannot find ext/main_ext")

                output = torch.empty((M, self.N), dtype=torch.float16, device=A.device)
                ext.pack_a_full_s4(A, scratch["A_pack"], scratch["a_scale"], self.eps)
                C = quarot_dense_gemm(qext, scratch["A_pack"], self.B_col, M, self.N, self.K)
                ext.scale_i32_to_fp16(C, scratch["a_scale"], self.w_scale, output)
                return output

            return quarot_pure

        def make_quarot_split_compute(module_name):
            def quarot_split_compute(self, A, scratch, B_col, dense_ready_event, dense_stream, sparse_stream):
                M = int(A.shape[0])
                device = A.device
                current = torch.cuda.current_stream(device)
                ext = getattr(self, "ext", None)
                if ext is None:
                    ext = getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError("cannot find ext/main_ext")

                indices, top_q, _ = self._prepare_split(A, scratch)

                dense_stream.wait_stream(current)
                with torch.cuda.stream(dense_stream):
                    C = quarot_dense_gemm(qext, scratch["A_pack"], B_col, M, self.N, self.K)
                    ext.scale_i32_to_fp16(C, scratch["body_scale"], self.w_scale, scratch["Y_body"])

                sparse_stream.wait_stream(dense_stream)
                with torch.cuda.stream(sparse_stream):
                    ext.sparse_top_add_rowmajor_quad_shared(
                        top_q,
                        indices,
                        self.B_row,
                        scratch["top_scale"],
                        self.w_scale,
                        scratch["Y_body"],
                        self.K,
                    )

                indices.record_stream(sparse_stream)
                top_q.record_stream(sparse_stream)
                current.wait_stream(sparse_stream)
                return scratch["Y_body"]

            return quarot_split_compute

        mod._pure = types.MethodType(make_quarot_pure(name), mod)
        if getattr(mod, "is_split", False):
            mod._split_compute = types.MethodType(make_quarot_split_compute(name), mod)
            if fused_topr_ext is not None:
                patch_fused_topr_prepare(mod, fused_topr_ext)

        patched.append({
            "name": name,
            "K": int(mod.K),
            "N": int(mod.N),
            "is_split": bool(getattr(mod, "is_split", False)),
            "R": int(getattr(mod, "R", 0)),
            "ratio": float(getattr(mod, "ratio", 0.0)),
            "fused_topr_pack": bool(fused_topr_ext is not None and getattr(mod, "is_split", False) and int(mod.K) == 4096 and 0 < int(getattr(mod, "R", 0)) <= 256),
        })

    log("[PATCH_QUAROT_DENSE_BACKEND_V43] " + json.dumps(patched, indent=2))
    return patched


def _split_select_params(mod):
    import math
    K = int(mod.K)
    R = int(getattr(mod, "R", 0))
    percentile = min(max(float(getattr(mod, "activation_percentile", 100.0)), 0.0), 100.0)
    body_len = K - R
    body_kth = min(K, max(1, int(math.ceil(body_len * percentile / 100.0))))
    descending_rank = K - body_kth + 1
    select_k = max(R, descending_rank)
    return R, descending_rank, select_k


def patch_fused_topr_prepare(mod: nn.Module, fused_topr_ext) -> bool:
    if getattr(mod, "_fused_topr_prepare_v43", False):
        return False

    orig_prepare = mod._prepare_split
    idx_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

    def fused_prepare(self, A: torch.Tensor, scratch: Dict[str, torch.Tensor]):
        M, K = int(A.shape[0]), int(A.shape[1])
        R, descending_rank, _ = _split_select_params(self)
        if K != 4096 or R <= 0 or R > 256 or A.dtype != torch.float16 or not A.is_contiguous():
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

        fused_topr_ext.fused_topr_pack(
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

    mod._prepare_split = types.MethodType(fused_prepare, mod)
    mod._fused_topr_prepare_v43 = True
    mod._fused_topr_idx_cache_v43 = idx_cache
    return True


def patch_shared_topk_prepare_for_qkv(attn: nn.Module):
    qkv = [getattr(attn, name, None) for name in ("q_proj", "k_proj", "v_proj")]
    split_qkv = [m for m in qkv if is_real_policy_linear(m) and bool(getattr(m, "is_split", False))]
    if len(split_qkv) < 2:
        return []

    max_select_k = max(_split_select_params(m)[2] for m in split_qkv)
    shared_cache = {}
    patched = []

    def make_prepare(orig_prepare):
        def shared_prepare(self, A: torch.Tensor, scratch: Dict[str, torch.Tensor]):
            R, descending_rank, _ = _split_select_params(self)
            if R <= 0:
                return orig_prepare(A, scratch)
            if max_select_k < max(R, descending_rank):
                return orig_prepare(A, scratch)

            M, K = int(A.shape[0]), int(A.shape[1])
            key = (int(A.data_ptr()), M, K, int(max_select_k))
            cached = shared_cache.get(key)
            if cached is None:
                abs_A = A.abs().float()
                top_values, top_indices = torch.topk(
                    abs_A,
                    k=max_select_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                cached = (top_values, top_indices)
                shared_cache[key] = cached
            top_values, top_indices = cached

            body_threshold = top_values[:, descending_rank - 1].contiguous()
            tail_threshold = top_values[:, 0].contiguous()
            tail_indices = top_indices[:, :R]
            tail_indices, _ = torch.sort(tail_indices, dim=1)
            tail_indices = tail_indices.to(torch.int32).contiguous()

            quant_buffers = self.scratch_pool.get_quant_buffers(M, self.K, self.N, R)
            tail_q = quant_buffers["top_q"]
            body_q_top = quant_buffers["body_q_top"]
            self.policy_pack_ext.pack_policy_split(
                A,
                tail_indices,
                body_threshold,
                tail_threshold,
                scratch["A_pack"],
                scratch["body_scale"],
                scratch["top_scale"],
                tail_q,
                self.eps,
            )
            body_q_top.zero_()
            return tail_indices, tail_q, body_q_top

        return shared_prepare

    for m in split_qkv:
        if getattr(m, "_shared_topk_prepare_v39", False):
            continue
        m._prepare_split = types.MethodType(make_prepare(m._prepare_split), m)
        m._shared_topk_prepare_v39 = True
        m._shared_topk_cache_v39 = shared_cache
        patched.append(getattr(m, "name", m.__class__.__name__))
    attn._shared_topk_cache_v39 = shared_cache
    return patched


def patch_qwen_qkv_shared_preprocess(layer: nn.Module, enable_shared_topk: bool = True):
    attn = getattr(layer, "self_attn", None)
    if attn is None or getattr(attn, "_shared_qkv_preprocess_v39", False):
        return []

    patched_topk = patch_shared_topk_prepare_for_qkv(attn) if enable_shared_topk else []
    q_proj = getattr(attn, "q_proj", None)
    k_proj = getattr(attn, "k_proj", None)
    v_proj = getattr(attn, "v_proj", None)
    if not all(is_real_policy_linear(m) for m in (q_proj, k_proj, v_proj)):
        return patched_topk

    def shared_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cache = getattr(self, "_shared_topk_cache_v39", None)
        if isinstance(cache, dict):
            cache.clear()

        qkv = [self.q_proj, self.k_proj, self.v_proj]
        rotate_flags = [bool(getattr(m, "rotate_online", False)) for m in qkv]
        shared_hidden = hidden_states
        if any(rotate_flags):
            shared_hidden = self.q_proj._rotate(hidden_states)
            for m in qkv:
                m.rotate_online = False
        try:
            query_states = self.q_norm(self.q_proj(shared_hidden).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(self.k_proj(shared_hidden).view(hidden_shape)).transpose(1, 2)
            value_states = self.v_proj(shared_hidden).view(hidden_shape).transpose(1, 2)
        finally:
            for m, flag in zip(qkv, rotate_flags):
                m.rotate_online = flag

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    attn.forward = types.MethodType(shared_forward, attn)
    attn._shared_qkv_preprocess_v39 = True
    return ["self_attn.forward_shared_qkv_hadamard"] + patched_topk




def make_romeo_args(args):
    return SimpleNamespace(
        rotate="hadamard",
        smooth_quant=False,
        rotate_opt=True,
        capture_layer_ids="",
        a_group=None,
        w_group=None,
        w_clip=False,
        a_bits=4,
        w_bits=4,
        mixed_precision="bitweaver",
        qfactory_kernel=True,
        activation_threshold=float(args.romeo_activation_threshold),
        weight_threshold=float(args.romeo_weight_threshold),
        threshold_policy="percentage",
        multistream=bool(args.romeo_multistream),
        unifiedkernel=False,
    )


def build_romeo_layer(base_layer, layer_idx, args, device):
    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.bfloat16).eval()
    rargs = make_romeo_args(args)
    for submodule_name in ["self_attn", "mlp"]:
        submodule = getattr(layer, submodule_name)
        for name, module in list(submodule.named_children()):
            if not isinstance(module, torch.nn.Linear):
                continue
            norm_scale = layer.input_layernorm.weight.data if submodule_name == "self_attn" else layer.post_attention_layernorm.weight.data
            setattr(submodule, name, MixedRQLinear(rargs, module, name, layer_idx, norm_scale))
    return layer


def load_quarot_qwen_model(config):
    spec = importlib.util.spec_from_file_location("_romeo_modeling_qwen3_quarot_v35", str(ROMEO_ROOT / "modeling_qwen3_quarot.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import modeling_qwen3_quarot")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_romeo_modeling_qwen3_quarot_v35"] = module
    spec.loader.exec_module(module)
    with no_init_weights():
        model = module.QuarotQwen3ForCausalLM(config)
    return model.eval()


def load_quarot_llama_model(config):
    path = ROOT / "e2e/quantized_llama/modeling_llama.py"
    spec = importlib.util.spec_from_file_location("_kq_modeling_llama_quarot_v35", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import quarot llama modeling")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_kq_modeling_llama_quarot_v35"] = module
    spec.loader.exec_module(module)
    config._attn_implementation = "flash_attention_2"
    with no_init_weights():
        model = module.QuarotLlamaForCausalLM(config)
    return model.eval()


def build_quarot_model(model_name: str, local_files_only: bool):
    ext = load_quarot_sm120_extension(verbose=bool(int(os.environ.get("QUAROT_SM120_VERBOSE", "0"))))
    log(f"[QUAROT_SM120_EXT] {getattr(ext, chr(95)+chr(95)+"file"+chr(95)+chr(95), ext)}")
    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True, local_files_only=local_files_only)
    mt = str(getattr(config, "model_type", "")).lower()
    old_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        if "qwen3" in mt or "qwen" in model_name.lower():
            return load_quarot_qwen_model(config)
        if "llama" in mt or "llama" in model_name.lower():
            return load_quarot_llama_model(config)
        raise RuntimeError(f"Unsupported QuaRot model type: {mt}")
    finally:
        torch.set_default_dtype(old_default_dtype)


def prepare_quarot_latency_layer(layer: nn.Module, device: torch.device):
    for mod in layer.modules():
        ws = getattr(mod, "weight_scales", None)
        if isinstance(ws, torch.Tensor):
            ws = ws.to(device=device, dtype=torch.float16).contiguous()
            with torch.no_grad():
                if torch.count_nonzero(ws).item() == 0:
                    ws.fill_(1.0)
            if "weight_scales" in getattr(mod, "_buffers", {}):
                mod._buffers["weight_scales"] = ws
            else:
                mod.weight_scales = ws
        bias = getattr(mod, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias = bias.to(device=device, dtype=torch.float16).contiguous()
            if "bias" in getattr(mod, "_buffers", {}):
                mod._buffers["bias"] = bias
            else:
                mod.bias = bias
    return layer




def make_pure_policy(policy: dict) -> dict:
    pure = copy.deepcopy(policy)
    for cfg in pure.get("modules", {}).values():
        cfg["ratio"] = 0.0
        cfg["ratio_continuous"] = 0.0
    return pure


def summarize_records(records: List[dict]) -> dict:
    if not records:
        return {"mean_ratio": 0.0, "max_ratio": 0.0, "nonzero_modules": 0, "sum_R": 0}
    ratios = [float(r["ratio"]) for r in records]
    return {
        "mean_ratio": sum(ratios) / len(ratios),
        "max_ratio": max(ratios),
        "nonzero_modules": sum(1 for r in records if float(r["ratio"]) > 0.0),
        "sum_R": sum(int(r["R"]) for r in records),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LABEL] {args.label}")
    log(f"[POLICY] {args.policy}")
    log(f"[ROTATION_CONFIG] {args.rotation_config}")
    log(f"[VARIANTS] {args.variants}")
    log("[NOTE] Complete method paths are preserved; Split dense W4A4 uses QuaRot SM120 CUTLASS backend; q/k/v share Hadamard; fused top-r+pack is used only for K=4096,R<=256 and otherwise falls back to the original prepare path; sparse correction is applied inplace to dense output.")

    V29.install_qfactory_fast_preset(args.qfactory_fast_preset)
    install_qfactory_mixed_fast_preset()
    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)
    policy = V29.load_policy(Path(args.policy), args.force_activation_percentile_100)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()
    layers = V8.get_layers(model)
    hidden_size = V8.infer_hidden_size(model)
    layer_indices = V29.parse_layers(args.layers, len(layers))
    batches = parse_ints(args.batches)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]

    quarot_dense_ext = None
    fused_topr_ext = None
    if "split" in variants or "pure" in variants:
        quarot_dense_ext = load_quarot_sm120_extension(verbose=bool(int(os.environ.get("QUAROT_SM120_VERBOSE", "0"))))
        log(f"[SPLIT_QUAROT_DENSE_EXT] {getattr(quarot_dense_ext, chr(95)+chr(95)+"file"+chr(95)+chr(95), quarot_dense_ext)}")
    if "split" in variants and args.fused_topr_pack:
        fused_topr_ext = load_fused_topr_pack_ext(verbose=bool(int(os.environ.get("FUSED_TOPR_VERBOSE", "0"))))
        log(f"[FUSED_TOPR_PACK_EXT] {getattr(fused_topr_ext, chr(95)+chr(95)+"file"+chr(95)+chr(95), fused_topr_ext)}")

    rot_flags = B.H.build_rotation_flags(model, args.rotation_config)
    log(f"[SPLIT_ROT_FLAGS] selected={sum(bool(v) for v in rot_flags.values())}/{len(rot_flags)}")

    quarot_model = None
    if "quarot" in variants:
        try:
            quarot_model = build_quarot_model(args.model, args.local_files_only)
            log("[QUAROT_MODEL] built")
        except Exception as exc:
            log(f"[QUAROT_MODEL_ERROR] {type(exc).__name__}: {exc}")

    rows = []
    errors = []
    for layer_idx in layer_indices:
        for batch in batches:
            hidden = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=torch.float16)
            position_ids = V8.make_position_ids(batch, args.seq_len, device)
            pe = V8.build_position_embeddings(model, hidden, position_ids, torch.float16)
            row = {
                "model": args.model,
                "model_label": args.label,
                "layer_idx": int(layer_idx),
                "batch": int(batch),
                "seq_len": int(args.seq_len),
                "hidden_size": int(hidden_size),
                "policy_file": str(Path(args.policy)),
                "rotation_config": str(Path(args.rotation_config)),
                "timing": "cuda_graph_events",
                "romeo_mode": "bitweaver_complete_multistream" if args.romeo_multistream else "bitweaver_complete_single_stream",
                "note": "hadamard_on_complete_paths_split_quarot_dense_sharedqkv_inplace_fusedtopr_v43_no_gptq_latency_only",
                "fused_topr_pack": bool(args.fused_topr_pack),
            }

            for variant in variants:
                log(f"[CASE] layer={layer_idx} batch={batch} variant={variant}")
                v_dtype = torch.bfloat16 if variant == "romeo" else torch.float16
                vhidden = hidden if hidden.dtype == v_dtype else hidden.to(v_dtype)
                vpe = V8.build_position_embeddings(model, vhidden, position_ids, v_dtype)
                try:
                    if variant == "pure":
                        pure_policy = make_pure_policy(policy)
                        layer, rec = build_split_layer_with_hadamard(layers[layer_idx], layer_idx, pure_policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, args.eps, device, quarot_dense_ext)
                        row.update({f"pure_{k}": v for k, v in summarize_records(rec).items()})
                        ms = bench_graph(lambda: run_layer_once(layer, vhidden, position_ids, vpe), args.warmup, args.iters, device)
                    elif variant == "split":
                        layer, rec = build_split_layer_with_hadamard(layers[layer_idx], layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, args.eps, device, quarot_dense_ext, fused_topr_ext=fused_topr_ext)
                        row.update({f"split_{k}": v for k, v in summarize_records(rec).items()})
                        ms = bench_graph(lambda: run_layer_once(layer, vhidden, position_ids, vpe), args.warmup, args.iters, device)
                    elif variant == "romeo":
                        layer = build_romeo_layer(layers[layer_idx], layer_idx, args, device)
                        ms = bench_graph(lambda: run_layer_once(layer, vhidden, position_ids, vpe), args.warmup, args.iters, device)
                    elif variant == "quarot":
                        if quarot_model is None:
                            raise RuntimeError("QuaRot model was not built")
                        qlayers = V8.get_layers(quarot_model)
                        qrot = quarot_model.model.rotary_emb.to(device=device, dtype=torch.float16)
                        qlayer = prepare_quarot_latency_layer(qlayers[layer_idx].to(device=device, dtype=torch.float16).eval(), device)
                        qpe = tuple(t.to(dtype=torch.float16) for t in qrot(vhidden, position_ids))
                        ms = bench_graph(lambda: run_layer_once(qlayer, vhidden, position_ids, qpe), args.warmup, args.iters, device)
                        qlayer.to("cpu")
                        layer = qlayer
                    else:
                        raise ValueError(f"unknown variant {variant}")
                    row[f"{variant}_total_ms"] = ms
                    log(f"[TIME] layer={layer_idx} batch={batch} variant={variant} ms={ms:.6f}")
                    del layer
                except Exception as exc:
                    err = {"layer_idx": layer_idx, "batch": batch, "variant": variant, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                    errors.append(err)
                    row[f"{variant}_error"] = err["error"]
                    log("[ERROR] " + json.dumps(err, ensure_ascii=False))
                torch.cuda.synchronize(device)
                gc.collect()
                torch.cuda.empty_cache()

            if row.get("split_total_ms") and row.get("pure_total_ms"):
                row["split_over_pure"] = row["split_total_ms"] / row["pure_total_ms"]
                row["split_minus_pure_ms"] = row["split_total_ms"] - row["pure_total_ms"]
            if row.get("split_total_ms") and row.get("romeo_total_ms"):
                row["split_over_romeo"] = row["split_total_ms"] / row["romeo_total_ms"]
            if row.get("quarot_total_ms") and row.get("romeo_total_ms"):
                row["quarot_over_romeo"] = row["quarot_total_ms"] / row["romeo_total_ms"]
            rows.append(row)

    csv_path = out_dir / f"{args.label}_hadamard_three_schemes_quarot_dense_sharedqkv_inplace_fusedtopr_v43.csv"
    json_path = out_dir / f"{args.label}_hadamard_three_schemes_quarot_dense_sharedqkv_inplace_fusedtopr_v43.json"
    meta_path = out_dir / f"{args.label}_hadamard_three_schemes_fusedtopr_meta_v43.json"
    write_csv(csv_path, rows)
    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)
    json.dump({"args": vars(args), "csv": str(csv_path), "errors": errors}, open(meta_path, "w"), indent=2, ensure_ascii=False)
    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[META] {meta_path}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=0")


if __name__ == "__main__":
    main()
