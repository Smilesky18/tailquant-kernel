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
from qlinear import MixedRQLinear


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


def build_split_layer_with_hadamard(base_layer, layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, eps, device):
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

    QF19.patch_qfactory_raw_backend(layer)
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
    log("[NOTE] Complete method paths are preserved; Split/RoMeo share qfactory W4A4 dense backend where applicable.")

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
                "note": "hadamard_on_complete_paths_no_gptq_latency_only",
            }

            for variant in variants:
                log(f"[CASE] layer={layer_idx} batch={batch} variant={variant}")
                v_dtype = torch.bfloat16 if variant == "romeo" else torch.float16
                vhidden = hidden if hidden.dtype == v_dtype else hidden.to(v_dtype)
                vpe = V8.build_position_embeddings(model, vhidden, position_ids, v_dtype)
                try:
                    if variant == "split":
                        layer, rec = build_split_layer_with_hadamard(layers[layer_idx], layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, args.eps, device)
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

            if row.get("split_total_ms") and row.get("romeo_total_ms"):
                row["split_over_romeo"] = row["split_total_ms"] / row["romeo_total_ms"]
            if row.get("quarot_total_ms") and row.get("romeo_total_ms"):
                row["quarot_over_romeo"] = row["quarot_total_ms"] / row["romeo_total_ms"]
            rows.append(row)

    csv_path = out_dir / f"{args.label}_hadamard_three_schemes_v35.csv"
    json_path = out_dir / f"{args.label}_hadamard_three_schemes_v35.json"
    meta_path = out_dir / f"{args.label}_hadamard_three_schemes_meta_v35.json"
    write_csv(csv_path, rows)
    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)
    json.dump({"args": vars(args), "csv": str(csv_path), "errors": errors}, open(meta_path, "w"), indent=2, ensure_ascii=False)
    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[META] {meta_path}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=0")


if __name__ == "__main__":
    main()
