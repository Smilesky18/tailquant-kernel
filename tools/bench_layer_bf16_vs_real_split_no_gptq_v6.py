import argparse
import copy
import csv
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--split_ratio", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--timing", choices=["graph", "eager", "both"], default="graph")
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/cutlass"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers.")


def infer_hidden_size(model: nn.Module):
    cfg = model.config
    for key in ["hidden_size", "n_embd", "d_model"]:
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    raise RuntimeError("Cannot infer hidden size.")


def make_position_ids(batch: int, seq_len: int, device: torch.device):
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1).contiguous()


def build_position_embeddings(model: nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb = model.model.rotary_emb.to(hidden_states.device)
        return rotary_emb(hidden_states, position_ids)

    layers = get_layers(model)
    attn = layers[0].self_attn if hasattr(layers[0], "self_attn") else None
    if attn is not None and hasattr(attn, "rotary_emb"):
        rotary_emb = attn.rotary_emb.to(hidden_states.device)
        return rotary_emb(hidden_states, position_ids)

    return None


def run_layer_once(layer, hidden_states, position_ids=None, position_embeddings=None):
    kwargs = {}
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    try:
        out = layer(hidden_states, **kwargs)
    except TypeError:
        kwargs.pop("position_ids", None)
        out = layer(hidden_states, **kwargs)

    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def bench_eager(layer, hidden_states, position_ids, position_embeddings, warmup: int, iters: int):
    device = hidden_states.device
    for _ in range(warmup):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings)
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


@torch.no_grad()
def bench_graph(layer, hidden_states, position_ids, position_embeddings, warmup: int, iters: int):
    device = hidden_states.device

    # 先 eager warmup，触发 lazy init / workspace init / JIT。
    for _ in range(warmup):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings)
    torch.cuda.synchronize(device)

    static_x = hidden_states.clone()
    static_pos = position_ids.clone() if position_ids is not None else None
    static_pe = None if position_embeddings is None else tuple(t.clone() for t in position_embeddings)

    for _ in range(warmup):
        _ = run_layer_once(layer, static_x, static_pos, static_pe)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _ = run_layer_once(layer, static_x, static_pos, static_pe)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


def dump_debug(out_dir: Path, name: str, obj: Any):
    path = out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        json.dump(obj, open(path, "w"), indent=2, ensure_ascii=False)
    except TypeError:
        path.write_text(str(obj))
    log(f"[DEBUG_DUMP] {path}")


def normalize_ext_tuple(ret):
    """
    尽量从各种返回格式中抽取 main_ext/layout_ext/policy_pack_ext。
    """
    main_ext = layout_ext = policy_pack_ext = None

    def inspect_obj(k, v):
        nonlocal main_ext, layout_ext, policy_pack_ext
        attrs = set(dir(v))
        if "pack_weight_rowmajor_s4_from_scale" in attrs:
            main_ext = v
        if "row_to_col_s4_tiled" in attrs:
            layout_ext = v
        # policy_pack_ext 的函数名不同版本可能不一样，先用弱匹配。
        low_attrs = " ".join(a.lower() for a in attrs)
        if ("policy" in low_attrs or "pack" in low_attrs) and (
            "ratio" in low_attrs or "gather" in low_attrs or "scatter" in low_attrs or "top" in low_attrs
        ):
            if v is not main_ext and v is not layout_ext:
                policy_pack_ext = v

    if isinstance(ret, dict):
        for k, v in ret.items():
            lk = str(k).lower()
            if "main" in lk and "ext" in lk:
                main_ext = v
            elif "layout" in lk and "ext" in lk:
                layout_ext = v
            elif "policy" in lk and "ext" in lk:
                policy_pack_ext = v
            inspect_obj(k, v)

    elif isinstance(ret, (tuple, list)):
        for i, v in enumerate(ret):
            inspect_obj(i, v)
        # 常见返回顺序兜底
        if len(ret) >= 3 and main_ext is None and layout_ext is None and policy_pack_ext is None:
            main_ext, layout_ext, policy_pack_ext = ret[0], ret[1], ret[2]

    else:
        inspect_obj("ret", ret)

    if main_ext is not None and layout_ext is not None and policy_pack_ext is not None:
        return main_ext, layout_ext, policy_pack_ext
    return None



def resolve_extensions(B, args, out_dir: Path):
    """
    v6: 不再猜测 extension loader，直接复用 bench_real_split_fullstack_v1.py main() 中的真实加载路径：

        cutlass = BASE.find_cutlass_path(cli.cutlass_path)
        main_ext = BASE.load_ext(cutlass, verbose=cli.verbose_compile)
        layout_ext = BASE.load_layout_ext(cli.verbose_compile)
        policy_pack_ext = load_policy_pack_ext(cli.verbose_compile)

    这样绕开 v5 的自动探测失败问题。
    """
    import os
    import json
    import torch
    from pathlib import Path

    BASE = getattr(B, "BASE", None)
    if BASE is None:
        raise RuntimeError("bench_real_split_fullstack_v1 has no BASE")

    load_policy_pack_ext = getattr(B, "load_policy_pack_ext", None)
    if load_policy_pack_ext is None:
        raise RuntimeError("bench_real_split_fullstack_v1 has no load_policy_pack_ext")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability(device)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        log(f"[EXT] set TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

    debug = {
        "cutlass_path_arg": args.cutlass_path,
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "BASE": repr(BASE),
        "has_find_cutlass_path": hasattr(BASE, "find_cutlass_path"),
        "has_load_ext": hasattr(BASE, "load_ext"),
        "has_load_layout_ext": hasattr(BASE, "load_layout_ext"),
        "has_load_policy_pack_ext": load_policy_pack_ext is not None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extension_loader_debug_v6.json").write_text(
        json.dumps(debug, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    log("[EXT] resolving cutlass path")
    cutlass = BASE.find_cutlass_path(args.cutlass_path)
    log(f"[EXT] cutlass={cutlass}")

    log("[EXT] loading main_ext via BASE.load_ext")
    main_ext = BASE.load_ext(cutlass, verbose=False)
    log(f"[EXT_MAIN] {main_ext}")

    log("[EXT] loading layout_ext via BASE.load_layout_ext")
    layout_ext = BASE.load_layout_ext(False)
    log(f"[EXT_LAYOUT] {layout_ext}")

    log("[EXT] loading policy_pack_ext via load_policy_pack_ext")
    policy_pack_ext = load_policy_pack_ext(False)
    log(f"[EXT_POLICY_PACK] {policy_pack_ext}")

    # 基础功能检查，提前暴露问题。
    missing = []
    if not hasattr(main_ext, "pack_weight_rowmajor_s4_from_scale"):
        missing.append("main_ext.pack_weight_rowmajor_s4_from_scale")
    if not hasattr(layout_ext, "row_to_col_s4_tiled"):
        missing.append("layout_ext.row_to_col_s4_tiled")
    if missing:
        raise RuntimeError("Missing required extension functions: " + ", ".join(missing))

    return main_ext, layout_ext, policy_pack_ext


def iter_named_linears(module: nn.Module):
    """
    遍历所有 Linear 或 ActQuantWrapper.module Linear。
    返回 parent, child_name, full_name, linear。
    """
    out = []

    def rec(parent: nn.Module, prefix: str):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            linear = None
            if isinstance(child, nn.Linear):
                linear = child
            else:
                inner = getattr(child, "module", None)
                if isinstance(inner, nn.Linear):
                    linear = inner

            if linear is not None:
                out.append((parent, name, full, linear))
            else:
                rec(child, full)

    rec(module, "")
    return out


def make_rtn_scale_cpu(weight_cpu: torch.Tensor, eps: float):
    # weight_cpu: [N, K]，RealPolicyLinear 需要 scale numel=N。
    scale = weight_cpu.detach().float().abs().amax(dim=1) / 7.0
    return scale.clamp_min(eps).contiguous()


def patch_layer_with_real_split(layer: nn.Module, B, main_ext, layout_ext, policy_pack_ext, ratio: float, eps: float, device: torch.device):
    BASE = getattr(B, "BASE")
    RealPolicyLinear = getattr(B, "RealPolicyLinear")

    linears = iter_named_linears(layer)
    if not linears:
        raise RuntimeError("No linears found in layer")

    max_r_by_shape: Dict[Tuple[int, int], int] = {}
    records = []
    for parent, child_name, full_name, linear in linears:
        N, K = map(int, linear.weight.shape)
        R = BASE.ceil_ratio_count(K, ratio)
        if R <= 0:
            raise RuntimeError(f"{full_name}: ratio={ratio} gives R={R}")
        max_r_by_shape[(K, N)] = max(max_r_by_shape.get((K, N), 0), R)
        records.append({"name": full_name, "K": K, "N": N, "R": R})

    log("[SPLIT_PATCH_TARGETS] " + json.dumps(records, indent=2))

    scratch_pool = BASE.SharedScratchPool(
        device=device,
        max_r_by_shape=max_r_by_shape,
        split=True,
    )

    for parent, child_name, full_name, linear in linears:
        weight_cpu = linear.weight.detach().cpu().contiguous()
        bias_cpu = None if linear.bias is None else linear.bias.detach().cpu().contiguous()
        scale_cpu = make_rtn_scale_cpu(weight_cpu, eps)

        cfg = {
            "ratio": float(ratio),
            "ratio_continuous": float(ratio),
            "activation_percentile": 100.0,
            "weight_percentile": 100.0,
        }

        repl = RealPolicyLinear(
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            mode="dual_policy",
            weight_cpu=weight_cpu,
            bias_cpu=bias_cpu,
            policy_cfg=cfg,
            gptq_scale_cpu=scale_cpu,
            eps=eps,
            device=device,
            name=full_name,
            scratch_pool=scratch_pool,
            prefetch_workspace=None,
            rotate_online=False,
            had_k=None,
            had_factor=1,
        )
        setattr(parent, child_name, repl)

    return layer, records


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[BATCHES] {args.batches}")
    log(f"[DTYPE] {dtype}")
    log(f"[DEVICE] {device}")
    log(f"[TIMING] {args.timing}")
    log(f"[SPLIT_RATIO] {args.split_ratio}")
    log("[NOTE] no GPTQ, no calibration, RTN/maxabs per-output weight scale for latency only")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

    # 解析 real split extensions。
    main_ext, layout_ext, policy_pack_ext = resolve_extensions(B, args, out_dir)
    log(f"[EXT_MAIN] {main_ext}")
    log(f"[EXT_LAYOUT] {layout_ext}")
    log(f"[EXT_POLICY_PACK] {policy_pack_ext}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    layers = get_layers(model)
    hidden_size = infer_hidden_size(model)

    if args.layer_idx < 0 or args.layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {args.layer_idx}, num_layers={len(layers)}")

    bf16_layer = copy.deepcopy(layers[args.layer_idx]).to(device=device, dtype=dtype).eval()
    split_layer = copy.deepcopy(layers[args.layer_idx]).to(device=device, dtype=dtype).eval()

    split_layer, split_records = patch_layer_with_real_split(
        split_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        ratio=args.split_ratio,
        eps=args.eps,
        device=device,
    )
    split_layer.to(device=device).eval()

    log(f"[HIDDEN_SIZE] {hidden_size}")
    log(f"[NUM_LAYERS] {len(layers)}")
    log(f"[BF16_LAYER_FORWARD_SIGNATURE] {inspect.signature(bf16_layer.forward)}")

    rows = []
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    for batch in batches:
        log(f"\n[CASE] batch={batch} seq_len={args.seq_len}")

        hidden_states = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=dtype)
        position_ids = make_position_ids(batch, args.seq_len, device)
        position_embeddings = build_position_embeddings(model, hidden_states, position_ids)
        if position_embeddings is None:
            raise RuntimeError("position_embeddings is None")
        log("[POSITION_EMBEDDINGS] " + str([tuple(t.shape) for t in position_embeddings]))

        torch.cuda.empty_cache()

        results = {}

        if args.timing in {"eager", "both"}:
            bf16_eager_ms = bench_eager(bf16_layer, hidden_states, position_ids, position_embeddings, args.warmup, args.iters)
            split_eager_ms = bench_eager(split_layer, hidden_states, position_ids, position_embeddings, args.warmup, args.iters)
            results["bf16_eager_ms"] = bf16_eager_ms
            results["split_eager_ms"] = split_eager_ms

        if args.timing in {"graph", "both"}:
            bf16_graph_ms = bench_graph(bf16_layer, hidden_states, position_ids, position_embeddings, args.warmup, args.iters)
            try:
                split_graph_ms = bench_graph(split_layer, hidden_states, position_ids, position_embeddings, args.warmup, args.iters)
            except Exception as e:
                log(f"[WARN] split CUDA graph timing failed, fallback eager. error={e!r}")
                split_graph_ms = None
                split_eager_fallback_ms = bench_eager(split_layer, hidden_states, position_ids, position_embeddings, args.warmup, args.iters)
                results["split_eager_fallback_ms"] = split_eager_fallback_ms

            results["bf16_graph_ms"] = bf16_graph_ms
            results["split_graph_ms"] = split_graph_ms

        bf16_ms = results.get("bf16_graph_ms", results.get("bf16_eager_ms"))
        split_ms = results.get("split_graph_ms", results.get("split_eager_ms", results.get("split_eager_fallback_ms")))

        row = {
            "model": args.model,
            "layer_idx": args.layer_idx,
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            "split_ratio": args.split_ratio,
            "weight_scale_mode": "rtn_maxabs_per_output_no_gptq",
            "timing": args.timing,
            "bf16_ms": bf16_ms,
            "split_ms": split_ms,
            "speedup_bf16_over_split": (bf16_ms / split_ms) if (bf16_ms and split_ms) else None,
            "normalized_split_latency": (split_ms / bf16_ms) if (bf16_ms and split_ms) else None,
            **results,
        }
        rows.append(row)
        log("[RESULT] " + json.dumps(row, indent=2))

    csv_path = out_dir / "bf16_vs_split_no_gptq_layer_latency_v5.csv"
    json_path = out_dir / "bf16_vs_split_no_gptq_layer_latency_v5.json"
    records_path = out_dir / "split_patch_records_v5.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)
    json.dump(split_records, open(records_path, "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[SPLIT_RECORDS] {records_path}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
