import argparse
import copy
import csv
import gc
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
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


def build_position_embeddings(model: nn.Module, hidden_states: torch.Tensor, position_ids: torch.Tensor, dtype: torch.dtype):
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb = model.model.rotary_emb.to(hidden_states.device)
        pe = rotary_emb(hidden_states, position_ids)
        return tuple(t.to(device=hidden_states.device, dtype=dtype).contiguous() for t in pe)
    raise RuntimeError("Cannot find model.model.rotary_emb")


def run_layer_once(layer, hidden_states, position_ids, position_embeddings):
    try:
        out = layer(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
    except TypeError:
        out = layer(
            hidden_states,
            position_embeddings=position_embeddings,
        )
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def bench_graph(layer, hidden_states, position_ids, position_embeddings, warmup: int, iters: int):
    device = hidden_states.device

    for _ in range(warmup):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings)
    torch.cuda.synchronize(device)

    static_x = hidden_states.clone()
    static_pos = position_ids.clone()
    static_pe = tuple(t.clone() for t in position_embeddings)

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


def resolve_extensions(B, args, out_dir: Path):
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

    cutlass = BASE.find_cutlass_path(args.cutlass_path)
    log(f"[EXT] cutlass={cutlass}")

    main_ext = BASE.load_ext(cutlass, verbose=False)
    layout_ext = BASE.load_layout_ext(False)
    policy_pack_ext = load_policy_pack_ext(False)

    log(f"[EXT_MAIN] {main_ext}")
    log(f"[EXT_LAYOUT] {layout_ext}")
    log(f"[EXT_POLICY_PACK] {policy_pack_ext}")

    return main_ext, layout_ext, policy_pack_ext


def iter_named_linears(module: nn.Module):
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
    # weight_cpu shape: [N, K]. RealPolicyLinear expects scale numel=N.
    scale = weight_cpu.detach().float().abs().amax(dim=1) / 7.0
    return scale.clamp_min(eps).contiguous()


def patch_layer_with_real_policy(
    *,
    layer: nn.Module,
    B,
    main_ext,
    layout_ext,
    policy_pack_ext,
    mode: str,
    ratio: float,
    eps: float,
    device: torch.device,
):
    BASE = getattr(B, "BASE")
    RealPolicyLinear = getattr(B, "RealPolicyLinear")

    assert mode in {"pure", "dual_policy"}

    split_mode = mode != "pure"
    linears = iter_named_linears(layer)
    if not linears:
        raise RuntimeError("No linears found in layer")

    max_r_by_shape: Dict[Tuple[int, int], int] = {}
    records = []

    for parent, child_name, full_name, linear in linears:
        N, K = map(int, linear.weight.shape)
        local_ratio = float(ratio if split_mode else 0.0)
        R = BASE.ceil_ratio_count(K, local_ratio)
        max_r_by_shape[(K, N)] = max(max_r_by_shape.get((K, N), 0), R)
        records.append({
            "name": full_name,
            "mode": mode,
            "K": K,
            "N": N,
            "ratio": local_ratio,
            "R": R,
        })

    log(f"[PATCH_TARGETS_{mode}] " + json.dumps(records, indent=2))

    scratch_pool = BASE.SharedScratchPool(
        device=device,
        max_r_by_shape=max_r_by_shape,
        split=split_mode,
    )

    for parent, child_name, full_name, linear in linears:
        weight_cpu = linear.weight.detach().cpu().contiguous()
        bias_cpu = None if linear.bias is None else linear.bias.detach().cpu().contiguous()
        scale_cpu = make_rtn_scale_cpu(weight_cpu, eps)

        local_ratio = float(ratio if split_mode else 0.0)
        cfg = {
            "ratio": local_ratio,
            "ratio_continuous": local_ratio,
            "ratio_projected": local_ratio,
            "activation_percentile": 100.0,
            "weight_percentile": 100.0,
        }

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

    bf16_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    int4_shell_dtype = torch.float16

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[BATCHES] {args.batches}")
    log(f"[BF16_DTYPE] {bf16_dtype}")
    log(f"[INT4_SHELL_DTYPE] {int4_shell_dtype}")
    log(f"[SPLIT_RATIO] {args.split_ratio}")
    log("[NOTE] no GPTQ, no calibration; Pure/Split use RTN maxabs per-output weight scale")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = resolve_extensions(B, args, out_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=bf16_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    layers = get_layers(model)
    hidden_size = infer_hidden_size(model)
    if args.layer_idx < 0 or args.layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {args.layer_idx}, num_layers={len(layers)}")

    base_layer = layers[args.layer_idx]

    bf16_layer = copy.deepcopy(base_layer).to(device=device, dtype=bf16_dtype).eval()

    pure_layer = copy.deepcopy(base_layer).to(device=device, dtype=int4_shell_dtype).eval()
    pure_layer, pure_records = patch_layer_with_real_policy(
        layer=pure_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="pure",
        ratio=0.0,
        eps=args.eps,
        device=device,
    )
    pure_layer.to(device=device).eval()

    split_layer = copy.deepcopy(base_layer).to(device=device, dtype=int4_shell_dtype).eval()
    split_layer, split_records = patch_layer_with_real_policy(
        layer=split_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="dual_policy",
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

        hidden_bf16 = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=bf16_dtype)
        hidden_int4 = hidden_bf16.to(int4_shell_dtype).contiguous()

        position_ids = make_position_ids(batch, args.seq_len, device)

        pe_bf16 = build_position_embeddings(model, hidden_bf16, position_ids, bf16_dtype)
        pe_int4 = build_position_embeddings(model, hidden_int4, position_ids, int4_shell_dtype)

        log("[POSITION_EMBEDDINGS_BF16] " + str([(tuple(t.shape), str(t.dtype)) for t in pe_bf16]))
        log("[POSITION_EMBEDDINGS_INT4] " + str([(tuple(t.shape), str(t.dtype)) for t in pe_int4]))

        torch.cuda.empty_cache()

        bf16_ms = bench_graph(bf16_layer, hidden_bf16, position_ids, pe_bf16, args.warmup, args.iters)
        pure_ms = bench_graph(pure_layer, hidden_int4, position_ids, pe_int4, args.warmup, args.iters)
        split_ms = bench_graph(split_layer, hidden_int4, position_ids, pe_int4, args.warmup, args.iters)

        row = {
            "model": args.model,
            "layer_idx": args.layer_idx,
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            "bf16_ms": bf16_ms,
            "pure_w4a4_ms": pure_ms,
            "split_w4a4_ms": split_ms,
            "pure_over_bf16": pure_ms / bf16_ms,
            "split_over_bf16": split_ms / bf16_ms,
            "split_over_pure": split_ms / pure_ms,
            "bf16_speedup_over_pure": bf16_ms / pure_ms,
            "bf16_speedup_over_split": bf16_ms / split_ms,
            "split_ratio": args.split_ratio,
            "weight_scale_mode": "rtn_maxabs_per_output_no_gptq",
            "bf16_dtype": str(bf16_dtype),
            "int4_shell_dtype": str(int4_shell_dtype),
            "timing": "cuda_graph",
        }
        rows.append(row)
        log("[RESULT] " + json.dumps(row, indent=2))

    csv_path = out_dir / "bf16_pure_split_no_gptq_layer_latency_v8.csv"
    json_path = out_dir / "bf16_pure_split_no_gptq_layer_latency_v8.json"
    pure_records_path = out_dir / "pure_patch_records_v8.json"
    split_records_path = out_dir / "split_patch_records_v8.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)
    json.dump(pure_records, open(pure_records_path, "w"), indent=2, ensure_ascii=False)
    json.dump(split_records, open(split_records_path, "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[PURE_RECORDS] {pure_records_path}")
    log(f"[SPLIT_RECORDS] {split_records_path}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
