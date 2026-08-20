import argparse
import copy
import csv
import gc
import json
import os
import time
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8
from bench_layer_qfactory_raw_backend_v19 import patch_qfactory_raw_backend


TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--layers", default="all")
    p.add_argument(
        "--variants",
        default=(
            "bf16,pure_current,romeo_qfactory,"
            "split_fixed_qfactory,split_policy_qfactory"
        ),
    )
    p.add_argument("--fixed_ratio", type=float, default=0.05)
    p.add_argument("--policy", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--cutlass_path",
        default=os.environ.get(
            "CUTLASS_PATH",
            "/data/yzy/quarot-gpt-2/third_party/cutlass",
        ),
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--force_activation_percentile_100", action="store_true")
    return p.parse_args()


def parse_csv_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def parse_layers(text: str, num_layers: int) -> List[int]:
    if text == "all":
        return list(range(num_layers))
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return [x for x in out if 0 <= x < num_layers]


def write_csv(path: Path, rows: List[dict]):
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_name(name: str) -> str:
    if name.startswith("model."):
        return name
    if name.startswith("layers."):
        return f"model.{name}"
    return name


def load_policy(path: Path, force_activation_percentile_100: bool) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    modules = {}
    for name, item in raw.get("modules", {}).items():
        key = normalize_name(name)
        ratio = float(
            item.get(
                "ratio_projected",
                item.get("ratio", item.get("ratio_continuous", 0.0)),
            )
        )
        ratio_cont = float(item.get("ratio_continuous", ratio))
        activation_percentile = float(item.get("activation_percentile", 100.0))
        if force_activation_percentile_100:
            activation_percentile = 100.0
        modules[key] = {
            "ratio": ratio,
            "ratio_continuous": ratio_cont,
            "ratio_projected": ratio,
            "activation_percentile": activation_percentile,
            "weight_percentile": float(item.get("weight_percentile", 100.0)),
        }
    return {
        "path": str(path),
        "summary": raw.get("summary", {}),
        "metadata": raw.get("metadata", {}),
        "modules": modules,
    }


def iter_target_linears(layer: nn.Module):
    found = []

    def rec(parent: nn.Module, prefix: str):
        for child_name, child in list(parent.named_children()):
            full = f"{prefix}.{child_name}" if prefix else child_name
            linear = None
            if isinstance(child, nn.Linear):
                linear = child
            else:
                inner = getattr(child, "module", None)
                if isinstance(inner, nn.Linear):
                    linear = inner
            if linear is not None:
                if full in TARGET_SUFFIXES:
                    found.append((parent, child_name, full, linear))
            else:
                rec(child, full)

    rec(layer, "")
    missing = sorted(set(TARGET_SUFFIXES) - {x[2] for x in found})
    if missing:
        raise RuntimeError(f"Layer target Linear modules missing: {missing}")
    return found


def make_rtn_scale_cpu(weight_cpu: torch.Tensor, eps: float):
    scale = weight_cpu.detach().float().abs().amax(dim=1) / 7.0
    return scale.clamp_min(eps).contiguous()


def policy_key(layer_idx: int, local_name: str) -> str:
    return f"model.layers.{layer_idx}.{local_name}"


def fixed_cfg(ratio: float) -> dict:
    return {
        "ratio": float(ratio),
        "ratio_continuous": float(ratio),
        "ratio_projected": float(ratio),
        "activation_percentile": 100.0,
        "weight_percentile": 100.0,
    }


def patch_layer_with_cfgs(
    *,
    layer: nn.Module,
    layer_idx: int,
    B,
    main_ext,
    layout_ext,
    policy_pack_ext,
    cfg_by_local_name: Dict[str, dict],
    eps: float,
    device: torch.device,
):
    BASE = getattr(B, "BASE")
    RealPolicyLinear = getattr(B, "RealPolicyLinear")
    linears = iter_target_linears(layer)

    pure_shapes: Dict[Tuple[int, int], int] = {}
    split_shapes: Dict[Tuple[int, int], int] = {}
    records = []

    for _, _, local_name, linear in linears:
        cfg = cfg_by_local_name[local_name]
        N, K = map(int, linear.weight.shape)
        ratio = float(cfg["ratio"])
        R = BASE.ceil_ratio_count(K, ratio)
        if R > 0:
            split_shapes[(K, N)] = max(split_shapes.get((K, N), 0), R)
        else:
            pure_shapes[(K, N)] = max(pure_shapes.get((K, N), 0), 0)
        records.append(
            {
                "name": policy_key(layer_idx, local_name),
                "local_name": local_name,
                "K": K,
                "N": N,
                "ratio": ratio,
                "ratio_continuous": float(cfg["ratio_continuous"]),
                "R": int(R),
                "activation_percentile": float(cfg["activation_percentile"]),
                "weight_percentile": float(cfg["weight_percentile"]),
                "mode": "dual_policy" if R > 0 else "pure",
            }
        )

    pure_pool = None
    split_pool = None
    if pure_shapes:
        pure_pool = BASE.SharedScratchPool(
            device=device,
            max_r_by_shape=pure_shapes,
            split=False,
        )
    if split_shapes:
        split_pool = BASE.SharedScratchPool(
            device=device,
            max_r_by_shape=split_shapes,
            split=True,
        )

    for parent, child_name, local_name, linear in linears:
        cfg = cfg_by_local_name[local_name]
        weight_cpu = linear.weight.detach().cpu().contiguous()
        bias_cpu = None if linear.bias is None else linear.bias.detach().cpu().contiguous()
        scale_cpu = make_rtn_scale_cpu(weight_cpu, eps)
        ratio = float(cfg["ratio"])
        R = BASE.ceil_ratio_count(int(linear.in_features), ratio)
        mode = "dual_policy" if R > 0 else "pure"
        scratch_pool = split_pool if mode == "dual_policy" else pure_pool
        if scratch_pool is None:
            raise RuntimeError(f"No scratch pool for {mode}")

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
            name=policy_key(layer_idx, local_name),
            scratch_pool=scratch_pool,
            prefetch_workspace=None,
            rotate_online=False,
            had_k=None,
            had_factor=1,
        )
        setattr(parent, child_name, repl)

    return layer, records


def build_variant_layer(
    *,
    variant: str,
    base_layer: nn.Module,
    layer_idx: int,
    policy: dict,
    fixed_ratio: float,
    B,
    main_ext,
    layout_ext,
    policy_pack_ext,
    eps: float,
    device: torch.device,
    bf16_dtype: torch.dtype,
):
    if variant == "bf16":
        return copy.deepcopy(base_layer).to(device=device, dtype=bf16_dtype).eval(), []

    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()

    if variant in {"pure_current", "romeo_qfactory"}:
        cfg_by_local = {name: fixed_cfg(0.0) for name in TARGET_SUFFIXES}
    elif variant in {"split_fixed_current", "split_fixed_qfactory"}:
        cfg_by_local = {name: fixed_cfg(fixed_ratio) for name in TARGET_SUFFIXES}
    elif variant in {"split_policy_current", "split_policy_qfactory"}:
        cfg_by_local = {}
        for name in TARGET_SUFFIXES:
            key = policy_key(layer_idx, name)
            if key not in policy["modules"]:
                raise KeyError(f"Policy missing {key}")
            cfg_by_local[name] = policy["modules"][key]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    layer, records = patch_layer_with_cfgs(
        layer=layer,
        layer_idx=layer_idx,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        cfg_by_local_name=cfg_by_local,
        eps=eps,
        device=device,
    )

    if variant in {
        "romeo_qfactory",
        "split_fixed_qfactory",
        "split_policy_qfactory",
    }:
        patch_qfactory_raw_backend(layer)

    return layer.to(device=device).eval(), records


def run_layer_once(layer, hidden_states, position_ids, position_embeddings):
    out = layer(
        hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def bench_graph(fn, warmup: int, iters: int, device: torch.device):
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


def summarize_records(records: List[dict]) -> dict:
    if not records:
        return {
            "mean_ratio": 0.0,
            "max_ratio": 0.0,
            "nonzero_modules": 0,
            "sum_R": 0,
        }
    ratios = [float(r["ratio"]) for r in records]
    return {
        "mean_ratio": float(sum(ratios) / len(ratios)),
        "max_ratio": float(max(ratios)),
        "nonzero_modules": int(sum(1 for r in records if float(r["ratio"]) > 0.0)),
        "sum_R": int(sum(int(r["R"]) for r in records)),
    }


def add_ratios(row: dict):
    bf16 = row.get("bf16_ms")
    pure = row.get("pure_current_ms")
    romeo = row.get("romeo_qfactory_ms")
    split_fixed = row.get("split_fixed_qfactory_ms")
    split_policy = row.get("split_policy_qfactory_ms")
    if bf16 and pure:
        row["pure_current_over_bf16"] = pure / bf16
    if bf16 and romeo:
        row["romeo_qfactory_over_bf16"] = romeo / bf16
    if bf16 and split_policy:
        row["split_policy_qfactory_over_bf16"] = split_policy / bf16
    if pure and romeo:
        row["romeo_qfactory_over_pure_current"] = romeo / pure
    if split_fixed and split_policy:
        row["split_policy_over_split_fixed_qfactory"] = split_policy / split_fixed
        row["split_policy_speedup_vs_fixed_qfactory"] = split_fixed / split_policy
    if romeo and split_policy:
        row["split_policy_over_romeo_qfactory"] = split_policy / romeo


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[BATCHES] {args.batches}")
    log(f"[VARIANTS] {args.variants}")
    log(f"[POLICY] {args.policy}")
    log(f"[FIXED_RATIO] {args.fixed_ratio}")
    log(f"[QFACTORY_ARCH] {os.environ.get('QFACTORY_ARCH')}")
    log(f"[QFACTORY_CACHE_DIR] {os.environ.get('QFACTORY_CACHE_DIR')}")
    log("[NOTE] layer-level latency only; RTN maxabs weight scale; no GPTQ/calibration/PPL.")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)
    policy = load_policy(
        Path(args.policy),
        force_activation_percentile_100=args.force_activation_percentile_100,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()

    layers = V8.get_layers(model)
    layer_ids = parse_layers(args.layers, len(layers))
    batches = parse_csv_ints(args.batches)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    hidden_size = V8.infer_hidden_size(model)

    log(f"[NUM_LAYERS] {len(layers)}")
    log(f"[LAYER_IDS] {layer_ids}")
    log(f"[HIDDEN_SIZE] {hidden_size}")
    log("[POLICY_SUMMARY] " + json.dumps(policy["summary"], ensure_ascii=False))

    rows = []
    policy_rows = []
    patch_records_by_variant = {}

    for layer_idx in layer_ids:
        base_layer = layers[layer_idx]
        log(f"\n[LAYER_BEGIN] {layer_idx}")

        variant_times: Dict[str, Dict[int, float]] = {v: {} for v in variants}
        variant_summaries: Dict[str, dict] = {}

        for variant in variants:
            log(f"[BUILD_VARIANT] layer={layer_idx} variant={variant}")
            layer, records = build_variant_layer(
                variant=variant,
                base_layer=base_layer,
                layer_idx=layer_idx,
                policy=policy,
                fixed_ratio=args.fixed_ratio,
                B=B,
                main_ext=main_ext,
                layout_ext=layout_ext,
                policy_pack_ext=policy_pack_ext,
                eps=args.eps,
                device=device,
                bf16_dtype=torch.bfloat16,
            )
            patch_records_by_variant[f"layer{layer_idx}_{variant}"] = records
            variant_summaries[variant] = summarize_records(records)

            for batch in batches:
                dtype = torch.bfloat16 if variant == "bf16" else torch.float16
                hidden = torch.randn(
                    batch,
                    args.seq_len,
                    hidden_size,
                    device=device,
                    dtype=dtype,
                )
                position_ids = V8.make_position_ids(batch, args.seq_len, device)
                pe = V8.build_position_embeddings(model, hidden, position_ids, dtype)

                log(f"[TIME_BEGIN] layer={layer_idx} variant={variant} batch={batch}")
                ms = bench_graph(
                    lambda: run_layer_once(layer, hidden, position_ids, pe),
                    args.warmup,
                    args.iters,
                    device,
                )
                variant_times[variant][batch] = ms
                log(f"[TIME] layer={layer_idx} variant={variant} batch={batch} {ms:.6f} ms")

                del hidden, position_ids, pe
                torch.cuda.empty_cache()

            del layer
            gc.collect()
            torch.cuda.empty_cache()

        for batch in batches:
            row = {
                "model": args.model,
                "layer_idx": layer_idx,
                "batch": batch,
                "seq_len": args.seq_len,
                "hidden_size": hidden_size,
                "timing": "cuda_graph_events",
                "weight_scale_mode": "rtn_maxabs",
                "policy_file": policy["path"],
                "note": "no_gptq_no_calibration_latency_only",
            }
            for variant in variants:
                row[f"{variant}_ms"] = variant_times[variant][batch]
                for key, value in variant_summaries[variant].items():
                    row[f"{variant}_{key}"] = value
            add_ratios(row)
            rows.append(row)
            log("[ROW] " + json.dumps(row, ensure_ascii=False))

        if "split_policy_qfactory" in variant_summaries:
            summary = variant_summaries["split_policy_qfactory"]
            policy_rows.append(
                {
                    "model": args.model,
                    "layer_idx": layer_idx,
                    "policy_file": policy["path"],
                    **summary,
                }
            )

        partial_csv = out_dir / "qwen3_8b_all_layers_policy_v23_partial.csv"
        partial_json = out_dir / "qwen3_8b_all_layers_policy_v23_partial.json"
        write_csv(partial_csv, rows)
        partial_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        log(f"[PARTIAL] {partial_csv}")

    csv_path = out_dir / "qwen3_8b_all_layers_policy_v23.csv"
    json_path = out_dir / "qwen3_8b_all_layers_policy_v23.json"
    policy_csv = out_dir / "qwen3_8b_all_layers_policy_summary_v23.csv"
    meta_path = out_dir / "qwen3_8b_all_layers_policy_v23_meta.json"

    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(policy_csv, policy_rows)
    meta_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "policy_summary": policy["summary"],
                "policy_metadata": policy["metadata"],
                "variants": variants,
                "patch_records_by_variant": patch_records_by_variant,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[POLICY_CSV] {policy_csv}")
    log(f"[META] {meta_path}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
