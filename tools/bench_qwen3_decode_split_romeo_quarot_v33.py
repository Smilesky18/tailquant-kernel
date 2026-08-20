#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8
import bench_multimodel_all_layers_policy_fastqf_v29 as V29


class OneLayerStaticCache:
    def __init__(self, key_cache, value_cache):
        self.key_cache = key_cache
        self.value_cache = value_cache

    def update(self, key_states, value_states, layer_idx=None):
        self.key_cache[:, :, -1:, :].copy_(key_states)
        self.value_cache[:, :, -1:, :].copy_(value_states)
        return self.key_cache, self.value_cache


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--past_len", type=int, default=128)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--layers", default="0,19")
    p.add_argument("--variants", default="bf16,quarot_current,romeo_qfactory,split_policy_qfactory")
    p.add_argument("--policy", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--fixed_ratio", type=float, default=0.05)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--force_activation_percentile_100", action="store_true")
    p.add_argument("--qfactory_fast_preset", default="qwen3_sm120_v1", choices=["none", "qwen3_sm120_v1"])
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    return p.parse_args()


def write_csv(path: Path, rows):
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def variant_to_build_name(variant: str):
    return "pure_current" if variant == "quarot_current" else variant


def run_decode_once(layer, hidden, position_ids, position_embeddings, cache):
    out = layer(
        hidden,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        past_key_values=cache,
        use_cache=True,
    )
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def bench_graph(fn, warmup, iters, device):
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


def make_decode_inputs(model, layer, batch, past_len, hidden_size, dtype, device):
    hidden = torch.randn(batch, 1, hidden_size, device=device, dtype=dtype)
    position_ids = torch.full((batch, 1), past_len, device=device, dtype=torch.long)
    pe = V8.build_position_embeddings(model, hidden, position_ids, dtype)
    head_dim = int(layer.self_attn.head_dim)
    num_kv_heads = int(getattr(model.config, "num_key_value_heads", getattr(layer.self_attn, "num_key_value_heads", 0)))
    if num_kv_heads <= 0:
        raise RuntimeError("cannot infer num_key_value_heads for decode cache")
    key_cache = torch.randn(batch, num_kv_heads, past_len + 1, head_dim, device=device, dtype=dtype)
    value_cache = torch.randn_like(key_cache)
    cache = OneLayerStaticCache(key_cache, value_cache)
    return hidden, position_ids, pe, cache


def add_ratios(row):
    bf16 = row.get("bf16_ms")
    quarot = row.get("quarot_current_ms")
    romeo = row.get("romeo_qfactory_ms")
    split = row.get("split_policy_qfactory_ms")
    if bf16 and quarot:
        row["quarot_over_bf16"] = quarot / bf16
    if bf16 and romeo:
        row["romeo_over_bf16"] = romeo / bf16
    if bf16 and split:
        row["split_over_bf16"] = split / bf16
    if romeo and split:
        row["split_over_romeo"] = split / romeo
        row["split_extra_vs_romeo_ms"] = split - romeo
        row["split_extra_over_romeo"] = (split - romeo) / romeo


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[WORKLOAD] decode q_len=1 past_len={args.past_len}")
    log(f"[BATCHES] {args.batches}")
    log(f"[LAYERS] {args.layers}")
    log(f"[VARIANTS] {args.variants}")
    log(f"[POLICY] {args.policy}")
    log(f"[QFACTORY_ARCH] {os.environ.get('QFACTORY_ARCH')}")
    log(f"[QFACTORY_CACHE_DIR] {os.environ.get('QFACTORY_CACHE_DIR')}")
    V29.install_qfactory_fast_preset(args.qfactory_fast_preset)

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)
    policy = V29.load_policy(Path(args.policy), args.force_activation_percentile_100)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()
    layers = V8.get_layers(model)
    layer_ids = V29.parse_layers(args.layers, len(layers))
    batches = V29.parse_csv_ints(args.batches)
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    hidden_size = V8.infer_hidden_size(model)

    rows = []
    for layer_idx in layer_ids:
        base_layer = layers[layer_idx]
        log(f"\n[LAYER_BEGIN] {layer_idx}")
        variant_times = {v: {} for v in variants}
        variant_summaries = {}
        for variant in variants:
            build_name = variant_to_build_name(variant)
            log(f"[BUILD_VARIANT] layer={layer_idx} variant={variant} build={build_name}")
            layer, records = V29.build_variant_layer(
                variant=build_name,
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
            variant_summaries[variant] = V29.summarize_records(records)
            for batch in batches:
                dtype = torch.bfloat16 if variant == "bf16" else torch.float16
                hidden, position_ids, pe, cache = make_decode_inputs(
                    model, layer, batch, args.past_len, hidden_size, dtype, device
                )
                log(f"[TIME_BEGIN] layer={layer_idx} variant={variant} batch={batch}")
                ms = bench_graph(
                    lambda: run_decode_once(layer, hidden, position_ids, pe, cache),
                    args.warmup,
                    args.iters,
                    device,
                )
                variant_times[variant][batch] = ms
                log(f"[TIME] layer={layer_idx} variant={variant} batch={batch} {ms:.6f} ms")
                del hidden, position_ids, pe, cache
                torch.cuda.empty_cache()
            del layer
            gc.collect()
            torch.cuda.empty_cache()

        for batch in batches:
            row = {
                "workload": "decode",
                "model": args.model,
                "layer_idx": layer_idx,
                "batch": batch,
                "q_len": 1,
                "past_len": args.past_len,
                "hidden_size": hidden_size,
                "timing": "cuda_graph_events",
                "policy_file": policy["path"],
                "note": "standalone_layer_decode_static_cache_smoke_no_gptq_no_calibration",
            }
            for variant in variants:
                row[f"{variant}_ms"] = variant_times[variant][batch]
                for key, value in variant_summaries[variant].items():
                    row[f"{variant}_{key}"] = value
            add_ratios(row)
            rows.append(row)
            log("[ROW] " + json.dumps(row, ensure_ascii=False))
        write_csv(out_dir / "qwen3_decode_split_romeo_quarot_v33_partial.csv", rows)
        (out_dir / "qwen3_decode_split_romeo_quarot_v33_partial.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    write_csv(out_dir / "qwen3_decode_split_romeo_quarot_v33.csv", rows)
    (out_dir / "qwen3_decode_split_romeo_quarot_v33.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    log(f"[CSV] {out_dir / 'qwen3_decode_split_romeo_quarot_v33.csv'}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
