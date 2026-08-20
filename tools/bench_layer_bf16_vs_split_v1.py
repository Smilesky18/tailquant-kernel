import argparse
import copy
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from split_backend_adapter_v1 import discover_split_backends, replace_linear_with_split


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["discover", "bench"], default="bench")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=7)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--split_ratio", type=float, default=0.05)
    p.add_argument("--split_backend_module", default=None)
    p.add_argument("--split_backend_callable", default=None)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--use_cuda_graph", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def get_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find decoder layers. Expected model.model.layers or model.transformer.h")


def infer_hidden_size(model: nn.Module):
    cfg = model.config
    for key in ["hidden_size", "n_embd", "d_model"]:
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    raise RuntimeError("Cannot infer hidden size from model.config")


def make_position_ids(batch: int, seq_len: int, device):
    return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1).contiguous()


def make_attention_mask(batch: int, seq_len: int, device, dtype):
    # Qwen/Llama decoder layer 在不同 transformers 版本里 mask 形状差异很大。
    # 这里先不给显式 mask，优先走 layer 内部默认 causal 逻辑。
    return None


def run_layer_once(layer, hidden_states, position_ids=None, attention_mask=None):
    kwargs = {}
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    try:
        out = layer(hidden_states, **kwargs)
    except TypeError:
        # 兼容部分 layer forward 不接受 position_ids。
        kwargs.pop("position_ids", None)
        try:
            out = layer(hidden_states, **kwargs)
        except TypeError:
            kwargs.pop("attention_mask", None)
            out = layer(hidden_states)

    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def bench_one(layer, hidden_states, position_ids, attention_mask, warmup: int, iters: int, use_cuda_graph: bool):
    device = hidden_states.device

    # 先普通 warmup，触发 lazy init/JIT/cache。
    for _ in range(max(3, warmup // 2)):
        y = run_layer_once(layer, hidden_states, position_ids, attention_mask)
    torch.cuda.synchronize(device)

    if use_cuda_graph:
        static_x = hidden_states.clone()
        static_pos = position_ids.clone() if position_ids is not None else None
        static_mask = attention_mask.clone() if attention_mask is not None else None

        graph = torch.cuda.CUDAGraph()
        for _ in range(warmup):
            y = run_layer_once(layer, static_x, static_pos, static_mask)
        torch.cuda.synchronize(device)

        with torch.cuda.graph(graph):
            y = run_layer_once(layer, static_x, static_pos, static_mask)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / iters)

    else:
        for _ in range(warmup):
            y = run_layer_once(layer, hidden_states, position_ids, attention_mask)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            y = run_layer_once(layer, hidden_states, position_ids, attention_mask)
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / iters)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "discover":
        candidates = discover_split_backends()
        path = out_dir / "split_backend_candidates.json"
        json.dump(candidates, open(path, "w"), indent=2, ensure_ascii=False)
        log(json.dumps(candidates, indent=2, ensure_ascii=False))
        log(f"[DISCOVER_OUT] {path}")
        return

    torch.manual_seed(args.seed)
    torch.cuda.set_device(torch.device(args.device))
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[BATCHES] {args.batches}")
    log(f"[DTYPE] {dtype}")
    log(f"[DEVICE] {device}")
    log(f"[CUDA_GRAPH] {args.use_cuda_graph}")
    log(f"[SPLIT_RATIO] {args.split_ratio}")
    log(f"[SPLIT_BACKEND] {args.split_backend_module}.{args.split_backend_callable}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    layers = get_layers(model)
    if args.layer_idx < 0 or args.layer_idx >= len(layers):
        raise ValueError(f"layer_idx out of range: {args.layer_idx}, num_layers={len(layers)}")

    hidden_size = infer_hidden_size(model)
    log(f"[HIDDEN_SIZE] {hidden_size}")
    log(f"[NUM_LAYERS] {len(layers)}")

    bf16_layer = copy.deepcopy(layers[args.layer_idx]).to(device=device, dtype=dtype).eval()
    split_layer = copy.deepcopy(layers[args.layer_idx]).to(device=device, dtype=dtype).eval()
    replace_linear_with_split(
        split_layer,
        ratio=args.split_ratio,
        backend_module=args.split_backend_module,
        backend_callable=args.split_backend_callable,
    )
    split_layer.to(device=device, dtype=dtype).eval()

    # 原始大模型释放掉，只保留两个 layer。
    del model
    gc.collect()
    torch.cuda.empty_cache()

    rows: List[Dict[str, object]] = []
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    for batch in batches:
        log(f"\n[CASE] batch={batch} seq_len={args.seq_len}")
        hidden_states = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=dtype)
        position_ids = make_position_ids(batch, args.seq_len, device)
        attention_mask = make_attention_mask(batch, args.seq_len, device, dtype)

        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated(device)

        bf16_ms = bench_one(
            bf16_layer,
            hidden_states,
            position_ids,
            attention_mask,
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_graph=args.use_cuda_graph,
        )

        torch.cuda.empty_cache()
        split_ms = bench_one(
            split_layer,
            hidden_states,
            position_ids,
            attention_mask,
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_graph=args.use_cuda_graph,
        )

        torch.cuda.synchronize(device)
        mem_after = torch.cuda.max_memory_allocated(device)

        speedup = bf16_ms / split_ms if split_ms > 0 else float("nan")

        row = {
            "model": args.model,
            "layer_idx": args.layer_idx,
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            "split_ratio": args.split_ratio,
            "bf16_ms": bf16_ms,
            "split_ms": split_ms,
            "speedup_bf16_over_split": speedup,
            "normalized_split_latency": split_ms / bf16_ms if bf16_ms > 0 else float("nan"),
            "cuda_graph": bool(args.use_cuda_graph),
            "dtype": args.dtype,
            "mem_alloc_before": int(mem_before),
            "max_mem_alloc_after": int(mem_after),
            "split_backend_module": args.split_backend_module,
            "split_backend_callable": args.split_backend_callable,
        }
        rows.append(row)
        log("[RESULT] " + json.dumps(row, indent=2))

    csv_path = out_dir / "bf16_vs_split_layer_latency.csv"
    json_path = out_dir / "bf16_vs_split_layer_latency.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json.dump(rows, open(json_path, "w"), indent=2)

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
