import argparse
import csv
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=7)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--use_cuda_graph", action="store_true")
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
    """
    Qwen3 / new Llama-style layers require position_embeddings=(cos, sin)
    when a single decoder layer is invoked standalone.
    """
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb = model.model.rotary_emb.to(hidden_states.device)
        return rotary_emb(hidden_states, position_ids)

    # Some older models keep rotary_emb inside attention.
    layers = get_layers(model)
    attn = layers[0].self_attn if hasattr(layers[0], "self_attn") else None
    if attn is not None and hasattr(attn, "rotary_emb"):
        rotary_emb = attn.rotary_emb.to(hidden_states.device)
        return rotary_emb(hidden_states, position_ids)

    return None


def run_layer_once(layer, hidden_states, position_ids=None, position_embeddings=None, attention_mask=None):
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings

    try:
        out = layer(hidden_states, **kwargs)
    except TypeError as e1:
        # 兼容不同 transformers 版本。
        kwargs.pop("position_ids", None)
        try:
            out = layer(hidden_states, **kwargs)
        except TypeError as e2:
            kwargs.pop("position_embeddings", None)
            try:
                out = layer(hidden_states, **kwargs)
            except TypeError as e3:
                kwargs.pop("attention_mask", None)
                out = layer(hidden_states)

    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def bench_one(layer, hidden_states, position_ids, position_embeddings, attention_mask, warmup: int, iters: int, use_cuda_graph: bool):
    device = hidden_states.device

    for _ in range(max(3, warmup // 2)):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings, attention_mask)
    torch.cuda.synchronize(device)

    if use_cuda_graph:
        static_x = hidden_states.clone()
        static_pos = position_ids.clone() if position_ids is not None else None
        if position_embeddings is None:
            static_pe = None
        else:
            static_pe = tuple(t.clone() for t in position_embeddings)
        static_mask = attention_mask.clone() if attention_mask is not None else None

        for _ in range(warmup):
            _ = run_layer_once(layer, static_x, static_pos, static_pe, static_mask)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _ = run_layer_once(layer, static_x, static_pos, static_pe, static_mask)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / iters)

    for _ in range(warmup):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings, attention_mask)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = run_layer_once(layer, hidden_states, position_ids, position_embeddings, attention_mask)
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


def dump_real_split_api(out_dir: Path):
    info = []
    try:
        import kernel_quant.scripts.bench_real_split_fullstack_v1 as realbench
        names = [
            "RealPolicyLinear",
            "patch_real_linears",
            "run_first_layer",
            "run_scope",
            "unwrap_linear",
        ]
        for name in names:
            obj = getattr(realbench, name, None)
            if obj is None:
                info.append({"name": name, "status": "missing"})
                continue
            try:
                sig = str(inspect.signature(obj))
            except Exception as e:
                sig = f"<signature error: {repr(e)}>"
            try:
                src = inspect.getsource(obj)
            except Exception as e:
                src = f"<source error: {repr(e)}>"
            info.append({
                "name": name,
                "status": "ok",
                "signature": sig,
                "source_head": "\n".join(src.splitlines()[:120]),
            })

        try:
            main_help = os.popen(f"{sys.executable} -u {realbench.__file__} --help 2>&1").read()
        except Exception as e:
            main_help = repr(e)

        info.append({
            "name": "bench_real_split_fullstack_v1_help",
            "status": "ok",
            "file": realbench.__file__,
            "help": main_help,
        })
    except Exception as e:
        info.append({"name": "import_realbench", "status": "failed", "error": repr(e)})

    path = out_dir / "real_split_fullstack_api_dump_v2.json"
    json.dump(info, open(path, "w"), indent=2, ensure_ascii=False)
    log(f"[REAL_SPLIT_API_DUMP] {path}")


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
    log(f"[CUDA_GRAPH] {args.use_cuda_graph}")

    dump_real_split_api(out_dir)

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

    layer = layers[args.layer_idx].to(device=device, dtype=dtype).eval()

    # 保留 model.model.rotary_emb，用于 standalone layer 的 position_embeddings。
    log(f"[HIDDEN_SIZE] {hidden_size}")
    log(f"[NUM_LAYERS] {len(layers)}")
    log(f"[LAYER_FORWARD_SIGNATURE] {inspect.signature(layer.forward)}")

    rows: List[Dict[str, object]] = []
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    for batch in batches:
        log(f"\n[CASE] batch={batch} seq_len={args.seq_len}")

        hidden_states = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=dtype)
        position_ids = make_position_ids(batch, args.seq_len, device)
        position_embeddings = build_position_embeddings(model, hidden_states, position_ids)
        attention_mask = None

        if position_embeddings is None:
            log("[WARN] position_embeddings=None")
        else:
            log("[POSITION_EMBEDDINGS] " + str([tuple(t.shape) for t in position_embeddings]))

        torch.cuda.empty_cache()
        ms = bench_one(
            layer,
            hidden_states,
            position_ids,
            position_embeddings,
            attention_mask,
            warmup=args.warmup,
            iters=args.iters,
            use_cuda_graph=args.use_cuda_graph,
        )

        row = {
            "model": args.model,
            "layer_idx": args.layer_idx,
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            "bf16_ms": ms,
            "cuda_graph": bool(args.use_cuda_graph),
            "dtype": args.dtype,
        }
        rows.append(row)
        log("[BF16_RESULT] " + json.dumps(row, indent=2))

    csv_path = out_dir / "bf16_layer_latency_probe_v2.csv"
    json_path = out_dir / "bf16_layer_latency_probe_v2.json"

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
