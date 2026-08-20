import argparse
import copy
import csv
import json
import os
import time
import types
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split_ratio", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--profile_runs", type=int, default=5)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def is_real_policy_linear(m: nn.Module) -> bool:
    cls = m.__class__.__name__
    return cls == "RealPolicyLinear" or "RealPolicyLinear" in cls


def instrument_real_policy_linears(layer: nn.Module, tag: str, records: List[Dict]):
    """
    对每个 RealPolicyLinear.forward 加 CUDA event 计时。
    这是诊断用，会强制 synchronize，所以不要把这个结果当最终 latency。
    """
    patched = []

    for name, module in layer.named_modules():
        if not is_real_policy_linear(module):
            continue

        orig_forward = module.forward

        def make_wrapped(orig, mod_name):
            def wrapped(self, *args, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                out = orig(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                ms = float(start.elapsed_time(end))
                records.append({
                    "tag": tag,
                    "module": mod_name,
                    "ms": ms,
                    "class": self.__class__.__name__,
                })
                return out
            return wrapped

        module.forward = types.MethodType(make_wrapped(orig_forward, name), module)
        patched.append(name)

    log(f"[INSTRUMENT] tag={tag} modules={patched}")
    return patched


def run_layer(layer, hidden_states, position_ids, position_embeddings):
    out = layer(
        hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    return out[0] if isinstance(out, tuple) else out


def summarize_module_records(records: List[Dict]):
    agg: Dict[tuple, Dict] = {}
    for r in records:
        key = (r["tag"], r["module"])
        a = agg.setdefault(key, {
            "tag": r["tag"],
            "module": r["module"],
            "count": 0,
            "total_ms": 0.0,
            "min_ms": 1e30,
            "max_ms": 0.0,
        })
        ms = float(r["ms"])
        a["count"] += 1
        a["total_ms"] += ms
        a["min_ms"] = min(a["min_ms"], ms)
        a["max_ms"] = max(a["max_ms"], ms)

    rows = []
    for a in agg.values():
        a["mean_ms"] = a["total_ms"] / max(a["count"], 1)
        rows.append(a)
    rows.sort(key=lambda x: (x["tag"], -x["mean_ms"]))
    return rows


def torch_profile_one(tag, fn, out_dir: Path, batch: int, active_runs: int):
    from torch.profiler import profile, ProfilerActivity

    # 先跑一次，避免 profiler 把 lazy init 算得太重。
    fn()
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for _ in range(active_runs):
            fn()
        torch.cuda.synchronize()

    rows = []
    for item in prof.key_averages():
        row = {
            "tag": tag,
            "batch": batch,
            "key": item.key,
            "count": int(item.count),
        }
        for attr in [
            "self_cuda_time_total",
            "cuda_time_total",
            "self_cpu_time_total",
            "cpu_time_total",
        ]:
            try:
                row[attr + "_us"] = float(getattr(item, attr))
            except Exception:
                row[attr + "_us"] = None
        rows.append(row)

    rows.sort(key=lambda x: x.get("self_cuda_time_total_us") or 0.0, reverse=True)

    json_path = out_dir / f"torch_profiler_{tag}_b{batch}.json"
    csv_path = out_dir / f"torch_profiler_{tag}_b{batch}.csv"

    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)

    fields = list(rows[0].keys()) if rows else ["tag", "batch", "key"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    log(f"[TORCH_PROFILE_JSON] {json_path}")
    log(f"[TORCH_PROFILE_CSV] {csv_path}")
    log(f"[TORCH_PROFILE_TOP_{tag}_b{batch}]")
    for r in rows[:20]:
        log(json.dumps(r, ensure_ascii=False))


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[BATCHES] {args.batches}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[SPLIT_RATIO] {args.split_ratio}")
    log("[NOTE] diagnostic only: per-module timings use synchronize and are not final latency numbers")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    layers = V8.get_layers(model)
    hidden_size = V8.infer_hidden_size(model)
    base_layer = layers[args.layer_idx]

    pure_layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    pure_layer, pure_records = V8.patch_layer_with_real_policy(
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

    split_layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    split_layer, split_records = V8.patch_layer_with_real_policy(
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

    module_records: List[Dict] = []
    instrument_real_policy_linears(pure_layer, "pure", module_records)
    instrument_real_policy_linears(split_layer, "split", module_records)

    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    for batch in batches:
        log(f"\n[CASE] batch={batch} seq_len={args.seq_len}")
        hidden = torch.randn(
            batch,
            args.seq_len,
            hidden_size,
            device=device,
            dtype=torch.float16,
        )
        position_ids = V8.make_position_ids(batch, args.seq_len, device)
        pe = V8.build_position_embeddings(model, hidden, position_ids, torch.float16)
        log("[POSITION_EMBEDDINGS] " + str([(tuple(t.shape), str(t.dtype)) for t in pe]))

        # warmup
        for _ in range(args.warmup):
            _ = run_layer(pure_layer, hidden, position_ids, pe)
            _ = run_layer(split_layer, hidden, position_ids, pe)
        torch.cuda.synchronize()

        # 清掉 warmup module record
        module_records.clear()

        for i in range(args.profile_runs):
            _ = run_layer(pure_layer, hidden, position_ids, pe)
            _ = run_layer(split_layer, hidden, position_ids, pe)
        torch.cuda.synchronize()

        batch_records = []
        for r in module_records:
            rr = dict(r)
            rr["batch"] = batch
            rr["seq_len"] = args.seq_len
            batch_records.append(rr)

        summary_rows = summarize_module_records(batch_records)

        mod_json = out_dir / f"module_forward_timing_b{batch}.json"
        mod_csv = out_dir / f"module_forward_timing_b{batch}.csv"
        sum_json = out_dir / f"module_forward_summary_b{batch}.json"
        sum_csv = out_dir / f"module_forward_summary_b{batch}.csv"

        json.dump(batch_records, open(mod_json, "w"), indent=2, ensure_ascii=False)
        json.dump(summary_rows, open(sum_json, "w"), indent=2, ensure_ascii=False)

        if batch_records:
            with open(mod_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(batch_records[0].keys()))
                writer.writeheader()
                writer.writerows(batch_records)

        if summary_rows:
            with open(sum_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)

        log(f"[MODULE_TIMING_JSON] {mod_json}")
        log(f"[MODULE_SUMMARY_JSON] {sum_json}")
        log(f"[MODULE_SUMMARY_TOP_b{batch}]")
        for r in summary_rows:
            log(json.dumps(r, ensure_ascii=False))

        # torch profiler 只对 batch=16,64 各做一次。用于看 CUDA kernel 名和 self CUDA time。
        def pure_fn():
            return run_layer(pure_layer, hidden, position_ids, pe)

        def split_fn():
            return run_layer(split_layer, hidden, position_ids, pe)

        torch_profile_one("pure", pure_fn, out_dir, batch, active_runs=2)
        torch_profile_one("split", split_fn, out_dir, batch, active_runs=2)

        module_records.clear()
        torch.cuda.empty_cache()

    json.dump(pure_records, open(out_dir / "pure_patch_records_v9.json", "w"), indent=2, ensure_ascii=False)
    json.dump(split_records, open(out_dir / "split_patch_records_v9.json", "w"), indent=2, ensure_ascii=False)

    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
