#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-profile split sparse tuning variants on one no-GPTQ split layer.

Experiment-only file. Original project sources are not modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
EXP = ROOT / "experiments/kernel_quant/layer_latency_split_v1"
TOOLS = EXP / "tools"
for item in (ROOT, ROOT / "fake_quant", ROOT / "kernel_quant", ROOT / "kernel_quant/scripts", TOOLS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

ENV_BIN = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = ENV_BIN + os.pathsep + os.environ.get("PATH", "")

import bench_layer_bf16_pure_split_no_gptq_v8 as V8  # noqa: E402
from split_sparse_tuned_ext_v21 import load_sparse_tuned_ext  # noqa: E402


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--ratio", type=float, default=0.05)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--variants", default="baseline,oct_b128,oct_b256,oct_auto")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", str(ROOT / "third_party/cutlass")))
    p.add_argument("--verbose_compile", action="store_true")
    return p.parse_args()


def run_layer_once(layer, hidden_states, position_ids, position_embeddings):
    out = layer(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings)
    return out[0] if isinstance(out, tuple) else out


def is_real_policy_linear(m):
    name = m.__class__.__name__
    return name == "RealPolicyLinear" or "RealPolicyLinear" in name


def event_ms(fn, device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    ret = fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)), ret


def sparse_variant_call(self, ext, tuned_ext, variant, top_q, indices, scratch):
    if variant == "baseline":
        scratch["Y_sparse"].zero_()
        ext.sparse_top_add_rowmajor_quad_shared(top_q, indices, self.B_row, scratch["top_scale"], self.w_scale, scratch["Y_sparse"], self.K)
    else:
        effective = variant
        if variant == "oct_auto":
            effective = "oct_b256" if self.N >= 4096 else "oct_b128"
        if effective == "oct_b256":
            tuned_ext.sparse_top_write_rowmajor_oct_b256_shared(top_q, indices, self.B_row, scratch["top_scale"], self.w_scale, scratch["Y_sparse"], self.K)
        elif effective == "oct_b128":
            tuned_ext.sparse_top_write_rowmajor_oct_b128_shared(top_q, indices, self.B_row, scratch["top_scale"], self.w_scale, scratch["Y_sparse"], self.K)
        else:
            raise ValueError(variant)


def install_split_stage_profiler(layer, records, variant, tuned_ext):
    patched = []
    for name, mod in layer.named_modules():
        if not is_real_policy_linear(mod):
            continue
        if not getattr(mod, "is_split", False):
            continue

        def make_profiled_split_compute(module_name):
            def profiled_split_compute(self, A, scratch, B_col, dense_ready_event, dense_stream, sparse_stream):
                M = int(A.shape[0])
                device = A.device
                ext = getattr(self, "ext", None) or getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError("cannot find ext/main_ext")

                prepare_ms, prep_ret = event_ms(lambda: self._prepare_split(A, scratch), device)
                indices, top_q, _ = prep_ret

                def dense_body_fn():
                    ext.cutlass_s4_gemm(scratch["A_pack"], B_col, scratch["C_body_i32"], M, self.N, self.K)
                    ext.scale_i32_to_fp16(scratch["C_body_i32"], scratch["body_scale"], self.w_scale, scratch["Y_body"])

                dense_ms, _ = event_ms(dense_body_fn, device)
                sparse_ms, _ = event_ms(lambda: sparse_variant_call(self, ext, tuned_ext, variant, top_q, indices, scratch), device)
                output = torch.empty((M, self.N), dtype=torch.float16, device=device)
                merge_ms, _ = event_ms(lambda: torch.add(scratch["Y_body"], scratch["Y_sparse"], out=output), device)

                total = prepare_ms + dense_ms + sparse_ms + merge_ms
                effective = variant
                if variant == "oct_auto":
                    effective = "oct_b256" if self.N >= 4096 else "oct_b128"
                records.append({
                    "variant": variant,
                    "effective_sparse_variant": effective,
                    "module": module_name,
                    "M": M,
                    "K": int(self.K),
                    "N": int(self.N),
                    "R": int(self.R),
                    "ratio": float(self.ratio),
                    "prepare_split_ms": prepare_ms,
                    "dense_body_ms": dense_ms,
                    "sparse_correction_ms": sparse_ms,
                    "merge_ms": merge_ms,
                    "sum_serial_ms": total,
                    "prepare_pct": prepare_ms / total if total > 0 else 0.0,
                    "dense_pct": dense_ms / total if total > 0 else 0.0,
                    "sparse_pct": sparse_ms / total if total > 0 else 0.0,
                    "merge_pct": merge_ms / total if total > 0 else 0.0,
                })
                return output
            return profiled_split_compute

        mod._split_compute = types.MethodType(make_profiled_split_compute(name), mod)
        patched.append(name)
    return patched


def summarize(records):
    agg = {}
    for r in records:
        key = (r["variant"], r["module"])
        a = agg.setdefault(key, {
            "variant": r["variant"],
            "effective_sparse_variant": r["effective_sparse_variant"],
            "module": r["module"],
            "count": 0,
            "M": r["M"],
            "K": r["K"],
            "N": r["N"],
            "R": r["R"],
            "ratio": r["ratio"],
            "prepare_split_ms": 0.0,
            "dense_body_ms": 0.0,
            "sparse_correction_ms": 0.0,
            "merge_ms": 0.0,
            "sum_serial_ms": 0.0,
        })
        a["count"] += 1
        for k in ["prepare_split_ms", "dense_body_ms", "sparse_correction_ms", "merge_ms", "sum_serial_ms"]:
            a[k] += float(r[k])
    rows = []
    for a in agg.values():
        c = max(a["count"], 1)
        for k in ["prepare_split_ms", "dense_body_ms", "sparse_correction_ms", "merge_ms", "sum_serial_ms"]:
            a[k] /= c
        total = a["sum_serial_ms"]
        a["prepare_pct"] = a["prepare_split_ms"] / total if total > 0 else 0.0
        a["dense_pct"] = a["dense_body_ms"] / total if total > 0 else 0.0
        a["sparse_pct"] = a["sparse_correction_ms"] / total if total > 0 else 0.0
        a["merge_pct"] = a["merge_ms"] / total if total > 0 else 0.0
        rows.append(a)
    rows.sort(key=lambda x: (x["variant"], -x["sum_serial_ms"]))
    return rows


def summarize_total(summary_rows, variant):
    rows = [r for r in summary_rows if r["variant"] == variant and r["module"] != "__TOTAL__"]
    total = {
        "variant": variant,
        "effective_sparse_variant": "mixed" if variant == "oct_auto" else variant,
        "module": "__TOTAL__",
        "count": sum(int(r["count"]) for r in rows),
        "M": "",
        "K": "",
        "N": "",
        "R": "",
        "ratio": "",
        "prepare_split_ms": sum(float(r["prepare_split_ms"]) for r in rows),
        "dense_body_ms": sum(float(r["dense_body_ms"]) for r in rows),
        "sparse_correction_ms": sum(float(r["sparse_correction_ms"]) for r in rows),
        "merge_ms": sum(float(r["merge_ms"]) for r in rows),
        "sum_serial_ms": sum(float(r["sum_serial_ms"]) for r in rows),
    }
    s = total["sum_serial_ms"]
    total["prepare_pct"] = total["prepare_split_ms"] / s if s > 0 else 0.0
    total["dense_pct"] = total["dense_body_ms"] / s if s > 0 else 0.0
    total["sparse_pct"] = total["sparse_correction_ms"] / s if s > 0 else 0.0
    total["merge_pct"] = total["merge_ms"] / s if s > 0 else 0.0
    return total


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[RATIO] {args.ratio}")
    log(f"[BATCHES] {args.batches}")
    log(f"[VARIANTS] {args.variants}")
    log("[NOTE] Split sparse tuned stage profile: serial attribution, no original source modifications.")

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability(device)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        log(f"[EXT] set TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)
    tuned_ext = load_sparse_tuned_ext(args.verbose_compile)
    log(f"[TUNED_EXT] {tuned_ext}")

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval()
    layers = V8.get_layers(model)
    hidden_size = V8.infer_hidden_size(model)
    base_layer = layers[args.layer_idx]
    split_layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    split_layer, patch_records = V8.patch_layer_with_real_policy(
        layer=split_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="dual_policy",
        ratio=args.ratio,
        eps=args.eps,
        device=device,
    )
    split_layer.to(device=device).eval()

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    all_rows = []

    for batch in batches:
        hidden = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=torch.float16)
        position_ids = V8.make_position_ids(batch, args.seq_len, device)
        pe = V8.build_position_embeddings(model, hidden, position_ids, torch.float16)
        for variant in variants:
            log(f"\n[CASE] batch={batch} seq_len={args.seq_len} variant={variant}")
            stage_records = []
            patched = install_split_stage_profiler(split_layer, stage_records, variant, tuned_ext)
            log("[PATCHED] " + json.dumps(patched))
            for _ in range(args.warmup):
                _ = run_layer_once(split_layer, hidden, position_ids, pe)
            torch.cuda.synchronize(device)
            stage_records.clear()
            for _ in range(args.runs):
                _ = run_layer_once(split_layer, hidden, position_ids, pe)
            torch.cuda.synchronize(device)
            batch_records = [dict(r, batch=batch, seq_len=args.seq_len) for r in stage_records]
            summary = summarize(batch_records)
            total_row = summarize_total(summary, variant)
            total_row["batch"] = batch
            total_row["seq_len"] = args.seq_len
            for r in summary:
                r["batch"] = batch
                r["seq_len"] = args.seq_len
            rows = summary + [total_row]
            all_rows.extend(rows)
            log(f"[TOTAL_b{batch}_{variant}] " + json.dumps(total_row, ensure_ascii=False))

    fields = sorted({k for r in all_rows for k in r.keys()})
    csv_path = out_dir / "split_sparse_tuned_stage_v21.csv"
    json_path = out_dir / "split_sparse_tuned_stage_v21.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    json.dump(all_rows, open(json_path, "w"), indent=2, ensure_ascii=False)
    json.dump(patch_records, open(out_dir / "split_patch_records_v21.json", "w"), indent=2, ensure_ascii=False)
    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
