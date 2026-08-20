import argparse
import copy
import csv
import gc
import json
import math
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for p in [TOOLS, Path(os.environ.get("ROMEO_ROOT", "/data/yzy/RoMeo")), ROOT, ROOT / "fake_quant", ROOT / "kernel_quant"]:
    sp = str(p)
    while sp in sys.path:
        sys.path.remove(sp)
    sys.path.insert(0, sp)

import bench_layer_bf16_pure_split_no_gptq_v8 as V8
import bench_multimodel_all_layers_policy_fastqf_v29 as V29
import bench_hadamard_three_schemes_quarot_dense_sharedqkv_inplace_fusedtopr_v43 as V43
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    p.add_argument("--label", default="llama3_8b")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--profile_warmup", type=int, default=1)
    p.add_argument("--profile_iters", type=int, default=3)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--romeo_activation_threshold", type=float, default=0.05)
    p.add_argument("--romeo_weight_threshold", type=float, default=0.05)
    p.add_argument("--romeo_multistream", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--qfactory_fast_preset", default="qwen3_sm120_v1", choices=["none", "qwen3_sm120_v1"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def write_csv(path: Path, rows: List[dict]):
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def event_ms(device: torch.device, fn):
    torch.cuda.synchronize(device)
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    out = fn()
    end.record(stream)
    end.synchronize()
    return out, float(start.elapsed_time(end))


def patch_romeo_quarot_dense_backend(layer: nn.Module, qext) -> List[str]:
    import quarot
    from qfactory.kernels.gemm_w4a4_mixed_precision import gemm_mixed_nt_perchannel

    patched: List[str] = []
    for module_name, mod in layer.named_modules():
        impl = getattr(mod, "linear_impl", None)
        if impl is None:
            continue
        needed = ("w", "w_scale", "w_idx", "w_outliers", "p_a_outlier", "act_quantize")
        if not all(hasattr(impl, x) for x in needed):
            continue

        def make_forward(name):
            def forward_quarot_dense(self, inp: torch.Tensor):
                torch.cuda.set_device(inp.device)
                M = int(inp.shape[0])
                a_outliers = int(math.ceil(M * float(self.p_a_outlier) / 256) * 256)
                w_outliers = int(self.w_outliers)
                body_w = self.w[:-2 * w_outliers] if w_outliers > 0 else self.w
                body_w_scale = self.w_scale[:-w_outliers] if w_outliers > 0 else self.w_scale
                N = int(body_w.shape[0])
                output = torch.empty((M + a_outliers, N + w_outliers), dtype=torch.bfloat16, device=inp.device)

                act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx = self.act_quantize(inp, a_outliers)

                streams = getattr(self, "streams", None)
                if streams is None or len(streams) < 4:
                    streams = [torch.cuda.Stream(device=inp.device) for _ in range(4)]
                current = torch.cuda.current_stream(inp.device)
                ready = current.record_event()
                events = []

                with torch.cuda.stream(streams[0]):
                    streams[0].wait_event(ready)
                    c_i32 = qext.matmul(act.view(M, -1).contiguous(), body_w.view(N, -1).contiguous())
                    y_fp16 = quarot.sym_dequant(
                        c_i32,
                        act_scale.to(dtype=torch.float16).contiguous(),
                        body_w_scale.to(dtype=torch.float16).contiguous(),
                    )
                    output[:M, :N].copy_(y_fp16.to(torch.bfloat16))
                    events.append(streams[0].record_event())

                if a_outliers > 0 and w_outliers > 0:
                    with torch.cuda.stream(streams[1]):
                        streams[1].wait_event(ready)
                        gemm_mixed_nt_perchannel(
                            act_outlier,
                            act_scale_outlier,
                            self.w[-2 * w_outliers:].view(w_outliers, -1),
                            self.w_scale[-w_outliers:],
                            output[-a_outliers:, -w_outliers:],
                            name="a8w8",
                        )
                        events.append(streams[1].record_event())

                if w_outliers > 0:
                    with torch.cuda.stream(streams[2]):
                        streams[2].wait_event(ready)
                        gemm_mixed_nt_perchannel(
                            act,
                            act_scale,
                            self.w[-2 * w_outliers:].view(w_outliers, -1),
                            self.w_scale[-w_outliers:],
                            output[:M, -w_outliers:],
                            name="a4w8",
                        )
                        events.append(streams[2].record_event())

                if a_outliers > 0:
                    with torch.cuda.stream(streams[3]):
                        streams[3].wait_event(ready)
                        gemm_mixed_nt_perchannel(
                            act_outlier,
                            act_scale_outlier,
                            body_w,
                            body_w_scale,
                            output[-a_outliers:, :N],
                            name="a8w4",
                        )
                        events.append(streams[3].record_event())

                for event in events:
                    current.wait_event(event)

                if a_outliers > 0:
                    output[inp_outlier_idx] = output[-a_outliers:].clone()
                if w_outliers > 0:
                    output[:, self.w_idx] = output[:, -w_outliers:]
                return output[:M, :N]

            return forward_quarot_dense

        impl.forward = types.MethodType(make_forward(module_name), impl)
        patched.append(module_name)
    return patched


def install_romeo_breakdown_profiler(layer: nn.Module, qext, collector: List[dict], case: Dict):
    import quarot
    from qfactory.kernels.gemm_w4a4_mixed_precision import gemm_mixed_nt_perchannel

    patched: List[str] = []
    for module_name, mod in layer.named_modules():
        impl = getattr(mod, "linear_impl", None)
        if impl is None:
            continue
        needed = ("w", "w_scale", "w_idx", "w_outliers", "p_a_outlier", "act_quantize")
        if not all(hasattr(impl, x) for x in needed):
            continue

        rotate_type = getattr(mod, "rotate_type", "")

        def make_forward(name, parent_rotate_type):
            def forward_profiled(self, inp: torch.Tensor):
                device = inp.device
                row = dict(case)
                row.update({
                    "module": name,
                    "rotate_type": parent_rotate_type,
                    "M": int(inp.shape[0]),
                    "K": int(inp.shape[1]),
                    "w_outliers": int(self.w_outliers),
                    "p_a_outlier": float(self.p_a_outlier),
                    "act_quant_ms": 0.0,
                    "dense_w4a4_body_ms": 0.0,
                    "outlier_a8w8_ms": 0.0,
                    "outlier_a4w8_ms": 0.0,
                    "outlier_a8w4_ms": 0.0,
                    "post_reorder_ms": 0.0,
                    "linear_impl_profiled_ms": 0.0,
                })

                total_start = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                total_start.record()

                M = int(inp.shape[0])
                a_outliers = int(math.ceil(M * float(self.p_a_outlier) / 256) * 256)
                w_outliers = int(self.w_outliers)
                body_w = self.w[:-2 * w_outliers] if w_outliers > 0 else self.w
                body_w_scale = self.w_scale[:-w_outliers] if w_outliers > 0 else self.w_scale
                N = int(body_w.shape[0])
                output = torch.empty((M + a_outliers, N + w_outliers), dtype=torch.bfloat16, device=device)
                row["a_outliers"] = a_outliers
                row["N"] = N

                (act, act_scale, act_outlier, act_scale_outlier, inp_outlier_idx), row["act_quant_ms"] = event_ms(
                    device, lambda: self.act_quantize(inp, a_outliers)
                )

                def dense_body():
                    c_i32 = qext.matmul(act.view(M, -1).contiguous(), body_w.view(N, -1).contiguous())
                    y_fp16 = quarot.sym_dequant(
                        c_i32,
                        act_scale.to(dtype=torch.float16).contiguous(),
                        body_w_scale.to(dtype=torch.float16).contiguous(),
                    )
                    output[:M, :N].copy_(y_fp16.to(torch.bfloat16))

                _, row["dense_w4a4_body_ms"] = event_ms(device, dense_body)

                if a_outliers > 0 and w_outliers > 0:
                    _, row["outlier_a8w8_ms"] = event_ms(
                        device,
                        lambda: gemm_mixed_nt_perchannel(
                            act_outlier,
                            act_scale_outlier,
                            self.w[-2 * w_outliers:].view(w_outliers, -1),
                            self.w_scale[-w_outliers:],
                            output[-a_outliers:, -w_outliers:],
                            name="a8w8",
                        ),
                    )

                if w_outliers > 0:
                    _, row["outlier_a4w8_ms"] = event_ms(
                        device,
                        lambda: gemm_mixed_nt_perchannel(
                            act,
                            act_scale,
                            self.w[-2 * w_outliers:].view(w_outliers, -1),
                            self.w_scale[-w_outliers:],
                            output[:M, -w_outliers:],
                            name="a4w8",
                        ),
                    )

                if a_outliers > 0:
                    _, row["outlier_a8w4_ms"] = event_ms(
                        device,
                        lambda: gemm_mixed_nt_perchannel(
                            act_outlier,
                            act_scale_outlier,
                            body_w,
                            body_w_scale,
                            output[-a_outliers:, :N],
                            name="a8w4",
                        ),
                    )

                def post_reorder():
                    if a_outliers > 0:
                        output[inp_outlier_idx] = output[-a_outliers:].clone()
                    if w_outliers > 0:
                        output[:, self.w_idx] = output[:, -w_outliers:]
                    return output[:M, :N]

                ret, row["post_reorder_ms"] = event_ms(device, post_reorder)
                total_end.record()
                total_end.synchronize()
                row["linear_impl_profiled_ms"] = float(total_start.elapsed_time(total_end))
                row["outlier_sum_ms"] = row["outlier_a8w8_ms"] + row["outlier_a4w8_ms"] + row["outlier_a8w4_ms"]
                row["linear_impl_other_ms"] = max(
                    0.0,
                    row["linear_impl_profiled_ms"] - row["dense_w4a4_body_ms"] - row["outlier_sum_ms"],
                )
                collector.append(row)
                return ret

            return forward_profiled

        impl.forward = types.MethodType(make_forward(module_name, rotate_type), impl)
        patched.append(module_name)
    return patched


def summarize(total_ms: float, profiled_layer_ms: float, rows: List[dict]) -> dict:
    iters = max(len(set(int(r["profile_iter"]) for r in rows)), 1)
    dense = sum(float(r["dense_w4a4_body_ms"]) for r in rows) / iters
    outlier = sum(float(r["outlier_sum_ms"]) for r in rows) / iters
    impl_other = sum(float(r["linear_impl_other_ms"]) for r in rows) / iters
    layer_other = max(0.0, float(profiled_layer_ms) - dense - outlier)
    raw_sum = max(dense + outlier + layer_other, 1e-9)
    scale = total_ms / raw_sum
    return {
        "romeo_total_graph_ms": total_ms,
        "romeo_profiled_layer_eager_ms": profiled_layer_ms,
        "romeo_dense_w4a4_body_raw_ms": dense,
        "romeo_outlier_w4a8_w8a4_w8a8_raw_ms": outlier,
        "romeo_linear_impl_other_raw_ms": impl_other,
        "romeo_other_raw_ms": layer_other,
        "romeo_breakdown_scale_to_graph_total": scale,
        "romeo_dense_w4a4_body_ms": dense * scale,
        "romeo_outlier_w4a8_w8a4_w8a8_ms": outlier * scale,
        "romeo_other_ms": layer_other * scale,
        "romeo_dense_w4a4_body_pct": dense / raw_sum,
        "romeo_outlier_w4a8_w8a4_w8a8_pct": outlier / raw_sum,
        "romeo_other_pct": layer_other / raw_sum,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log("[NOTE] v52 RoMeo smoke breakdown: normal a4w4 dense body uses QuaRot matmul; a4w8/a8w4/a8w8 outlier GEMMs remain qfactory kernels. Raw component profiling uses synchronized CUDA events, then scales to graph total for percentage reporting.")
    log(f"[MODEL] {args.model} [LAYER] {args.layer} [BATCH] {args.batch} [SEQ_LEN] {args.seq_len}")

    try:
        V29.install_qfactory_fast_preset(args.qfactory_fast_preset)
        V43.install_qfactory_mixed_fast_preset()
        qext = load_quarot_sm120_extension(verbose=bool(int(os.environ.get("QUAROT_SM120_VERBOSE", "0"))))
        log(f"[QUAROT_DENSE_EXT] {getattr(qext, chr(95)+chr(95)+'file'+chr(95)+chr(95), qext)}")

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        ).eval()
        layers = V8.get_layers(model)
        hidden_size = V8.infer_hidden_size(model)
        base_layer = layers[int(args.layer)]

        hidden_fp16 = torch.randn(args.batch, args.seq_len, hidden_size, device=device, dtype=torch.float16)
        hidden_bf16 = hidden_fp16.to(torch.bfloat16)
        position_ids = V8.make_position_ids(args.batch, args.seq_len, device)
        pe_bf16 = V8.build_position_embeddings(model, hidden_bf16, position_ids, torch.bfloat16)

        total_layer = V43.build_romeo_layer(base_layer, args.layer, args, device)
        patched = patch_romeo_quarot_dense_backend(total_layer, qext)
        log(f"[TOTAL_PATCHED] {patched}")
        total_ms = V43.bench_graph(
            lambda: V43.run_layer_once(total_layer, hidden_bf16, position_ids, pe_bf16),
            args.warmup,
            args.iters,
            device,
        )
        log(f"[TIME] romeo_total_graph_ms={total_ms:.6f}")
        total_layer.to("cpu")
        del total_layer
        torch.cuda.empty_cache()

        prof_layer = V43.build_romeo_layer(base_layer, args.layer, args, device)
        collector: List[dict] = []
        case = {
            "model": args.model,
            "model_label": args.label,
            "layer_idx": int(args.layer),
            "batch": int(args.batch),
            "seq_len": int(args.seq_len),
            "hidden_size": int(hidden_size),
            "romeo_dense_kernel": "quarot_a4w4_body",
            "romeo_keeps_outlier_path": True,
            "romeo_multistream_total_path": bool(args.romeo_multistream),
        }
        prof_patched = install_romeo_breakdown_profiler(prof_layer, qext, collector, case)
        log(f"[PROFILE_PATCHED] {prof_patched}")
        for _ in range(args.profile_warmup):
            _ = V43.run_layer_once(prof_layer, hidden_bf16, position_ids, pe_bf16)
        torch.cuda.synchronize(device)
        collector.clear()
        profiled_layer_times = []
        for profile_iter in range(args.profile_iters):
            before = len(collector)
            _, layer_ms = event_ms(device, lambda: V43.run_layer_once(prof_layer, hidden_bf16, position_ids, pe_bf16))
            profiled_layer_times.append(layer_ms)
            for row in collector[before:]:
                row["profile_iter"] = profile_iter
        profiled_layer_ms = sum(profiled_layer_times) / max(len(profiled_layer_times), 1)
        summary = summarize(total_ms, profiled_layer_ms, collector)
        summary.update({
            "args": vars(args),
            "patched_modules": patched,
            "profile_patched_modules": prof_patched,
            "profile_iters": int(args.profile_iters),
            "errors": [],
        })

        rows_csv = out_dir / f"{args.label}_romeo_quarotdense_breakdown_v52.csv"
        rows_json = out_dir / f"{args.label}_romeo_quarotdense_breakdown_v52.json"
        summary_json = out_dir / f"{args.label}_romeo_quarotdense_breakdown_summary_v52.json"
        write_csv(rows_csv, collector)
        json.dump(collector, open(rows_json, "w"), indent=2, ensure_ascii=False)
        json.dump(summary, open(summary_json, "w"), indent=2, ensure_ascii=False)
        log(f"[ROWS_CSV] {rows_csv}")
        log(f"[SUMMARY_JSON] {summary_json}")
        log("[SUMMARY] " + json.dumps(summary, ensure_ascii=False))
        prof_layer.to("cpu")
        del prof_layer, hidden_fp16, hidden_bf16, position_ids, pe_bf16
        gc.collect()
        torch.cuda.empty_cache()
        log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=0")
    except Exception as exc:
        err = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        json.dump(err, open(out_dir / f"{args.label}_romeo_quarotdense_breakdown_error_v52.json", "w"), indent=2, ensure_ascii=False)
        log("[ERROR] " + json.dumps(err, ensure_ascii=False))
        log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=1")
        raise


if __name__ == "__main__":
    main()
