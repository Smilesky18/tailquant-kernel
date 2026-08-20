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
from typing import List

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
from fused_topr_pack_ext_v42 import load_fused_topr_pack_ext
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension


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
    p.add_argument("--variants", default="bf16,romeo,split")
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
    p.add_argument("--fused_topr_pack", action=argparse.BooleanOptionalAction, default=True)
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


def is_qwen_model(model_name: str, model) -> bool:
    mt = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    return "qwen" in mt or "qwen" in model_name.lower()


def build_bf16_layer(base_layer: nn.Module, device: torch.device):
    return copy.deepcopy(base_layer).to(device=device, dtype=torch.bfloat16).eval()


def build_split_layer(base_layer, layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, eps, device, qext, fused_topr_ext, qwen_shared: bool):
    if qwen_shared:
        return V43.build_split_layer_with_hadamard(
            base_layer, layer_idx, policy, rot_flags, B,
            main_ext, layout_ext, policy_pack_ext, eps, device, qext,
            fused_topr_ext=fused_topr_ext,
        )

    old_patch = V43.patch_qwen_qkv_shared_preprocess
    try:
        V43.patch_qwen_qkv_shared_preprocess = lambda layer, enable_shared_topk=True: []
        return V43.build_split_layer_with_hadamard(
            base_layer, layer_idx, policy, rot_flags, B,
            main_ext, layout_ext, policy_pack_ext, eps, device, qext,
            fused_topr_ext=fused_topr_ext,
        )
    finally:
        V43.patch_qwen_qkv_shared_preprocess = old_patch


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
    log(f"[BATCHES] {args.batches} [LAYERS] {args.layers} [SEQ_LEN] {args.seq_len}")
    log("[NOTE] v53 total-only per-layer latency. Variants: BF16, RoMeO with QuaRot a4w4 dense body and complete outlier path, Split with QuaRot dense backend. No dense-only or component profiling is run.")

    V29.install_qfactory_fast_preset(args.qfactory_fast_preset)
    V43.install_qfactory_mixed_fast_preset()

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)
    policy = V29.load_policy(Path(args.policy), args.force_activation_percentile_100)
    qext = load_quarot_sm120_extension(verbose=bool(int(os.environ.get("QUAROT_SM120_VERBOSE", "0"))))
    log(f"[QUAROT_DENSE_EXT] {getattr(qext, chr(95)+chr(95)+'file'+chr(95)+chr(95), qext)}")
    fused_topr_ext = None
    if args.fused_topr_pack:
        fused_topr_ext = load_fused_topr_pack_ext(verbose=bool(int(os.environ.get("FUSED_TOPR_VERBOSE", "0"))))
        log(f"[FUSED_TOPR_EXT] {getattr(fused_topr_ext, chr(95)+chr(95)+'file'+chr(95)+chr(95), fused_topr_ext)}")

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
    qwen_shared = is_qwen_model(args.model, model)
    log(f"[MODEL_KIND] qwen_shared_patch={qwen_shared}")
    rot_flags = B.H.build_rotation_flags(model, args.rotation_config)
    log(f"[SPLIT_ROT_FLAGS] selected={sum(bool(v) for v in rot_flags.values())}/{len(rot_flags)}")

    rows = []
    errors = []
    for layer_idx in layer_indices:
        base_layer = layers[layer_idx]
        for batch in batches:
            log(f"[CASE] label={args.label} layer={layer_idx} batch={batch}")
            hidden_fp16 = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=torch.float16)
            hidden_bf16 = hidden_fp16.to(torch.bfloat16)
            position_ids = V8.make_position_ids(batch, args.seq_len, device)
            pe_fp16 = V8.build_position_embeddings(model, hidden_fp16, position_ids, torch.float16)
            pe_bf16 = V8.build_position_embeddings(model, hidden_bf16, position_ids, torch.bfloat16)
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
                "romeo_impl": "romeo_complete_mixed_outliers_quarot_a4w4_body_v53",
                "split_impl": "split_real_policy_quarot_dense_v53",
                "fused_topr_pack": bool(args.fused_topr_pack),
            }

            try:
                if "bf16" in variants:
                    bf16_layer = build_bf16_layer(base_layer, device)
                    row["bf16_ms"] = V43.bench_graph(lambda: V43.run_layer_once(bf16_layer, hidden_bf16, position_ids, pe_bf16), args.warmup, args.iters, device)
                    log(f"[TIME] layer={layer_idx} batch={batch} bf16_ms={row['bf16_ms']:.6f}")
                    del bf16_layer
                    torch.cuda.empty_cache()

                if "romeo" in variants:
                    romeo_layer = V43.build_romeo_layer(base_layer, layer_idx, args, device)
                    patched_romeo = patch_romeo_quarot_dense_backend(romeo_layer, qext)
                    row["romeo_quarotdense_patched_modules"] = len(patched_romeo)
                    row["romeo_ms"] = V43.bench_graph(lambda: V43.run_layer_once(romeo_layer, hidden_bf16, position_ids, pe_bf16), args.warmup, args.iters, device)
                    log(f"[TIME] layer={layer_idx} batch={batch} romeo_ms={row['romeo_ms']:.6f} patched={len(patched_romeo)}")
                    romeo_layer.to("cpu")
                    del romeo_layer
                    torch.cuda.empty_cache()

                if "split" in variants:
                    split_layer, rec = build_split_layer(base_layer, layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, args.eps, device, qext, fused_topr_ext, qwen_shared)
                    row["split_mean_ratio"] = sum(float(r["ratio"]) for r in rec) / max(len(rec), 1)
                    row["split_max_ratio"] = max([float(r["ratio"]) for r in rec] or [0.0])
                    row["split_nonzero_modules"] = sum(1 for r in rec if float(r["ratio"]) > 0.0)
                    row["split_sum_R"] = sum(int(r["R"]) for r in rec)
                    row["split_ms"] = V43.bench_graph(lambda: V43.run_layer_once(split_layer, hidden_fp16, position_ids, pe_fp16), args.warmup, args.iters, device)
                    log(f"[TIME] layer={layer_idx} batch={batch} split_ms={row['split_ms']:.6f}")
                    del split_layer
                    torch.cuda.empty_cache()

                if row.get("bf16_ms") and row.get("romeo_ms"):
                    row["romeo_over_bf16"] = row["romeo_ms"] / row["bf16_ms"]
                if row.get("bf16_ms") and row.get("split_ms"):
                    row["split_over_bf16"] = row["split_ms"] / row["bf16_ms"]
                if row.get("romeo_ms") and row.get("split_ms"):
                    row["split_over_romeo"] = row["split_ms"] / row["romeo_ms"]

            except Exception as exc:
                err = {"model": args.model, "layer_idx": int(layer_idx), "batch": int(batch), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                errors.append(err)
                row["error"] = err["error"]
                log("[ERROR] " + json.dumps(err, ensure_ascii=False))

            rows.append(row)
            write_csv(out_dir / f"{args.label}_prefill_layer_total_v53_partial.csv", rows)
            json.dump(rows, open(out_dir / f"{args.label}_prefill_layer_total_v53_partial.json", "w"), indent=2, ensure_ascii=False)
            del hidden_fp16, hidden_bf16, position_ids, pe_fp16, pe_bf16
            gc.collect()
            torch.cuda.empty_cache()

    layer_csv = out_dir / f"{args.label}_prefill_layer_total_v53.csv"
    layer_json = out_dir / f"{args.label}_prefill_layer_total_v53.json"
    meta_path = out_dir / f"{args.label}_prefill_layer_total_meta_v53.json"
    write_csv(layer_csv, rows)
    json.dump(rows, open(layer_json, "w"), indent=2, ensure_ascii=False)
    summary = {
        "args": vars(args),
        "layer_csv": str(layer_csv),
        "layer_json": str(layer_json),
        "errors": errors,
        "num_rows": len(rows),
        "sum_bf16_ms": sum(float(r.get("bf16_ms", 0.0)) for r in rows),
        "sum_romeo_ms": sum(float(r.get("romeo_ms", 0.0)) for r in rows),
        "sum_split_ms": sum(float(r.get("split_ms", 0.0)) for r in rows),
    }
    json.dump(summary, open(meta_path, "w"), indent=2, ensure_ascii=False)
    log(f"[LAYER_CSV] {layer_csv}")
    log(f"[META] {meta_path}")
    log("[SUMMARY] " + json.dumps(summary, ensure_ascii=False))
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=0")


if __name__ == "__main__":
    main()
