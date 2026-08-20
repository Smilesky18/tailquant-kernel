
import argparse
import copy
import csv
import gc
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Dict, List, Tuple

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
from fused_topr_pack_ext_v42 import load_fused_topr_pack_ext


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
    p.add_argument("--variants", default="bf16,quarot,split")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--component_warmup", type=int, default=1)
    p.add_argument("--component_iters", type=int, default=3)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--force_activation_percentile_100", action="store_true")
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


def is_real_policy_linear(m: nn.Module) -> bool:
    name = m.__class__.__name__
    return name == "RealPolicyLinear" or "RealPolicyLinear" in name


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


def build_bf16_layer(base_layer: nn.Module, device: torch.device):
    return copy.deepcopy(base_layer).to(device=device, dtype=torch.bfloat16).eval()


def build_quarot_linear4bit_fallback_layer(base_layer: nn.Module, device: torch.device):
    import quarot

    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()

    def replace_linear(parent: nn.Module, name: str, add_hadamard: bool = False):
        linear = getattr(parent, name, None)
        if not isinstance(linear, nn.Linear):
            return False
        parts = []
        if add_hadamard:
            parts.append(quarot.nn.OnlineHadamard(int(linear.in_features)))
        parts.append(quarot.nn.Quantizer())
        parts.append(quarot.nn.Linear4bit.from_float(linear))
        setattr(parent, name, nn.Sequential(*parts))
        return True

    attn = getattr(layer, "self_attn", None)
    if attn is not None:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            replace_linear(attn, name, add_hadamard=False)

    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        replace_linear(mlp, "gate_proj", add_hadamard=False)
        replace_linear(mlp, "up_proj", add_hadamard=False)
        replace_linear(mlp, "down_proj", add_hadamard=True)

    layer = layer.to(device=device, dtype=torch.float16).eval()
    return V43.prepare_quarot_latency_layer(layer, device)


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


def install_split_component_profiler(layer: nn.Module, qext, collector: List[dict], model_label: str, layer_idx: int, batch: int, seq_len: int):
    patched = []
    for module_name, mod in layer.named_modules():
        if not is_real_policy_linear(mod):
            continue

        def make_forward(name):
            def profiled_forward(self, x: torch.Tensor):
                device = x.device
                original_shape = x.shape[:-1]
                row = {
                    "model_label": model_label,
                    "layer_idx": int(layer_idx),
                    "batch": int(batch),
                    "seq_len": int(seq_len),
                    "module": name,
                    "K": int(self.K),
                    "N": int(self.N),
                    "R": int(getattr(self, "R", 0)),
                    "ratio": float(getattr(self, "ratio", 0.0)),
                    "is_split": bool(getattr(self, "is_split", False)),
                    "rotate_online": bool(getattr(self, "rotate_online", False)),
                    "hadamard_ms": 0.0,
                    "reshape_ms": 0.0,
                    "prepare_pack_select_ms": 0.0,
                    "dense_pack_ms": 0.0,
                    "dense_gemm_ms": 0.0,
                    "dense_scale_ms": 0.0,
                    "sparse_add_ms": 0.0,
                    "bias_view_ms": 0.0,
                    "linear_profiled_ms": 0.0,
                }

                total_start = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                total_start.record()

                if bool(getattr(self, "rotate_online", False)):
                    x, row["hadamard_ms"] = event_ms(device, lambda: self._rotate(x))

                def reshape_input():
                    A = x.reshape(-1, self.K).contiguous()
                    if A.dtype != torch.float16:
                        A = A.to(torch.float16)
                    return A

                A, row["reshape_ms"] = event_ms(device, reshape_input)
                M = int(A.shape[0])
                scratch = self.scratch_pool.get(M, self.K, self.N)
                ext = getattr(self, "ext", None) or getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError(f"{name}: cannot find ext/main_ext")

                if not bool(getattr(self, "is_split", False)):
                    _, row["dense_pack_ms"] = event_ms(device, lambda: ext.pack_a_full_s4(A, scratch["A_pack"], scratch["a_scale"], self.eps))
                    C, row["dense_gemm_ms"] = event_ms(device, lambda: V43.quarot_dense_gemm(qext, scratch["A_pack"], self.B_col, M, self.N, self.K))
                    output = torch.empty((M, self.N), dtype=torch.float16, device=device)
                    _, row["dense_scale_ms"] = event_ms(device, lambda: ext.scale_i32_to_fp16(C, scratch["a_scale"], self.w_scale, output))
                else:
                    (indices, top_q, _), row["prepare_pack_select_ms"] = event_ms(device, lambda: self._prepare_split(A, scratch))
                    C, row["dense_gemm_ms"] = event_ms(device, lambda: V43.quarot_dense_gemm(qext, scratch["A_pack"], self.B_col, M, self.N, self.K))
                    _, row["dense_scale_ms"] = event_ms(device, lambda: ext.scale_i32_to_fp16(C, scratch["body_scale"], self.w_scale, scratch["Y_body"]))
                    _, row["sparse_add_ms"] = event_ms(
                        device,
                        lambda: ext.sparse_top_add_rowmajor_quad_shared(
                            top_q,
                            indices,
                            self.B_row,
                            scratch["top_scale"],
                            self.w_scale,
                            scratch["Y_body"],
                            self.K,
                        ),
                    )
                    output = scratch["Y_body"]
                    indices.record_stream(torch.cuda.current_stream(device))
                    top_q.record_stream(torch.cuda.current_stream(device))

                def bias_view():
                    y = output
                    if self.bias is not None:
                        y = y + self.bias
                    return y.view(*original_shape, self.N)

                y, row["bias_view_ms"] = event_ms(device, bias_view)
                total_end.record()
                total_end.synchronize()
                row["linear_profiled_ms"] = float(total_start.elapsed_time(total_end))
                row["dense_component_ms"] = row["dense_gemm_ms"] + row["dense_scale_ms"]
                row["other_component_ms"] = (
                    row["hadamard_ms"] + row["reshape_ms"] + row["prepare_pack_select_ms"]
                    + row["dense_pack_ms"] + row["bias_view_ms"]
                )
                collector.append(row)
                return y

            return profiled_forward

        mod.forward = types.MethodType(make_forward(module_name), mod)
        patched.append(module_name)
    return patched


def summarize_records(records: List[dict]) -> dict:
    if not records:
        return {"split_mean_ratio": 0.0, "split_max_ratio": 0.0, "split_nonzero_modules": 0, "split_sum_R": 0}
    ratios = [float(r["ratio"]) for r in records]
    return {
        "split_mean_ratio": sum(ratios) / len(ratios),
        "split_max_ratio": max(ratios),
        "split_nonzero_modules": sum(1 for r in records if float(r["ratio"]) > 0.0),
        "split_sum_R": sum(int(r["R"]) for r in records),
    }


def summarize_components(rows: List[dict], split_total_ms: float) -> dict:
    if not rows:
        return {}
    iters = max(len(set(int(r.get("profile_iter", 0)) for r in rows)), 1)
    def s(key):
        return sum(float(r.get(key, 0.0)) for r in rows) / iters

    dense_raw = s("dense_component_ms")
    sparse_raw = s("sparse_add_ms")
    other_raw = (
        s("hadamard_ms") + s("reshape_ms") + s("prepare_pack_select_ms")
        + s("dense_pack_ms") + s("bias_view_ms")
    )
    raw_total = max(dense_raw + sparse_raw + other_raw, 1e-9)
    scale = float(split_total_ms) / raw_total

    out = {
        "split_dense_ms": dense_raw * scale,
        "split_sparse_ms": sparse_raw * scale,
        "split_other_ms": other_raw * scale,
        "split_component_scale_to_total": scale,
        "split_dense_raw_profiled_ms": dense_raw,
        "split_sparse_raw_profiled_ms": sparse_raw,
        "split_other_raw_profiled_ms": other_raw,
        "split_dense_gemm_raw_profiled_ms": s("dense_gemm_ms"),
        "split_dense_scale_raw_profiled_ms": s("dense_scale_ms"),
        "split_prepare_pack_select_raw_profiled_ms": s("prepare_pack_select_ms"),
        "split_hadamard_raw_profiled_ms": s("hadamard_ms"),
        "split_dense_pack_raw_profiled_ms": s("dense_pack_ms"),
        "split_bias_view_raw_profiled_ms": s("bias_view_ms"),
        "split_linear_raw_profiled_sum_ms": s("linear_profiled_ms"),
    }
    denom = max(float(split_total_ms), 1e-9)
    out["split_dense_pct"] = out["split_dense_ms"] / denom
    out["split_sparse_pct"] = out["split_sparse_ms"] / denom
    out["split_other_pct"] = out["split_other_ms"] / denom
    return out


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
    log("[NOTE] total timings use CUDA Graph + events. split_dense_only_ms is an ablation using the same RealPolicyLinear/QuaRot-dense wrapper with all ratios set to zero. Per-stage dense/sparse/other fields are synchronized diagnostic proportions only; use split_dense_only_ms and split_dynamic_overhead_ms for paper-level overhead claims.")

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

    quarot_model = None
    quarot_model_error = None
    if "quarot" in variants:
        try:
            quarot_model = V43.build_quarot_model(args.model, args.local_files_only)
            log("[QUAROT_MODEL] built")
        except Exception as exc:
            quarot_model_error = f"{type(exc).__name__}: {exc}"
            log(f"[QUAROT_MODEL_FALLBACK] {quarot_model_error}")

    rows = []
    linear_rows = []
    errors = []
    qlayers = V8.get_layers(quarot_model) if quarot_model is not None else None

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
                "split_component_timing": "cuda_events_synchronized_per_stage",
                "fused_topr_pack": bool(args.fused_topr_pack),
            }

            try:
                if "bf16" in variants:
                    bf16_layer = build_bf16_layer(base_layer, device)
                    row["bf16_ms"] = V43.bench_graph(lambda: V43.run_layer_once(bf16_layer, hidden_bf16, position_ids, pe_bf16), args.warmup, args.iters, device)
                    log(f"[TIME] layer={layer_idx} batch={batch} bf16_ms={row['bf16_ms']:.6f}")
                    del bf16_layer
                    torch.cuda.empty_cache()

                if "quarot" in variants:
                    if qlayers is not None:
                        qrot = quarot_model.model.rotary_emb.to(device=device, dtype=torch.float16)
                        qlayer = V43.prepare_quarot_latency_layer(qlayers[layer_idx].to(device=device, dtype=torch.float16).eval(), device)
                        qpe = tuple(t.to(dtype=torch.float16) for t in qrot(hidden_fp16, position_ids))
                        row["quarot_impl"] = "modeling_quarot"
                    else:
                        qlayer = build_quarot_linear4bit_fallback_layer(base_layer, device)
                        qpe = pe_fp16
                        row["quarot_impl"] = "linear4bit_fallback_v48"
                        row["quarot_model_error"] = quarot_model_error
                    row["quarot_ms"] = V43.bench_graph(lambda: V43.run_layer_once(qlayer, hidden_fp16, position_ids, qpe), args.warmup, args.iters, device)
                    log(f"[TIME] layer={layer_idx} batch={batch} quarot_ms={row['quarot_ms']:.6f} impl={row['quarot_impl']}")
                    qlayer.to("cpu")
                    del qlayer
                    torch.cuda.empty_cache()

                if "split" in variants:
                    pure_policy = V43.make_pure_policy(policy)
                    dense_only_layer, pure_rec = build_split_layer(base_layer, layer_idx, pure_policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, args.eps, device, qext, None, qwen_shared)
                    row.update({f"split_dense_only_{k}": v for k, v in summarize_records(pure_rec).items()})
                    row["split_dense_only_ms"] = V43.bench_graph(lambda: V43.run_layer_once(dense_only_layer, hidden_fp16, position_ids, pe_fp16), args.warmup, args.iters, device)
                    log(f"[TIME] layer={layer_idx} batch={batch} split_dense_only_ms={row['split_dense_only_ms']:.6f}")
                    del dense_only_layer
                    torch.cuda.empty_cache()

                    split_layer, rec = build_split_layer(base_layer, layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, args.eps, device, qext, fused_topr_ext, qwen_shared)
                    row.update(summarize_records(rec))
                    row["split_ms"] = V43.bench_graph(lambda: V43.run_layer_once(split_layer, hidden_fp16, position_ids, pe_fp16), args.warmup, args.iters, device)
                    log(f"[TIME] layer={layer_idx} batch={batch} split_ms={row['split_ms']:.6f}")
                    del split_layer
                    torch.cuda.empty_cache()

                    prof_layer, _ = build_split_layer(base_layer, layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, args.eps, device, qext, fused_topr_ext, qwen_shared)
                    collector: List[dict] = []
                    patched = install_split_component_profiler(prof_layer, qext, collector, args.label, layer_idx, batch, args.seq_len)
                    log(f"[PROFILE_PATCHED] layer={layer_idx} batch={batch} modules={patched}")
                    for _ in range(args.component_warmup):
                        _ = V43.run_layer_once(prof_layer, hidden_fp16, position_ids, pe_fp16)
                    torch.cuda.synchronize(device)
                    collector.clear()
                    for profile_iter in range(args.component_iters):
                        before = len(collector)
                        _ = V43.run_layer_once(prof_layer, hidden_fp16, position_ids, pe_fp16)
                        torch.cuda.synchronize(device)
                        for r in collector[before:]:
                            r["profile_iter"] = profile_iter
                            r["model"] = args.model
                            r["policy_file"] = str(Path(args.policy))
                    linear_rows.extend(collector)
                    row.update(summarize_components(collector, row["split_ms"]))
                    del prof_layer
                    torch.cuda.empty_cache()

                if row.get("bf16_ms") and row.get("split_ms"):
                    row["split_over_bf16"] = row["split_ms"] / row["bf16_ms"]
                if row.get("bf16_ms") and row.get("quarot_ms"):
                    row["quarot_over_bf16"] = row["quarot_ms"] / row["bf16_ms"]
                if row.get("quarot_ms") and row.get("split_ms"):
                    row["split_over_quarot"] = row["split_ms"] / row["quarot_ms"]
                if row.get("split_dense_only_ms") and row.get("split_ms"):
                    row["split_dynamic_overhead_ms"] = row["split_ms"] - row["split_dense_only_ms"]
                    row["split_dynamic_overhead_pct_of_split"] = row["split_dynamic_overhead_ms"] / row["split_ms"]
                    row["split_over_dense_only"] = row["split_ms"] / row["split_dense_only_ms"]
                if row.get("quarot_ms") and row.get("split_dense_only_ms"):
                    row["split_dense_only_over_quarot"] = row["split_dense_only_ms"] / row["quarot_ms"]

            except Exception as exc:
                err = {"model": args.model, "layer_idx": int(layer_idx), "batch": int(batch), "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                errors.append(err)
                row["error"] = err["error"]
                log("[ERROR] " + json.dumps(err, ensure_ascii=False))

            rows.append(row)
            write_csv(out_dir / f"{args.label}_prefill_layer_latency_v49_partial.csv", rows)
            json.dump(rows, open(out_dir / f"{args.label}_prefill_layer_latency_v49_partial.json", "w"), indent=2, ensure_ascii=False)
            del hidden_fp16, hidden_bf16, position_ids, pe_fp16, pe_bf16
            gc.collect()
            torch.cuda.empty_cache()

    layer_csv = out_dir / f"{args.label}_prefill_layer_latency_v49.csv"
    layer_json = out_dir / f"{args.label}_prefill_layer_latency_v49.json"
    linear_csv = out_dir / f"{args.label}_split_component_linear_v49.csv"
    linear_json = out_dir / f"{args.label}_split_component_linear_v49.json"
    meta_path = out_dir / f"{args.label}_prefill_layer_latency_meta_v49.json"
    write_csv(layer_csv, rows)
    json.dump(rows, open(layer_json, "w"), indent=2, ensure_ascii=False)
    write_csv(linear_csv, linear_rows)
    json.dump(linear_rows, open(linear_json, "w"), indent=2, ensure_ascii=False)
    summary = {
        "args": vars(args),
        "layer_csv": str(layer_csv),
        "linear_csv": str(linear_csv),
        "errors": errors,
        "sum_bf16_ms": sum(float(r.get("bf16_ms", 0.0)) for r in rows),
        "sum_quarot_ms": sum(float(r.get("quarot_ms", 0.0)) for r in rows),
        "sum_split_ms": sum(float(r.get("split_ms", 0.0)) for r in rows),
        "sum_split_dense_only_ms": sum(float(r.get("split_dense_only_ms", 0.0)) for r in rows),
        "sum_split_dynamic_overhead_ms": sum(float(r.get("split_dynamic_overhead_ms", 0.0)) for r in rows),
        "sum_split_dense_ms": sum(float(r.get("split_dense_ms", 0.0)) for r in rows),
        "sum_split_sparse_ms": sum(float(r.get("split_sparse_ms", 0.0)) for r in rows),
        "sum_split_other_ms": sum(float(r.get("split_other_ms", 0.0)) for r in rows),
    }
    json.dump(summary, open(meta_path, "w"), indent=2, ensure_ascii=False)
    log(f"[LAYER_CSV] {layer_csv}")
    log(f"[LINEAR_CSV] {linear_csv}")
    log(f"[META] {meta_path}")
    log(f"[SUMMARY] " + json.dumps(summary, ensure_ascii=False))
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=0")


if __name__ == "__main__":
    main()
