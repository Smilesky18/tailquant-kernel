import argparse
import csv
import gc
import json
import os
import time
import types
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8
import bench_multimodel_all_layers_policy_fastqf_v29 as V29
import bench_hadamard_three_schemes_quarot_dense_v36 as V36
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
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--component_warmup", type=int, default=2)
    p.add_argument("--component_iters", type=int, default=5)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--force_activation_percentile_100", action="store_true")
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
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


def cuda_elapsed_ms(device: torch.device, fn):
    torch.cuda.synchronize(device)
    stream = torch.cuda.current_stream(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    out = fn()
    end.record(stream)
    end.synchronize()
    return out, float(start.elapsed_time(end))


def is_real_policy_linear(m) -> bool:
    name = m.__class__.__name__
    return name == "RealPolicyLinear" or "RealPolicyLinear" in name


def install_component_profiler(layer, qext, collector: List[dict], model_label: str, layer_idx: int, batch: int, seq_len: int):
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
                    "mode": str(getattr(self, "mode", "")),
                    "rotate_online": bool(getattr(self, "rotate_online", False)),
                    "hadamard_ms": 0.0,
                    "input_reshape_ms": 0.0,
                    "top_r_select_ms": 0.0,
                    "pack_quant_scale_ms": 0.0,
                    "dense_gemm_ms": 0.0,
                    "dense_scale_ms": 0.0,
                    "sparse_zero_ms": 0.0,
                    "sparse_add_ms": 0.0,
                    "output_add_ms": 0.0,
                    "bias_view_ms": 0.0,
                    "linear_total_profiled_ms": 0.0,
                }

                total_start = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                total_start.record()

                if bool(getattr(self, "rotate_online", False)):
                    x, row["hadamard_ms"] = cuda_elapsed_ms(device, lambda: self._rotate(x))

                def reshape_input():
                    A_local = x.reshape(-1, self.K).contiguous()
                    if A_local.dtype != torch.float16:
                        A_local = A_local.to(torch.float16)
                    return A_local

                A, row["input_reshape_ms"] = cuda_elapsed_ms(device, reshape_input)
                M = int(A.shape[0])
                scratch = self.scratch_pool.get(M, self.K, self.N)
                ext = getattr(self, "ext", None) or getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError(f"{name}: cannot find ext/main_ext")

                if not bool(getattr(self, "is_split", False)):
                    _, row["pack_quant_scale_ms"] = cuda_elapsed_ms(
                        device,
                        lambda: ext.pack_a_full_s4(A, scratch["A_pack"], scratch["a_scale"], self.eps),
                    )
                    C, row["dense_gemm_ms"] = cuda_elapsed_ms(
                        device,
                        lambda: V36.quarot_dense_gemm(qext, scratch["A_pack"], self.B_col, M, self.N, self.K),
                    )
                    output = torch.empty((M, self.N), dtype=torch.float16, device=device)
                    _, row["dense_scale_ms"] = cuda_elapsed_ms(
                        device,
                        lambda: ext.scale_i32_to_fp16(C, scratch["a_scale"], self.w_scale, output),
                    )
                else:
                    R = int(self.R)
                    percentile = min(max(float(self.activation_percentile), 0.0), 100.0)
                    body_len = int(self.K) - R
                    import math
                    body_kth = min(int(self.K), max(1, int(math.ceil(body_len * percentile / 100.0))))
                    descending_rank = int(self.K) - body_kth + 1
                    select_k = max(R, descending_rank)

                    def select_top_r():
                        abs_A = A.abs().float()
                        top_values, top_indices = torch.topk(abs_A, k=select_k, dim=1, largest=True, sorted=True)
                        body_threshold = top_values[:, descending_rank - 1].contiguous()
                        tail_threshold = top_values[:, 0].contiguous()
                        tail_indices = top_indices[:, :R]
                        tail_indices, _ = torch.sort(tail_indices, dim=1)
                        tail_indices = tail_indices.to(torch.int32).contiguous()
                        return tail_indices, body_threshold, tail_threshold

                    selected, row["top_r_select_ms"] = cuda_elapsed_ms(device, select_top_r)
                    indices, body_threshold, tail_threshold = selected
                    quant_buffers = self.scratch_pool.get_quant_buffers(M, self.K, self.N, R)
                    top_q = quant_buffers["top_q"]
                    body_q_top = quant_buffers["body_q_top"]

                    def policy_pack():
                        self.policy_pack_ext.pack_policy_split(
                            A,
                            indices,
                            body_threshold,
                            tail_threshold,
                            scratch["A_pack"],
                            scratch["body_scale"],
                            scratch["top_scale"],
                            top_q,
                            self.eps,
                        )
                        body_q_top.zero_()

                    _, row["pack_quant_scale_ms"] = cuda_elapsed_ms(device, policy_pack)
                    C, row["dense_gemm_ms"] = cuda_elapsed_ms(
                        device,
                        lambda: V36.quarot_dense_gemm(qext, scratch["A_pack"], self.B_col, M, self.N, self.K),
                    )
                    _, row["dense_scale_ms"] = cuda_elapsed_ms(
                        device,
                        lambda: ext.scale_i32_to_fp16(C, scratch["body_scale"], self.w_scale, scratch["Y_body"]),
                    )
                    _, row["sparse_zero_ms"] = cuda_elapsed_ms(device, lambda: scratch["Y_sparse"].zero_())
                    _, row["sparse_add_ms"] = cuda_elapsed_ms(
                        device,
                        lambda: ext.sparse_top_add_rowmajor_quad_shared(
                            top_q,
                            indices,
                            self.B_row,
                            scratch["top_scale"],
                            self.w_scale,
                            scratch["Y_sparse"],
                            self.K,
                        ),
                    )
                    output = torch.empty((M, self.N), dtype=torch.float16, device=device)
                    _, row["output_add_ms"] = cuda_elapsed_ms(
                        device,
                        lambda: torch.add(scratch["Y_body"], scratch["Y_sparse"], out=output),
                    )
                    indices.record_stream(torch.cuda.current_stream(device))
                    top_q.record_stream(torch.cuda.current_stream(device))

                def bias_view():
                    y = output
                    if self.bias is not None:
                        y = y + self.bias
                    return y.view(*original_shape, self.N)

                y, row["bias_view_ms"] = cuda_elapsed_ms(device, bias_view)
                total_end.record()
                total_end.synchronize()
                row["linear_total_profiled_ms"] = float(total_start.elapsed_time(total_end))
                row["dense_compute_ms"] = row["dense_gemm_ms"] + row["dense_scale_ms"]
                row["sparse_compute_ms"] = row["sparse_zero_ms"] + row["sparse_add_ms"]
                row["measured_component_sum_ms"] = (
                    row["hadamard_ms"]
                    + row["input_reshape_ms"]
                    + row["top_r_select_ms"]
                    + row["pack_quant_scale_ms"]
                    + row["dense_compute_ms"]
                    + row["sparse_compute_ms"]
                    + row["output_add_ms"]
                    + row["bias_view_ms"]
                )
                row["other_profiled_ms"] = max(0.0, row["linear_total_profiled_ms"] - row["measured_component_sum_ms"])
                collector.append(row)
                return y

            return profiled_forward

        mod.forward = types.MethodType(make_forward(module_name), mod)
        patched.append(module_name)

    return patched


def summarize_layer_components(rows: List[dict], layer_total_ms: float) -> dict:
    keys = [
        "hadamard_ms",
        "input_reshape_ms",
        "top_r_select_ms",
        "pack_quant_scale_ms",
        "dense_compute_ms",
        "dense_gemm_ms",
        "dense_scale_ms",
        "sparse_compute_ms",
        "sparse_zero_ms",
        "sparse_add_ms",
        "output_add_ms",
        "bias_view_ms",
        "other_profiled_ms",
        "linear_total_profiled_ms",
        "measured_component_sum_ms",
    ]
    iters = max(len(set(r.get("profile_iter", 0) for r in rows)), 1)
    out = {k: sum(float(r.get(k, 0.0)) for r in rows) / iters for k in keys}
    out["layer_total_ms"] = float(layer_total_ms)
    first_iter = [r for r in rows if int(r.get("profile_iter", 0)) == 0]
    out["sum_R"] = int(sum(int(r.get("R", 0)) for r in first_iter))
    out["nonzero_linears"] = int(sum(1 for r in first_iter if float(r.get("ratio", 0.0)) > 0.0))
    ratios = [float(r.get("ratio", 0.0)) for r in first_iter]
    out["mean_ratio"] = sum(ratios) / max(len(ratios), 1)
    out["max_ratio"] = max(ratios) if ratios else 0.0
    denom = max(out["layer_total_ms"], 1e-9)
    for k in ["hadamard_ms", "top_r_select_ms", "pack_quant_scale_ms", "dense_compute_ms", "sparse_compute_ms", "sparse_zero_ms", "sparse_add_ms", "output_add_ms", "other_profiled_ms"]:
        out[k.replace("_ms", "_pct_of_layer_total")] = out[k] / denom
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LABEL] {args.label}")
    log(f"[POLICY] {args.policy}")
    log(f"[ROTATION_CONFIG] {args.rotation_config}")
    log("[NOTE] layer_total_ms uses CUDA Graph on v36 async split with QuaRot dense backend; component_ms uses synchronized per-stage diagnostic timing.")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)
    qext = load_quarot_sm120_extension(verbose=bool(int(os.environ.get("QUAROT_SM120_VERBOSE", "0"))))
    log(f"[QUAROT_DENSE_EXT] {getattr(qext, chr(95)+chr(95)+'file'+chr(95)+chr(95), qext)}")
    policy = V29.load_policy(Path(args.policy), args.force_activation_percentile_100)

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
    rot_flags = B.H.build_rotation_flags(model, args.rotation_config)
    log(f"[SPLIT_ROT_FLAGS] selected={sum(bool(v) for v in rot_flags.values())}/{len(rot_flags)}")

    linear_rows = []
    layer_rows = []

    for layer_idx in layer_indices:
        base_layer = layers[layer_idx]
        for batch in batches:
            log(f"[CASE] label={args.label} layer={layer_idx} batch={batch}")
            hidden = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=torch.float16)
            position_ids = V8.make_position_ids(batch, args.seq_len, device)
            pe = V8.build_position_embeddings(model, hidden, position_ids, torch.float16)

            total_layer, _ = V36.build_split_layer_with_hadamard(
                base_layer, layer_idx, policy, rot_flags, B,
                main_ext, layout_ext, policy_pack_ext, args.eps, device, qext,
            )
            total_ms = V36.bench_graph(
                lambda: V36.run_layer_once(total_layer, hidden, position_ids, pe),
                args.warmup,
                args.iters,
                device,
            )
            del total_layer
            torch.cuda.empty_cache()

            prof_layer, _ = V36.build_split_layer_with_hadamard(
                base_layer, layer_idx, policy, rot_flags, B,
                main_ext, layout_ext, policy_pack_ext, args.eps, device, qext,
            )
            collector: List[dict] = []
            patched = install_component_profiler(prof_layer, qext, collector, args.label, layer_idx, batch, args.seq_len)
            log(f"[PROFILE_PATCHED] {patched}")

            for _ in range(args.component_warmup):
                _ = V36.run_layer_once(prof_layer, hidden, position_ids, pe)
            torch.cuda.synchronize(device)
            collector.clear()

            for profile_iter in range(args.component_iters):
                before = len(collector)
                _ = V36.run_layer_once(prof_layer, hidden, position_ids, pe)
                torch.cuda.synchronize(device)
                for r in collector[before:]:
                    r["profile_iter"] = profile_iter

            case_rows = [r for r in collector if int(r.get("layer_idx", -1)) == layer_idx and int(r.get("batch", -1)) == batch]
            summary = summarize_layer_components(case_rows, total_ms)
            summary.update({
                "model": args.model,
                "model_label": args.label,
                "layer_idx": int(layer_idx),
                "batch": int(batch),
                "seq_len": int(args.seq_len),
                "hidden_size": int(hidden_size),
                "policy_file": str(Path(args.policy)),
                "rotation_config": str(Path(args.rotation_config)),
                "timing_total": "cuda_graph_events_v36_async",
                "timing_components": "cuda_events_synchronized_per_stage",
                "component_iters": int(args.component_iters),
            })
            layer_rows.append(summary)
            linear_rows.extend(case_rows)
            log("[LAYER_SUMMARY] " + json.dumps(summary, ensure_ascii=False))

            del prof_layer, hidden, position_ids, pe
            gc.collect()
            torch.cuda.empty_cache()

    linear_csv = out_dir / f"{args.label}_split_component_linear_quarot_dense_v37.csv"
    layer_csv = out_dir / f"{args.label}_split_component_layer_quarot_dense_v37.csv"
    linear_json = out_dir / f"{args.label}_split_component_linear_quarot_dense_v37.json"
    layer_json = out_dir / f"{args.label}_split_component_layer_quarot_dense_v37.json"
    write_csv(linear_csv, linear_rows)
    write_csv(layer_csv, layer_rows)
    json.dump(linear_rows, open(linear_json, "w"), indent=2, ensure_ascii=False)
    json.dump(layer_rows, open(layer_json, "w"), indent=2, ensure_ascii=False)
    json.dump({"args": vars(args), "linear_csv": str(linear_csv), "layer_csv": str(layer_csv)}, open(out_dir / f"{args.label}_split_component_meta_quarot_dense_v37.json", "w"), indent=2, ensure_ascii=False)
    log(f"[LINEAR_CSV] {linear_csv}")
    log(f"[LAYER_CSV] {layer_csv}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=0")


if __name__ == "__main__":
    main()
