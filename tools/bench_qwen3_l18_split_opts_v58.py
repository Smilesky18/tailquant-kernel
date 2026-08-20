import argparse
import copy
import csv
import gc
import json
import os
import sys
import time
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
import bench_qwen3_l18_split_opts_v57 as V57
from fused_topr_pack_ext_v55 import load_fused_topr_pack_ext as load_fused_topr_pack_ext_v55
from fused_threshold_topr_pack_ext_v57 import load_threshold_topr_pack_ext
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext
from load_quarot_sm120_extension_v1 import load_quarot_sm120_extension


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--label", default="qwen3_8b_l18")
    p.add_argument("--policy", default="/data/yzy/quarot-gpt-2/experiments/kernel_quant/qwen_per_linear_diff_calibration_v6_rotate/lambda_0p08/policy.json")
    p.add_argument("--rotation_config", default="/data/yzy/quarot/qwen3-8B_layer_all.csv")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--local_files_only", action="store_true")
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


def patch_fused_sparse_epilogue(layer: nn.Module, qext, epilogue_ext, kind: str) -> List[str]:
    patched = []
    for name, mod in layer.named_modules():
        if not V43.is_real_policy_linear(mod) or not bool(getattr(mod, "is_split", False)):
            continue
        if not hasattr(mod, "B_row"):
            continue

        def make_split_compute(module_name):
            def split_compute(self, A, scratch, B_col, dense_ready_event, dense_stream, sparse_stream):
                M = int(A.shape[0])
                device = A.device
                current = torch.cuda.current_stream(device)
                indices, top_q, _ = self._prepare_split(A, scratch)

                dense_stream.wait_stream(current)
                with torch.cuda.stream(dense_stream):
                    C = V43.quarot_dense_gemm(qext, scratch["A_pack"], B_col, M, self.N, self.K)

                sparse_stream.wait_stream(dense_stream)
                with torch.cuda.stream(sparse_stream):
                    if kind == "quad":
                        epilogue_ext.scale_sparse_epilogue_quad(
                            C,
                            scratch["body_scale"],
                            top_q,
                            indices,
                            self.B_row,
                            scratch["top_scale"],
                            self.w_scale,
                            scratch["Y_body"],
                            self.K,
                        )
                    elif kind == "oct":
                        epilogue_ext.scale_sparse_epilogue_oct(
                            C,
                            scratch["body_scale"],
                            top_q,
                            indices,
                            self.B_row,
                            scratch["top_scale"],
                            self.w_scale,
                            scratch["Y_body"],
                            self.K,
                        )
                    else:
                        raise ValueError(kind)

                indices.record_stream(sparse_stream)
                top_q.record_stream(sparse_stream)
                C.record_stream(sparse_stream)
                current.wait_stream(sparse_stream)
                return scratch["Y_body"]

            return split_compute

        mod._split_compute = types.MethodType(make_split_compute(name), mod)
        mod._fused_sparse_epilogue_v58 = kind
        patched.append({"name": name, "K": int(mod.K), "N": int(mod.N), "R": int(mod.R), "kind": kind})
    log("[PATCH_FUSED_SPARSE_EPILOGUE_V58] " + json.dumps(patched, indent=2))
    return [p["name"] for p in patched]


def policy_key(layer_idx: int, local_name: str) -> str:
    return f"model.layers.{layer_idx}.{local_name}"


def iter_target_linears(layer: nn.Module):
    return V29.iter_target_linears(layer)


def make_up_type_weight_prerotated_layer(base_layer: nn.Module, layer_idx: int, rot_flags: dict, B, device: torch.device):
    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    new_flags = dict(rot_flags)
    patched = []
    up_type_suffixes = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
    }
    had_cache = {}

    with torch.no_grad():
        for _, _, local_name, linear in iter_target_linears(layer):
            if local_name not in up_type_suffixes:
                continue
            full_name = policy_key(layer_idx, local_name)
            if not bool(rot_flags.get(full_name, False)):
                continue
            k = int(linear.in_features)
            if k not in had_cache:
                had_cache[k] = B.H.hadamard_utils.get_hadK(k)
            had_k, had_factor = had_cache[k]
            had_k = had_k.detach().to(device=device, dtype=torch.float32).contiguous()
            rotated = B.H.apply_hadamard_last_dim(
                linear.weight.detach().to(device=device, dtype=torch.float16),
                had_k,
                had_factor,
            )
            linear.weight.data.copy_(rotated.to(dtype=linear.weight.dtype))
            new_flags[full_name] = False
            patched.append({"name": full_name, "K": k, "N": int(linear.out_features)})

    log("[PATCH_WEIGHT_PREROTATE_UPTYPE_V58] " + json.dumps(patched, indent=2))
    return layer.to("cpu"), new_flags, patched


def build_layer(base_layer, layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, eps, device, qext, top_ext, epilogue_ext, gateup: bool, epilogue_kind: str | None):
    layer, rec = V43.build_split_layer_with_hadamard(
        base_layer,
        layer_idx,
        policy,
        rot_flags,
        B,
        main_ext,
        layout_ext,
        policy_pack_ext,
        eps,
        device,
        qext,
        fused_topr_ext=top_ext,
    )
    if gateup:
        V57.patch_gate_up_shared_hadamard(layer)
    if epilogue_kind:
        patch_fused_sparse_epilogue(layer, qext, epilogue_ext, epilogue_kind)
    return layer, rec


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log("[NOTE] v58 benchmark: v57 threshold frontend plus fused dense-scale/sparse-correction epilogue.")
    log(f"[MODEL] {args.model} [LAYER] {args.layer} [BATCHES] {args.batches}")

    V29.install_qfactory_fast_preset("qwen3_sm120_v1")
    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)
    policy = V29.load_policy(Path(args.policy), False)
    qext = load_quarot_sm120_extension(verbose=bool(int(os.environ.get("QUAROT_SM120_VERBOSE", "0"))))
    fused_v55 = load_fused_topr_pack_ext_v55(verbose=bool(int(os.environ.get("FUSED_TOPR_VERBOSE", "0"))))
    threshold_v57 = load_threshold_topr_pack_ext(verbose=bool(int(os.environ.get("FUSED_TOPR_VERBOSE", "0"))))
    epilogue_v58 = load_fused_sparse_epilogue_ext(verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0"))))
    log(f"[QUAROT_DENSE_EXT] {getattr(qext, chr(95)+chr(95)+'file'+chr(95)+chr(95), qext)}")
    log(f"[FUSED_V55] {getattr(fused_v55, chr(95)+chr(95)+'file'+chr(95)+chr(95), fused_v55)}")
    log(f"[THRESHOLD_V57] {getattr(threshold_v57, chr(95)+chr(95)+'file'+chr(95)+chr(95), threshold_v57)}")
    log(f"[EPILOGUE_V58] {getattr(epilogue_v58, chr(95)+chr(95)+'file'+chr(95)+chr(95), epilogue_v58)}")

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
    rot_flags = B.H.build_rotation_flags(model, args.rotation_config)

    variants = [
        ("fused512_v55", fused_v55, "v55", False, None),
        ("threshold_v57_gateup_shared", threshold_v57, "v57", True, None),
        ("threshold_v57_epilogue_quad", threshold_v57, "v57", False, "quad"),
        ("threshold_v57_epilogue_oct", threshold_v57, "v57", False, "oct"),
        ("threshold_v57_gateup_epilogue_oct", threshold_v57, "v57", True, "oct"),
        ("threshold_v57_prerot_up_epilogue_oct", threshold_v57, "v57", False, "oct"),
    ]
    original_patch = V43.patch_fused_topr_prepare

    rows = []
    errors = []
    for batch in parse_ints(args.batches):
        hidden_fp16 = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=torch.float16)
        position_ids = V8.make_position_ids(batch, args.seq_len, device)
        pe_fp16 = V8.build_position_embeddings(model, hidden_fp16, position_ids, torch.float16)

        for variant, top_ext, patch_kind, gateup, epilogue_kind in variants:
            if patch_kind == "v57":
                V57.install_threshold_topr_prepare_v57()
            else:
                V57.install_fused_topr_prepare_v55()
            log(f"[CASE] batch={batch} variant={variant}")
            row = {
                "model": args.model,
                "model_label": args.label,
                "layer_idx": int(args.layer),
                "batch": int(batch),
                "seq_len": int(args.seq_len),
                "variant": variant,
                "timing": "cuda_graph_events",
                "topr_backend": "v57_threshold_K4096_12288_R512" if patch_kind == "v57" else "v55_K4096_R512",
                "gate_up_shared_hadamard": bool(gateup),
                "fused_sparse_epilogue": epilogue_kind or "none",
                "weight_prerotate_up_type": bool("prerot_up" in variant),
            }
            try:
                local_base_layer = base_layer
                local_rot_flags = rot_flags
                if "prerot_up" in variant:
                    local_base_layer, local_rot_flags, pre_records = make_up_type_weight_prerotated_layer(
                        base_layer,
                        int(args.layer),
                        rot_flags,
                        B,
                        device,
                    )
                    row["weight_prerotate_modules"] = len(pre_records)
                layer, rec = build_layer(
                    local_base_layer,
                    int(args.layer),
                    policy,
                    local_rot_flags,
                    B,
                    main_ext,
                    layout_ext,
                    policy_pack_ext,
                    args.eps,
                    device,
                    qext,
                    top_ext,
                    epilogue_v58,
                    gateup,
                    epilogue_kind,
                )
                row.update(V57.summarize_records(rec))
                row["split_ms"] = V43.bench_graph(lambda: V43.run_layer_once(layer, hidden_fp16, position_ids, pe_fp16), args.warmup, args.iters, device)
                log(f"[TIME] batch={batch} variant={variant} split_ms={row['split_ms']:.6f}")
                layer.to("cpu")
                del layer
            except Exception as exc:
                err = {"batch": batch, "variant": variant, "error": f"{type(exc).__name__}: {exc}"}
                errors.append(err)
                row["error"] = err["error"]
                log("[ERROR] " + json.dumps(err))
            rows.append(row)
            write_csv(out_dir / f"{args.label}_split_opts_v58_partial.csv", rows)
            json.dump(rows, open(out_dir / f"{args.label}_split_opts_v58_partial.json", "w"), indent=2)
            gc.collect()
            torch.cuda.empty_cache()

        del hidden_fp16, position_ids, pe_fp16
        gc.collect()
        torch.cuda.empty_cache()

    for batch in parse_ints(args.batches):
        base_rows = [r for r in rows if r.get("batch") == batch and r.get("variant") == "threshold_v57_gateup_shared" and "split_ms" in r]
        if not base_rows:
            continue
        base = float(base_rows[0]["split_ms"])
        for r in rows:
            if r.get("batch") == batch and "split_ms" in r:
                r["speedup_over_threshold_v57_gateup"] = base / float(r["split_ms"]) if float(r["split_ms"]) > 0 else 0.0
                r["delta_vs_threshold_v57_gateup_ms"] = float(r["split_ms"]) - base

    csv_path = out_dir / f"{args.label}_split_opts_v58.csv"
    json_path = out_dir / f"{args.label}_split_opts_v58.json"
    meta_path = out_dir / f"{args.label}_split_opts_meta_v58.json"
    write_csv(csv_path, rows)
    json.dump(rows, open(json_path, "w"), indent=2)
    json.dump({"args": vars(args), "csv": str(csv_path), "json": str(json_path), "errors": errors, "rows": rows}, open(meta_path, "w"), indent=2)
    V43.patch_fused_topr_prepare = original_patch
    log(f"[CSV] {csv_path}")
    log(f"[META] {meta_path}")
    log("[SUMMARY] " + json.dumps({"errors": errors, "rows": rows}, ensure_ascii=False))
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=0")


if __name__ == "__main__":
    main()
