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
from fused_topr_pack_ext_v42 import load_fused_topr_pack_ext as load_fused_topr_pack_ext_v42
from fused_topr_pack_ext_v55 import load_fused_topr_pack_ext as load_fused_topr_pack_ext_v55
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


def policy_key(layer_idx: int, local_name: str) -> str:
    return f"model.layers.{layer_idx}.{local_name}"


def make_romeo_style_rot_flags(rot_flags: Dict[str, bool]) -> Dict[str, bool]:
    out = {}
    for key in rot_flags:
        out[key] = key.endswith(".self_attn.o_proj") or key.endswith(".mlp.down_proj")
    return out


def install_fused_topr_prepare_v55():
    def _split_select_params(mod):
        import math
        K = int(mod.K)
        R = int(getattr(mod, "R", 0))
        percentile = min(max(float(getattr(mod, "activation_percentile", 100.0)), 0.0), 100.0)
        body_len = K - R
        body_kth = min(K, max(1, int(math.ceil(body_len * percentile / 100.0))))
        descending_rank = K - body_kth + 1
        select_k = max(R, descending_rank)
        return R, descending_rank, select_k

    def patch_fused_topr_prepare_v55(mod: nn.Module, fused_topr_ext) -> bool:
        if getattr(mod, "_fused_topr_prepare_v55", False):
            return False

        orig_prepare = mod._prepare_split
        idx_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

        def fused_prepare(self, A: torch.Tensor, scratch: Dict[str, torch.Tensor]):
            M, K = int(A.shape[0]), int(A.shape[1])
            R, descending_rank, _ = _split_select_params(self)
            if K != 4096 or R <= 0 or R > 512 or A.dtype != torch.float16 or not A.is_contiguous():
                return orig_prepare(A, scratch)

            quant_buffers = self.scratch_pool.get_quant_buffers(M, self.K, self.N, R)
            tail_q = quant_buffers["top_q"]
            body_q_top = quant_buffers["body_q_top"]

            device_index = A.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            key = (M, K, R, int(device_index))
            tail_indices = idx_cache.get(key)
            if tail_indices is None:
                tail_indices = torch.empty((M, R), dtype=torch.int32, device=A.device)
                idx_cache[key] = tail_indices

            fused_topr_ext.fused_topr_pack(
                A,
                R,
                descending_rank,
                scratch["A_pack"],
                scratch["body_scale"],
                scratch["top_scale"],
                tail_q,
                tail_indices,
                self.eps,
            )
            body_q_top.zero_()
            return tail_indices, tail_q, body_q_top

        mod._prepare_split = types.MethodType(fused_prepare, mod)
        mod._fused_topr_prepare_v55 = True
        mod._fused_topr_idx_cache_v55 = idx_cache
        return True

    V43.patch_fused_topr_prepare = patch_fused_topr_prepare_v55
    log("[INSTALL_FUSED_TOPR_PREPARE_V55] K=4096 R<=512")


def build_split_layer(base_layer, layer_idx, policy, rot_flags, B, main_ext, layout_ext, policy_pack_ext, eps, device, qext, fused_topr_ext):
    return V43.build_split_layer_with_hadamard(
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
        fused_topr_ext=fused_topr_ext,
    )


def summarize_records(records: List[dict]) -> dict:
    ratios = [float(r["ratio"]) for r in records]
    return {
        "split_mean_ratio": sum(ratios) / max(len(ratios), 1),
        "split_max_ratio": max(ratios) if ratios else 0.0,
        "split_nonzero_modules": sum(1 for r in records if float(r["ratio"]) > 0.0),
        "split_sum_R": sum(int(r["R"]) for r in records),
        "fused_eligible_modules": sum(1 for r in records if int(r["K"]) == 4096 and 0 < int(r["R"]) <= 512),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[NOTE] v55 Qwen3 L18 split optimization benchmark. romeo_style rotation is latency ablation only unless corresponding weight pre-rotation is added.")
    log(f"[MODEL] {args.model} [LAYER] {args.layer} [BATCHES] {args.batches}")

    V29.install_qfactory_fast_preset("qwen3_sm120_v1")
    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)
    policy = V29.load_policy(Path(args.policy), False)
    qext = load_quarot_sm120_extension(verbose=bool(int(os.environ.get("QUAROT_SM120_VERBOSE", "0"))))
    fused_v42 = load_fused_topr_pack_ext_v42(verbose=bool(int(os.environ.get("FUSED_TOPR_VERBOSE", "0"))))
    fused_v55 = load_fused_topr_pack_ext_v55(verbose=bool(int(os.environ.get("FUSED_TOPR_VERBOSE", "0"))))
    log(f"[QUAROT_DENSE_EXT] {getattr(qext, chr(95)+chr(95)+'file'+chr(95)+chr(95), qext)}")
    log(f"[FUSED_V42] {getattr(fused_v42, chr(95)+chr(95)+'file'+chr(95)+chr(95), fused_v42)}")
    log(f"[FUSED_V55] {getattr(fused_v55, chr(95)+chr(95)+'file'+chr(95)+chr(95), fused_v55)}")

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
    rot_flags_original = B.H.build_rotation_flags(model, args.rotation_config)
    rot_flags_romeo = make_romeo_style_rot_flags(rot_flags_original)
    log(f"[ROT_ORIGINAL] selected={sum(bool(v) for v in rot_flags_original.values())}/{len(rot_flags_original)}")
    log(f"[ROT_ROMEO_STYLE_ABLATION] selected={sum(bool(v) for v in rot_flags_romeo.values())}/{len(rot_flags_romeo)}")

    variants = [
        ("baseline_v42", rot_flags_original, fused_v42, False),
        ("romeo_rotate_ablation_v42", rot_flags_romeo, fused_v42, False),
        ("fused512_v55", rot_flags_original, fused_v55, True),
        ("combined_ablation_v55", rot_flags_romeo, fused_v55, True),
    ]
    original_patch_fused_topr_prepare = V43.patch_fused_topr_prepare

    rows = []
    for batch in parse_ints(args.batches):
        hidden_fp16 = torch.randn(batch, args.seq_len, hidden_size, device=device, dtype=torch.float16)
        position_ids = V8.make_position_ids(batch, args.seq_len, device)
        pe_fp16 = V8.build_position_embeddings(model, hidden_fp16, position_ids, torch.float16)

        for variant, rot_flags, fused_ext, use_v55_patch in variants:
            if use_v55_patch:
                install_fused_topr_prepare_v55()
            else:
                V43.patch_fused_topr_prepare = original_patch_fused_topr_prepare
            log(f"[CASE] batch={batch} variant={variant}")
            layer, rec = build_split_layer(
                base_layer,
                int(args.layer),
                policy,
                rot_flags,
                B,
                main_ext,
                layout_ext,
                policy_pack_ext,
                args.eps,
                device,
                qext,
                fused_ext,
            )
            row = {
                "model": args.model,
                "model_label": args.label,
                "layer_idx": int(args.layer),
                "batch": int(batch),
                "seq_len": int(args.seq_len),
                "variant": variant,
                "timing": "cuda_graph_events",
                "rotation_mode": "romeo_style_latency_ablation" if rot_flags is rot_flags_romeo else "original_all_selected",
                "fused_topr": "v55_K4096_R512" if use_v55_patch else "v42_K4096_R256",
            }
            row.update(summarize_records(rec))
            row["split_ms"] = V43.bench_graph(lambda: V43.run_layer_once(layer, hidden_fp16, position_ids, pe_fp16), args.warmup, args.iters, device)
            log(f"[TIME] batch={batch} variant={variant} split_ms={row['split_ms']:.6f}")
            rows.append(row)
            write_csv(out_dir / f"{args.label}_split_opts_v55_partial.csv", rows)
            json.dump(rows, open(out_dir / f"{args.label}_split_opts_v55_partial.json", "w"), indent=2)
            layer.to("cpu")
            del layer
            torch.cuda.empty_cache()

        del hidden_fp16, position_ids, pe_fp16
        gc.collect()
        torch.cuda.empty_cache()

    # Add speedup columns relative to baseline in each batch.
    for batch in parse_ints(args.batches):
        base = next(r["split_ms"] for r in rows if r["batch"] == batch and r["variant"] == "baseline_v42")
        for r in rows:
            if r["batch"] == batch:
                r["speedup_over_baseline"] = base / r["split_ms"]
                r["delta_vs_baseline_ms"] = r["split_ms"] - base

    csv_path = out_dir / f"{args.label}_split_opts_v55.csv"
    json_path = out_dir / f"{args.label}_split_opts_v55.json"
    meta_path = out_dir / f"{args.label}_split_opts_meta_v55.json"
    write_csv(csv_path, rows)
    json.dump(rows, open(json_path, "w"), indent=2)
    meta = {"args": vars(args), "csv": str(csv_path), "json": str(json_path), "rows": rows}
    json.dump(meta, open(meta_path, "w"), indent=2)
    log(f"[CSV] {csv_path}")
    log(f"[META] {meta_path}")
    log("[SUMMARY] " + json.dumps(meta, ensure_ascii=False))
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc=0")


if __name__ == "__main__":
    main()
