import argparse
import copy
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import bench_full_qwen3_8b_layer_latency_policy_projected_v22b as V22B


TARGET_LINEAR_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def log(x):
    print(x, flush=True)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--policy", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--layers", default="all")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    return p.parse_args()


def get_submodule(root, name):
    m = root
    for p in name.split("."):
        m = getattr(m, p)
    return m


def ext_of(m):
    ext = getattr(m, "ext", None)
    if ext is None:
        ext = getattr(m, "main_ext", None)
    if ext is None:
        raise RuntimeError("cannot find ext/main_ext")
    return ext


def bench(fn, warmup, iters, device):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)

        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        ed.record()
        torch.cuda.synchronize(device)
        return float(st.elapsed_time(ed) / iters)


def make_fixed_allrows(indices):
    return indices[:1].repeat(indices.shape[0], 1).contiguous()


def make_block_fixed(indices, block=16):
    out = indices.clone()
    M = indices.shape[0]
    for s in range(0, M, block):
        e = min(M, s + block)
        out[s:e] = indices[s:s+1]
    return out.contiguous()


def sparse_add(linear, top_q, indices, top_scale, Y):
    ext = ext_of(linear)
    ext.sparse_top_add_rowmajor_quad_shared(
        top_q,
        indices,
        linear.B_row,
        top_scale,
        linear.w_scale,
        Y,
        linear.K,
    )


def measure_one(linear, batch, seq_len, warmup, iters, device):
    M = batch * seq_len
    K = int(linear.K)
    N = int(linear.N)
    R = int(getattr(linear, "R", 0))
    ratio = R / K if K > 0 else 0.0

    if R <= 0:
        return None

    A = torch.randn((M, K), device=device, dtype=torch.float16)
    scratch = linear.scratch_pool.get(M, K, N)

    with torch.inference_mode():
        indices, top_q, _ = linear._prepare_split(A, scratch)
        torch.cuda.synchronize(device)

    top_scale = scratch["top_scale"]
    Y_sparse = scratch["Y_sparse"]
    Y_body = scratch["Y_body"]
    Y_out = torch.empty_like(Y_sparse)

    indices_fixed = make_fixed_allrows(indices)
    indices_block16 = make_block_fixed(indices, block=16)

    zero_ms = bench(lambda: Y_sparse.zero_(), warmup, iters, device)

    sparse_random_add_only_ms = bench(
        lambda: sparse_add(linear, top_q, indices, top_scale, Y_sparse),
        warmup,
        iters,
        device,
    )

    def zero_plus_sparse_random():
        Y_sparse.zero_()
        sparse_add(linear, top_q, indices, top_scale, Y_sparse)

    sparse_random_total_ms = bench(zero_plus_sparse_random, warmup, iters, device)

    sparse_fixed_allrows_add_only_ms = bench(
        lambda: sparse_add(linear, top_q, indices_fixed, top_scale, Y_sparse),
        warmup,
        iters,
        device,
    )

    sparse_block16_fixed_add_only_ms = bench(
        lambda: sparse_add(linear, top_q, indices_block16, top_scale, Y_sparse),
        warmup,
        iters,
        device,
    )

    merge_ms = bench(
        lambda: torch.add(Y_body, Y_sparse, out=Y_out),
        warmup,
        iters,
        device,
    )

    # 粗略下界估计。真实 bytes 会因 cache、unpack、scale、write policy 有差异。
    b_weight_bytes_lower = M * R * N * 0.5
    topq_bytes = top_q.numel() * top_q.element_size()
    indices_bytes = indices.numel() * indices.element_size()
    scale_bytes = top_scale.numel() * top_scale.element_size() + linear.w_scale.numel() * linear.w_scale.element_size()
    y_bytes = M * N * 2
    merge_bytes_lower = 3 * y_bytes
    zero_bytes_lower = y_bytes

    ops_int4_mac = 2.0 * M * R * N

    def gops(ms):
        return ops_int4_mac / (ms / 1000.0) / 1e9 if ms > 0 else 0.0

    def gbps(bytes_, ms):
        return bytes_ / (ms / 1000.0) / 1e9 if ms > 0 else 0.0

    row = {
        "batch": batch,
        "seq_len": seq_len,
        "M": M,
        "K": K,
        "N": N,
        "R": R,
        "ratio": ratio,

        "zero_ms": zero_ms,
        "sparse_random_add_only_ms": sparse_random_add_only_ms,
        "sparse_random_total_ms": sparse_random_total_ms,
        "sparse_fixed_allrows_add_only_ms": sparse_fixed_allrows_add_only_ms,
        "sparse_block16_fixed_add_only_ms": sparse_block16_fixed_add_only_ms,
        "merge_ms": merge_ms,

        "fixed_allrows_speedup_over_random": sparse_random_add_only_ms / sparse_fixed_allrows_add_only_ms if sparse_fixed_allrows_add_only_ms > 0 else 0.0,
        "block16_fixed_speedup_over_random": sparse_random_add_only_ms / sparse_block16_fixed_add_only_ms if sparse_block16_fixed_add_only_ms > 0 else 0.0,

        "estimated_weight_bytes_lower_GB": b_weight_bytes_lower / 1e9,
        "estimated_topq_bytes_MB": topq_bytes / 1e6,
        "estimated_indices_bytes_MB": indices_bytes / 1e6,
        "estimated_scale_bytes_MB": scale_bytes / 1e6,
        "estimated_y_bytes_MB": y_bytes / 1e6,
        "estimated_zero_bytes_MB": zero_bytes_lower / 1e6,
        "estimated_merge_bytes_MB": merge_bytes_lower / 1e6,

        "sparse_random_add_only_GOPs": gops(sparse_random_add_only_ms),
        "sparse_fixed_allrows_add_only_GOPs": gops(sparse_fixed_allrows_add_only_ms),
        "sparse_block16_fixed_add_only_GOPs": gops(sparse_block16_fixed_add_only_ms),

        "sparse_random_weight_only_GBps_lower": gbps(b_weight_bytes_lower, sparse_random_add_only_ms),
        "sparse_fixed_allrows_weight_only_GBps_lower": gbps(b_weight_bytes_lower, sparse_fixed_allrows_add_only_ms),
        "zero_GBps_lower": gbps(zero_bytes_lower, zero_ms),
        "merge_GBps_lower": gbps(merge_bytes_lower, merge_ms),
    }

    return row


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[POLICY] {args.policy}")
    log(f"[OUT] {out}")
    log(f"[DEVICE] {torch.cuda.get_device_name(device)}")
    log("[NOTE] v26 sparse correction roofline: zero / sparse random / fixed-index / block16-fixed / merge.")

    raw_policy = json.load(open(args.policy))
    ratio_map = V22B.flatten_policy_ratios(raw_policy)

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V22B.V8.resolve_extensions(B, args, out)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    layers = V22B.V8.get_layers(model)
    if args.layers == "all":
        layer_ids = list(range(len(layers)))
    else:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    rows = []

    for layer_idx in layer_ids:
        base_layer = layers[layer_idx]
        policy_rows = V22B.policy_ratios_for_layer(
            base_layer,
            ratio_map,
            layer_idx,
            missing_ratio=0.0,
        )

        split_layer, patch_records, seed_records = V22B.patch_layer_with_policy_ratios(
            base_layer=base_layer,
            B=B,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            policy_rows=policy_rows,
            eps=args.eps,
            device=device,
        )

        ratio_sum = V22B.layer_ratio_summary(policy_rows)
        log(f"\n[LAYER] {layer_idx} " + json.dumps(ratio_sum, ensure_ascii=False))

        for module_name in TARGET_LINEAR_NAMES:
            linear = get_submodule(split_layer, module_name)
            if not getattr(linear, "is_split", False):
                continue

            for batch in batches:
                log(f"[CASE] layer={layer_idx} module={module_name} batch={batch}")
                row = measure_one(
                    linear=linear,
                    batch=batch,
                    seq_len=args.seq_len,
                    warmup=args.warmup,
                    iters=args.iters,
                    device=device,
                )
                if row is None:
                    continue

                row.update({
                    "layer_idx": layer_idx,
                    "module": module_name,
                    **ratio_sum,
                })

                rows.append(row)
                log("[RESULT] " + json.dumps(row, ensure_ascii=False))

                write_csv(out / "sparse_roofline_by_linear_v26.csv", rows)
                json.dump(rows, open(out / "sparse_roofline_by_linear_v26.json", "w"), indent=2, ensure_ascii=False)

        del split_layer
        torch.cuda.empty_cache()

    # 按 module 汇总
    summary = []
    for module in TARGET_LINEAR_NAMES:
        rs = [r for r in rows if r["module"] == module]
        if not rs:
            continue

        def avg(k):
            vals = [float(r[k]) for r in rs if k in r]
            return sum(vals) / len(vals) if vals else 0.0

        summary.append({
            "module": module,
            "num_cases": len(rs),
            "avg_sparse_random_add_only_ms": avg("sparse_random_add_only_ms"),
            "avg_sparse_random_total_ms": avg("sparse_random_total_ms"),
            "avg_zero_ms": avg("zero_ms"),
            "avg_merge_ms": avg("merge_ms"),
            "avg_fixed_allrows_speedup_over_random": avg("fixed_allrows_speedup_over_random"),
            "avg_block16_fixed_speedup_over_random": avg("block16_fixed_speedup_over_random"),
            "avg_sparse_random_GOPs": avg("sparse_random_add_only_GOPs"),
            "avg_sparse_random_weight_only_GBps_lower": avg("sparse_random_weight_only_GBps_lower"),
            "avg_estimated_weight_bytes_lower_GB": avg("estimated_weight_bytes_lower_GB"),
        })

    write_csv(out / "sparse_roofline_summary_by_module_v26.csv", summary)
    json.dump(summary, open(out / "sparse_roofline_summary_by_module_v26.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {out / 'sparse_roofline_by_linear_v26.csv'}")
    log(f"[CSV] {out / 'sparse_roofline_summary_by_module_v26.csv'}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
