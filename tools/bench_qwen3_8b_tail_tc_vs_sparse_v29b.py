import argparse
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
    p.add_argument("--modules", default="all")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--sweep_ratios", default="policy,0.001,0.002,0.005,0.01,0.02,0.05,0.08,0.10,0.15")
    p.add_argument("--max_rows_for_tail_build", type=int, default=8192)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    return p.parse_args()


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("cannot locate model.model.layers")


def get_submodule(root, name):
    m = root
    for p in name.split("."):
        if not hasattr(m, p):
            return None
        m = getattr(m, p)
    return m


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def normalize_ratio(x):
    x = safe_float(x, 0.0)
    if x > 1.0 and x <= 100.0:
        x /= 100.0
    return max(0.0, min(1.0, x))


def normalize_percentile(x):
    x = safe_float(x, 1.0)
    if x > 1.0 and x <= 100.0:
        x /= 100.0
    return max(0.0, min(1.0, x))


def flatten_policy_entries(policy_obj):
    entries = {}

    def maybe_record(d, path):
        if not isinstance(d, dict):
            return

        text = " ".join(str(x) for x in path)
        name = d.get("name") or d.get("module") or d.get("linear") or d.get("key") or ""
        full_text = text + " " + str(name)

        hit_module = None
        hit_layer = None

        for mod in TARGET_LINEAR_NAMES:
            if mod in full_text:
                hit_module = mod
                break

        import re
        m = re.search(r"layers\.(\d+)\.", full_text)
        if m:
            hit_layer = int(m.group(1))
        elif "layer_idx" in d:
            try:
                hit_layer = int(d["layer_idx"])
            except Exception:
                hit_layer = None
        elif "layer" in d:
            try:
                hit_layer = int(d["layer"])
            except Exception:
                hit_layer = None

        if hit_module is None or hit_layer is None:
            return

        ratio_keys = [
            "ratio_projected", "projected_ratio", "ratio",
            "ratio_continuous", "continuous_ratio", "r"
        ]
        has_ratio = any(k in d for k in ratio_keys)
        has_pct = any(k in d for k in [
            "activation_percentile", "act_percentile", "body_percentile",
            "a_percentile", "weight_percentile", "w_percentile"
        ])

        if not (has_ratio or has_pct):
            return

        ratio_projected = d.get("ratio_projected", d.get("projected_ratio", d.get("ratio", 0.0)))
        ratio_continuous = d.get("ratio_continuous", d.get("continuous_ratio", d.get("ratio", ratio_projected)))
        ratio = d.get("ratio", ratio_projected)

        act_pct = d.get(
            "activation_percentile",
            d.get("act_percentile",
              d.get("a_percentile",
                d.get("body_percentile",
                  d.get("body_activation_percentile",
                    d.get("input_percentile",
                      d.get("percentile", 1.0)))))),
        )
        w_pct = d.get(
            "weight_percentile",
            d.get("w_percentile",
              d.get("weight_clip_percentile",
                d.get("w_clip_percentile",
                  d.get("weight_percent", 1.0)))),
        )

        entries[(hit_layer, hit_module)] = {
            "name": str(name) if name else f"model.layers.{hit_layer}.{hit_module}",
            "module": hit_module,
            "layer_idx": hit_layer,
            "ratio": normalize_ratio(ratio),
            "ratio_continuous": normalize_ratio(ratio_continuous),
            "ratio_projected": normalize_ratio(ratio_projected),
            "activation_percentile": normalize_percentile(act_pct),
            "weight_percentile": normalize_percentile(w_pct),
        }

    def walk(x, path):
        if isinstance(x, dict):
            maybe_record(x, path)
            for k, v in x.items():
                walk(v, path + [str(k)])
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + [str(i)])

    walk(policy_obj, [])
    return entries


def make_policy_rows(base_layer, entries, layer_idx):
    rows = []
    for module_name in TARGET_LINEAR_NAMES:
        mod = get_submodule(base_layer, module_name)
        if mod is None or not hasattr(mod, "weight"):
            continue

        K = int(mod.weight.shape[1])
        N = int(mod.weight.shape[0])
        e = entries.get((layer_idx, module_name), None)

        if e is None:
            ratio = 0.0
            act_pct = 1.0
            w_pct = 1.0
        else:
            ratio = float(e["ratio_projected"])
            act_pct = float(e["activation_percentile"])
            w_pct = float(e["weight_percentile"])

        rows.append({
            "name": f"model.layers.{layer_idx}.{module_name}",
            "module": module_name,
            "layer_idx": layer_idx,
            "ratio": ratio,
            "ratio_projected": ratio,
            "ratio_continuous": ratio,
            "activation_percentile": act_pct,
            "act_percentile": act_pct,
            "body_percentile": act_pct,
            "weight_percentile": w_pct,
            "w_percentile": w_pct,
            "K": K,
            "N": N,
        })

    return rows


def bench_cuda(fn, warmup, iters, device):
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


def get_linear_attr(linear, names, required=True):
    for n in names:
        if hasattr(linear, n):
            v = getattr(linear, n)
            if v is not None:
                return v
    if required:
        attrs = [a for a in dir(linear) if not a.startswith("__")]
        raise AttributeError(f"cannot find any of {names}; attrs sample={attrs[:120]}")
    return None


def ensure_scratch(linear, scratch, M, K, N, device):
    if "A_pack" not in scratch:
        scratch["A_pack"] = torch.empty((M, K // 2), device=device, dtype=torch.uint8)
    if "a_scale" not in scratch:
        scratch["a_scale"] = torch.empty((M,), device=device, dtype=torch.float16)
    if "C_i32" not in scratch:
        scratch["C_i32"] = torch.empty((M, N), device=device, dtype=torch.int32)
    return scratch


def get_scratch(linear, M, K, N, device):
    """
    v29b:
    Do NOT reuse RealPolicyLinear.scratch_pool here.
    Its a_scale buffer may be Half, while pack_a_full_s4 expects Float.
    This benchmark's tail-TC path calls _pure directly, so we allocate a clean
    scratch dict with a_scale=float32.
    """
    return {
        "A_pack": torch.empty((M, K // 2), device=device, dtype=torch.uint8),
        "a_scale": torch.empty((M,), device=device, dtype=torch.float32),
        "C_i32": torch.empty((M, N), device=device, dtype=torch.int32),
        "Y": torch.empty((M, N), device=device, dtype=torch.float16),
        "output": torch.empty((M, N), device=device, dtype=torch.float16),
    }


def linear_pure(linear, A, device):
    K = int(getattr(linear, "K", A.shape[1]))
    N = int(getattr(linear, "N", get_linear_attr(linear, ["N"], required=False) or 0))
    if N <= 0:
        B_col = get_linear_attr(linear, ["B_col", "B_pack", "B", "qweight", "weight_q"], required=False)
        if B_col is not None and hasattr(B_col, "shape"):
            N = int(B_col.shape[0] if B_col.ndim >= 2 else 0)

    if A.dtype != torch.float32:
        A = A.float()

    scratch = get_scratch(linear, A.shape[0], K, N, device)

    if not getattr(linear, "_v29b_dtype_logged", False):
        print("[PURE_DTYPE]", {
            "A": str(A.dtype),
            "A_pack": str(scratch["A_pack"].dtype),
            "a_scale": str(scratch["a_scale"].dtype),
            "C_i32": str(scratch["C_i32"].dtype),
        }, flush=True)
        linear._v29b_dtype_logged = True

    return linear._pure(A, scratch)


def make_random_indices(M, K, R, device):
    # latency benchmark only: random gather pattern.
    # Duplicates are possible, but shape-driven cost remains representative.
    idx = torch.randint(0, K, (M, R), device=device, dtype=torch.int32)
    idx, _ = torch.sort(idx, dim=1)
    return idx


def make_tail_tensor(A, indices):
    # Pessimistic prototype: materialize full [M,K] tail mask.
    M, K = A.shape
    vals = A.gather(1, indices.to(torch.long))
    tail = torch.zeros_like(A)
    tail.scatter_(1, indices.to(torch.long), vals)
    return tail


def measure_case(linear, ext, A, K, N, ratio, ratio_name, batch, seq_len, layer_idx, module_name, warmup, iters, device):
    M = A.shape[0]
    R = int(math.ceil(K * ratio))
    if R <= 0:
        return None

    B_row = get_linear_attr(linear, ["B_row", "B_rowmajor", "B_row_major", "qweight_row", "weight_row"], required=True)
    w_scale = get_linear_attr(linear, ["w_scale", "weight_scale", "scales_w"], required=True)

    indices = make_random_indices(M, K, R, device)
    top_q = torch.randint(-7, 8, (M, R), device=device, dtype=torch.int8)
    top_scale = torch.empty((M,), device=device, dtype=torch.float16).fill_(1.0)
    Y_sparse = torch.empty((M, N), device=device, dtype=torch.float16)

    def fn_sparse_add_only():
        ext.sparse_top_add_rowmajor_quad_shared(
            top_q,
            indices,
            B_row,
            top_scale,
            w_scale,
            Y_sparse,
            K,
        )

    def fn_sparse_total_zero_add():
        Y_sparse.zero_()
        ext.sparse_top_add_rowmajor_quad_shared(
            top_q,
            indices,
            B_row,
            top_scale,
            w_scale,
            Y_sparse,
            K,
        )

    sparse_add_ms = bench_cuda(fn_sparse_add_only, warmup, iters, device)
    sparse_total_ms = bench_cuda(fn_sparse_total_zero_add, warmup, iters, device)

    # full-K masked Tensor Core tail path:
    # 1) build materialized tail mask: pessimistic implementation cost
    tail_holder = {}

    def fn_build_tail():
        tail_holder["tail"] = make_tail_tensor(A, indices)

    build_tail_ms = bench_cuda(fn_build_tail, max(1, warmup // 2), max(3, iters // 3), device)
    A_tail = tail_holder["tail"]

    # 2) prebuilt-tail QFactory pure path: optimistic if tail packing can be fused with topk/pack
    def fn_tail_tc_prebuilt():
        linear_pure(linear, A_tail, device)

    tail_tc_ms = bench_cuda(fn_tail_tc_prebuilt, warmup, iters, device)

    # 3) one full pure GEMM for scale reference
    def fn_full_pure():
        linear_pure(linear, A, device)

    full_pure_ms = bench_cuda(fn_full_pure, warmup, iters, device)

    return {
        "layer_idx": layer_idx,
        "module": module_name,
        "batch": batch,
        "seq_len": seq_len,
        "M": M,
        "K": K,
        "N": N,
        "ratio_name": ratio_name,
        "ratio": ratio,
        "R": R,
        "sparse_add_only_ms": sparse_add_ms,
        "sparse_total_zero_add_ms": sparse_total_ms,
        "tail_mask_build_ms": build_tail_ms,
        "tail_fullK_tc_prebuilt_ms": tail_tc_ms,
        "tail_fullK_tc_with_build_ms": tail_tc_ms + build_tail_ms,
        "full_pure_qfactory_ms": full_pure_ms,
        "tc_prebuilt_over_sparse_add": tail_tc_ms / sparse_add_ms if sparse_add_ms > 0 else None,
        "tc_prebuilt_over_sparse_total": tail_tc_ms / sparse_total_ms if sparse_total_ms > 0 else None,
        "tc_with_build_over_sparse_total": (tail_tc_ms + build_tail_ms) / sparse_total_ms if sparse_total_ms > 0 else None,
        "tail_tc_over_full_pure": tail_tc_ms / full_pure_ms if full_pure_ms > 0 else None,
        "two_tc_lower_bound_over_one_pure": (full_pure_ms + tail_tc_ms) / full_pure_ms if full_pure_ms > 0 else None,
        "winner_prebuilt": "tail_tc" if tail_tc_ms < sparse_total_ms else "sparse",
        "winner_with_build": "tail_tc" if (tail_tc_ms + build_tail_ms) < sparse_total_ms else "sparse",
    }


def parse_layers(s, n_layers):
    if s == "all":
        return list(range(n_layers))
    ans = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        ans.append(int(x))
    return ans


def parse_modules(s):
    if s == "all":
        return TARGET_LINEAR_NAMES
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_sweep_ratios(s, policy_ratio):
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if x == "policy":
            vals.append(("policy", float(policy_ratio)))
        else:
            vals.append((x, normalize_ratio(float(x))))
    # 去重，保序
    out = []
    seen = set()
    for name, r in vals:
        key = round(float(r), 8)
        if key <= 0:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append((name, float(r)))
    return out


def summarize(rows):
    groups = {}
    for r in rows:
        key = (r["batch"], r["ratio_name"])
        groups.setdefault(key, []).append(r)

    out = []
    for (batch, ratio_name), rs in sorted(groups.items(), key=lambda x: (int(x[0][0]), str(x[0][1]))):
        n = len(rs)
        def avg(k):
            return sum(float(x[k]) for x in rs) / n if n else 0.0
        out.append({
            "batch": batch,
            "ratio_name": ratio_name,
            "num_cases": n,
            "avg_ratio": avg("ratio"),
            "avg_R": avg("R"),
            "avg_sparse_add_only_ms": avg("sparse_add_only_ms"),
            "avg_sparse_total_zero_add_ms": avg("sparse_total_zero_add_ms"),
            "avg_tail_fullK_tc_prebuilt_ms": avg("tail_fullK_tc_prebuilt_ms"),
            "avg_tail_fullK_tc_with_build_ms": avg("tail_fullK_tc_with_build_ms"),
            "avg_full_pure_qfactory_ms": avg("full_pure_qfactory_ms"),
            "avg_tc_prebuilt_over_sparse_total": avg("tc_prebuilt_over_sparse_total"),
            "avg_tc_with_build_over_sparse_total": avg("tc_with_build_over_sparse_total"),
            "tail_tc_prebuilt_win_rate": sum(1 for x in rs if x["winner_prebuilt"] == "tail_tc") / n,
            "tail_tc_with_build_win_rate": sum(1 for x in rs if x["winner_with_build"] == "tail_tc") / n,
        })
    return out


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(args.policy):
        raise FileNotFoundError(args.policy)

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[POLICY] {args.policy}")
    log(f"[OUT] {out}")
    log(f"[DEVICE] {torch.cuda.get_device_name(device)}")
    log(f"[BATCHES] {args.batches}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[SWEEP_RATIOS] {args.sweep_ratios}")

    raw_policy = json.load(open(args.policy))
    entries = flatten_policy_entries(raw_policy)
    log(f"[POLICY_ENTRIES] {len(entries)}")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V22B.V8.resolve_extensions(B, args, out)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    layers = get_layers(model)
    layer_ids = parse_layers(args.layers, len(layers))
    wanted_modules = set(parse_modules(args.modules))
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    rows = []
    failures = []

    for layer_idx in layer_ids:
        base_layer = layers[layer_idx]
        policy_rows = make_policy_rows(base_layer, entries, layer_idx)

        # 用真实 policy patch 一次，拿到 RealPolicyLinear、B_col/B_row/w_scale。
        patched_layer, patch_records, _ = V22B.patch_layer_with_policy_ratios(
            base_layer=base_layer,
            B=B,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            policy_rows=policy_rows,
            eps=args.eps,
            device=device,
        )

        log("")
        log(f"[LAYER] {layer_idx}")

        for pr in policy_rows:
            module_name = pr["module"]
            if module_name not in wanted_modules:
                continue

            linear = get_submodule(patched_layer, module_name)
            if linear is None:
                continue

            K = int(pr["K"])
            N = int(pr["N"])
            policy_ratio = float(pr["ratio_projected"])

            log("[LINEAR] " + json.dumps({
                "layer_idx": layer_idx,
                "module": module_name,
                "K": K,
                "N": N,
                "policy_ratio": policy_ratio,
                "policy_R": int(math.ceil(K * policy_ratio)),
                "activation_percentile": pr["activation_percentile"],
                "weight_percentile": pr["weight_percentile"],
            }, ensure_ascii=False))

            for batch in batches:
                M = batch * args.seq_len
                A = torch.randn((M, K), device=device, dtype=torch.float32)

                for ratio_name, ratio in parse_sweep_ratios(args.sweep_ratios, policy_ratio):
                    try:
                        row = measure_case(
                            linear=linear,
                            ext=main_ext,
                            A=A,
                            K=K,
                            N=N,
                            ratio=ratio,
                            ratio_name=ratio_name,
                            batch=batch,
                            seq_len=args.seq_len,
                            layer_idx=layer_idx,
                            module_name=module_name,
                            warmup=args.warmup,
                            iters=args.iters,
                            device=device,
                        )
                        if row is not None:
                            rows.append(row)
                            log("[RESULT] " + json.dumps(row, ensure_ascii=False))
                            write_csv(out / "qwen3_8b_tail_tc_vs_sparse_v29b.csv", rows)
                            json.dump(rows, open(out / "qwen3_8b_tail_tc_vs_sparse_v29b.json", "w"), indent=2, ensure_ascii=False)
                            write_csv(out / "qwen3_8b_tail_tc_vs_sparse_summary_v29b.csv", summarize(rows))
                    except Exception as e:
                        fail = {
                            "layer_idx": layer_idx,
                            "module": module_name,
                            "batch": batch,
                            "ratio_name": ratio_name,
                            "ratio": ratio,
                            "error": repr(e),
                        }
                        failures.append(fail)
                        log("[FAIL] " + json.dumps(fail, ensure_ascii=False))
                        write_csv(out / "qwen3_8b_tail_tc_vs_sparse_failures_v29b.csv", failures)

                del A
                torch.cuda.empty_cache()

        del patched_layer
        torch.cuda.empty_cache()

    summary = summarize(rows)
    write_csv(out / "qwen3_8b_tail_tc_vs_sparse_v29b.csv", rows)
    write_csv(out / "qwen3_8b_tail_tc_vs_sparse_summary_v29b.csv", summary)
    json.dump(rows, open(out / "qwen3_8b_tail_tc_vs_sparse_v29b.json", "w"), indent=2, ensure_ascii=False)
    json.dump(summary, open(out / "qwen3_8b_tail_tc_vs_sparse_summary_v29b.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {out / 'qwen3_8b_tail_tc_vs_sparse_v29b.csv'}")
    log(f"[CSV] {out / 'qwen3_8b_tail_tc_vs_sparse_summary_v29b.csv'}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
