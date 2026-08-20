import argparse
import copy
import csv
import inspect
import json
import os
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8
from qfactory.kernels.gemm_w4a4 import gemm_int4_int4_nt


TARGET_LINEAR_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--policy", required=True)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--fixed_ratio", type=float, default=0.05)
    p.add_argument("--missing_ratio", type=float, default=0.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    return p.parse_args()


def get_submodule(root, name):
    m = root
    for part in name.split("."):
        m = getattr(m, part)
    return m


def set_submodule(root, name, new_module):
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def is_real_policy_linear(m):
    name = m.__class__.__name__
    return name == "RealPolicyLinear" or "RealPolicyLinear" in name


def run_layer_once(layer, hidden_states, position_ids, position_embeddings):
    out = layer(
        hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    return out[0] if isinstance(out, tuple) else out


def qfactory_gemm_raw(A_pack, B_pack, C_i32, M, N, K):
    A2 = A_pack.view(M, K // 2).contiguous()
    B2 = B_pack.view(N, K // 2).contiguous()
    C2 = C_i32.view(M, N).contiguous()
    ret = gemm_int4_int4_nt(A2, B2, C2)
    if isinstance(ret, torch.Tensor):
        return ret.view(M, N)
    return C2


def patch_qfactory_raw_backend(layer):
    patched = []

    for name, mod in layer.named_modules():
        if not is_real_policy_linear(mod):
            continue
        if not hasattr(mod, "B_col"):
            continue

        def make_qf_pure(module_name):
            def qf_pure(self, A, scratch):
                M = int(A.shape[0])
                ext = getattr(self, "ext", None)
                if ext is None:
                    ext = getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError("cannot find ext/main_ext")

                output = torch.empty((M, self.N), dtype=torch.float16, device=A.device)

                ext.pack_a_full_s4(
                    A,
                    scratch["A_pack"],
                    scratch["a_scale"],
                    self.eps,
                )

                C = qfactory_gemm_raw(
                    scratch["A_pack"],
                    self.B_col,
                    scratch["C_i32"],
                    M,
                    self.N,
                    self.K,
                )

                ext.scale_i32_to_fp16(
                    C,
                    scratch["a_scale"],
                    self.w_scale,
                    output,
                )
                return output

            return qf_pure

        def make_qf_split_compute(module_name):
            def qf_split_compute(
                self,
                A,
                scratch,
                B_col,
                dense_ready_event,
                dense_stream,
                sparse_stream,
            ):
                M = int(A.shape[0])
                device = A.device
                current = torch.cuda.current_stream(device)

                ext = getattr(self, "ext", None)
                if ext is None:
                    ext = getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError("cannot find ext/main_ext")

                indices, top_q, _ = self._prepare_split(A, scratch)

                dense_stream.wait_stream(current)
                sparse_stream.wait_stream(current)

                with torch.cuda.stream(dense_stream):
                    C = qfactory_gemm_raw(
                        scratch["A_pack"],
                        B_col,
                        scratch["C_body_i32"],
                        M,
                        self.N,
                        self.K,
                    )
                    ext.scale_i32_to_fp16(
                        C,
                        scratch["body_scale"],
                        self.w_scale,
                        scratch["Y_body"],
                    )

                with torch.cuda.stream(sparse_stream):
                    scratch["Y_sparse"].zero_()
                    ext.sparse_top_add_rowmajor_quad_shared(
                        top_q,
                        indices,
                        self.B_row,
                        scratch["top_scale"],
                        self.w_scale,
                        scratch["Y_sparse"],
                        self.K,
                    )

                indices.record_stream(sparse_stream)
                top_q.record_stream(sparse_stream)

                current.wait_stream(dense_stream)
                current.wait_stream(sparse_stream)

                output = torch.empty((M, self.N), dtype=torch.float16, device=device)
                torch.add(scratch["Y_body"], scratch["Y_sparse"], out=output)
                return output

            return qf_split_compute

        mod._pure = types.MethodType(make_qf_pure(name), mod)

        if getattr(mod, "is_split", False):
            mod._split_compute = types.MethodType(make_qf_split_compute(name), mod)

        patched.append(name)

    return patched


def flatten_policy_ratios(obj):
    ratio_map = {}

    # v22b: 正式使用 projected ratio，而不是 continuous ratio。
    # policy.json 里一般同时有 ratio_continuous / ratio_projected；
    # v22 默认优先拿到了 ratio_continuous，这里强制优先 ratio_projected。
    ratio_keys = [
        "ratio_projected",
        "selected_ratio",
        "ratio",
        "ratio_continuous",
        "tail_ratio",
        "r",
    ]

    def normalize_ratio(v):
        try:
            x = float(v)
        except Exception:
            return None
        if x > 1.0 and x <= 100.0:
            x = x / 100.0
        if x < 0.0:
            x = 0.0
        if x > 1.0:
            x = 1.0
        return x

    def walk(x, path):
        if isinstance(x, dict):
            found_ratio = None
            for rk in ratio_keys:
                if rk in x:
                    found_ratio = normalize_ratio(x[rk])
                    break

            if found_ratio is not None:
                name = (
                    x.get("name")
                    or x.get("module")
                    or x.get("linear")
                    or x.get("key")
                    or ".".join(path)
                )
                ratio_map[str(name)] = found_ratio

            for k, v in x.items():
                walk(v, path + [str(k)])

        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + [str(i)])

        elif isinstance(x, (int, float)) and path:
            # 支持 {"model.layers.0.xxx": 0.02} 这种格式
            r = normalize_ratio(x)
            if r is not None:
                ratio_map[".".join(path)] = r

    walk(obj, [])
    return ratio_map


def lookup_ratio(ratio_map, layer_idx, module_name, missing_ratio):
    exact_candidates = [
        f"model.layers.{layer_idx}.{module_name}",
        f"layers.{layer_idx}.{module_name}",
        f"{layer_idx}.{module_name}",
        f"layer{layer_idx}.{module_name}",
        f"layer_{layer_idx}.{module_name}",
    ]

    for c in exact_candidates:
        if c in ratio_map:
            return ratio_map[c], c, False

    suffix_candidates = [
        f"model.layers.{layer_idx}.{module_name}",
        f"layers.{layer_idx}.{module_name}",
        f".{layer_idx}.{module_name}",
        f"{layer_idx}.{module_name}",
    ]

    for k, v in ratio_map.items():
        kk = str(k)
        if any(kk.endswith(suf) for suf in suffix_candidates):
            return float(v), kk, False
        if f"layers.{layer_idx}.{module_name}" in kk:
            return float(v), kk, False

    return float(missing_ratio), "", True


def get_linear_shape(base_layer, module_name):
    m = get_submodule(base_layer, module_name)
    W = m.weight
    N = int(W.shape[0])
    K = int(W.shape[1])
    return K, N


def policy_ratios_for_layer(base_layer, ratio_map, layer_idx, missing_ratio):
    rows = []
    for name in TARGET_LINEAR_NAMES:
        K, N = get_linear_shape(base_layer, name)
        ratio, key, missing = lookup_ratio(ratio_map, layer_idx, name, missing_ratio)
        R = int((K * ratio + 0.999999999))
        rows.append({
            "layer_idx": layer_idx,
            "module": name,
            "K": K,
            "N": N,
            "ratio": float(ratio),
            "R": R,
            "mac": K * N,
            "policy_key": key,
            "missing": bool(missing),
        })
    return rows


def make_real_policy_module_from_seed(
    seed_mod,
    orig_linear,
    module_name,
    ratio,
    main_ext,
    layout_ext,
    policy_pack_ext,
    eps,
    device,
):
    cls = seed_mod.__class__

    mode = "dual_policy" if ratio > 0 else "pure"
    policy_cfg = {
        "ratio": float(ratio),
        "ratio_continuous": float(ratio),
        "activation_percentile": 100.0,
        "weight_percentile": 100.0,
    }

    kwargs = {
        "main_ext": main_ext,
        "layout_ext": layout_ext,
        "policy_pack_ext": policy_pack_ext,
        "mode": mode,
        "weight_cpu": orig_linear.weight.detach().cpu(),
        "bias_cpu": orig_linear.bias.detach().cpu() if getattr(orig_linear, "bias", None) is not None else None,
        "policy_cfg": policy_cfg,
        "gptq_scale_cpu": seed_mod.w_scale.detach().cpu(),
        "eps": eps,
        "device": device,
        "name": module_name,
        "scratch_pool": getattr(seed_mod, "scratch_pool", None),
        "serial_workspace": getattr(seed_mod, "serial_workspace", None),
        "prefetch_workspace": getattr(seed_mod, "prefetch_workspace", None),
        "rotate_online": getattr(seed_mod, "rotate_online", False),
        "had_k": getattr(seed_mod, "had_k", None),
        "had_factor": getattr(seed_mod, "had_factor", 1),
    }

    sig = inspect.signature(cls)
    usable = {k: v for k, v in kwargs.items() if k in sig.parameters}

    try:
        return cls(**usable)
    except TypeError as e:
        log("[CONSTRUCTOR_SIGNATURE] " + str(sig))
        log("[CONSTRUCTOR_USABLE_KWARGS] " + json.dumps(sorted(usable.keys()), ensure_ascii=False))
        raise e


def patch_layer_with_policy_ratios(
    base_layer,
    B,
    main_ext,
    layout_ext,
    policy_pack_ext,
    policy_rows,
    eps,
    device,
):
    # 先用固定 ratio 创建 seed layer，复用其 RealPolicyLinear class、scale、scratch_pool 等。
    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    layer, seed_records = V8.patch_layer_with_real_policy(
        layer=layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="dual_policy",
        ratio=0.05,
        eps=eps,
        device=device,
    )

    row_by_module = {r["module"]: r for r in policy_rows}
    patch_records = []

    for module_name in TARGET_LINEAR_NAMES:
        seed_mod = get_submodule(layer, module_name)
        orig_linear = get_submodule(base_layer, module_name)
        ratio = float(row_by_module[module_name]["ratio"])

        new_mod = make_real_policy_module_from_seed(
            seed_mod=seed_mod,
            orig_linear=orig_linear,
            module_name=module_name,
            ratio=ratio,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            eps=eps,
            device=device,
        )

        set_submodule(layer, module_name, new_mod)

        patch_records.append({
            "module": module_name,
            "ratio": ratio,
            "mode": "dual_policy" if ratio > 0 else "pure",
            "K": int(new_mod.K),
            "N": int(new_mod.N),
            "R": int(getattr(new_mod, "R", 0)),
            "is_split": bool(getattr(new_mod, "is_split", False)),
        })

    layer.to(device=device).eval()
    return layer, patch_records, seed_records


def bench_graph_or_eager(fn, warmup, iters, device):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)

        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iters):
                g.replay()
            end.record()
            torch.cuda.synchronize(device)

            ms = float(start.elapsed_time(end) / iters)
            del g
            torch.cuda.empty_cache()
            return ms, "graph"
        except Exception as e:
            log(f"[WARN] graph failed, fallback eager: {repr(e)}")
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(iters):
                fn()
            end.record()
            torch.cuda.synchronize(device)

            return float(start.elapsed_time(end) / iters), "eager"


def build_bf16_layer(base_layer, device):
    return copy.deepcopy(base_layer).to(device=device, dtype=torch.bfloat16).eval()


def build_pure_layer(base_layer, B, main_ext, layout_ext, policy_pack_ext, eps, device):
    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    layer, records = V8.patch_layer_with_real_policy(
        layer=layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="pure",
        ratio=0.0,
        eps=eps,
        device=device,
    )
    layer.to(device=device).eval()
    patch_qfactory_raw_backend(layer)
    return layer, records


def build_fixed_split_layer(base_layer, B, main_ext, layout_ext, policy_pack_ext, ratio, eps, device):
    layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    layer, records = V8.patch_layer_with_real_policy(
        layer=layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="dual_policy",
        ratio=ratio,
        eps=eps,
        device=device,
    )
    layer.to(device=device).eval()
    patch_qfactory_raw_backend(layer)
    return layer, records


def layer_ratio_summary(policy_rows):
    total_mac = sum(r["mac"] for r in policy_rows)
    nonzero = [r for r in policy_rows if r["ratio"] > 0]
    mac_ratio = sum(r["ratio"] * r["mac"] for r in policy_rows) / total_mac if total_mac > 0 else 0.0
    avg_ratio = sum(r["ratio"] for r in policy_rows) / len(policy_rows) if policy_rows else 0.0
    max_ratio = max((r["ratio"] for r in policy_rows), default=0.0)
    return {
        "policy_avg_ratio": avg_ratio,
        "policy_mac_weighted_ratio": mac_ratio,
        "policy_max_ratio": max_ratio,
        "policy_nonzero_linears": len(nonzero),
        "policy_missing_linears": sum(1 for r in policy_rows if r["missing"]),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise FileNotFoundError(f"policy not found: {policy_path}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[POLICY] {policy_path}")
    log(f"[BATCHES] {args.batches}")
    log(f"[FIXED_RATIO] {args.fixed_ratio}")
    log(f"[DEVICE] {torch.cuda.get_device_name(device)}")
    log(f"[QFACTORY_ARCH] {os.environ.get('QFACTORY_ARCH')}")
    log(f"[QFACTORY_CACHE_DIR] {os.environ.get('QFACTORY_CACHE_DIR')}")
    log("[NOTE] v22b projected-ratio run: BF16 / Pure W4A4 / Split fixed / Split policy_projected. W4A4 dense backend = QFactory raw A4W4.")

    raw_policy = json.load(open(policy_path))
    ratio_map = flatten_policy_ratios(raw_policy)
    log(f"[POLICY_RATIO_KEYS] {len(ratio_map)}")
    for k in list(ratio_map.keys())[:20]:
        log(f"[POLICY_SAMPLE] {k} -> {ratio_map[k]}")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    layers = V8.get_layers(model)
    hidden_size = V8.infer_hidden_size(model)
    num_layers = len(layers)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    log(f"[NUM_LAYERS] {num_layers}")
    log(f"[HIDDEN_SIZE] {hidden_size}")

    latency_rows = []
    policy_rows_all = []
    missing_rows = []

    for layer_idx, base_layer in enumerate(layers):
        policy_rows = policy_ratios_for_layer(base_layer, ratio_map, layer_idx, args.missing_ratio)
        ratio_sum = layer_ratio_summary(policy_rows)
        policy_rows_all.extend(policy_rows)

        for r in policy_rows:
            if r["missing"]:
                missing_rows.append(r)

        log(f"\n[LAYER] {layer_idx}/{num_layers-1} " + json.dumps(ratio_sum, ensure_ascii=False))

        # 逐 layer 构建四种 layer，避免整模型三份常驻显存。
        bf16_layer = build_bf16_layer(base_layer, device)

        pure_layer, pure_patch = build_pure_layer(
            base_layer,
            B,
            main_ext,
            layout_ext,
            policy_pack_ext,
            args.eps,
            device,
        )

        split_fixed_layer, fixed_patch = build_fixed_split_layer(
            base_layer,
            B,
            main_ext,
            layout_ext,
            policy_pack_ext,
            args.fixed_ratio,
            args.eps,
            device,
        )

        split_policy_layer, policy_patch, seed_records = patch_layer_with_policy_ratios(
            base_layer=base_layer,
            B=B,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            policy_rows=policy_rows,
            eps=args.eps,
            device=device,
        )
        patch_qfactory_raw_backend(split_policy_layer)

        for batch in batches:
            log(f"[CASE] layer={layer_idx} batch={batch} seq_len={args.seq_len}")

            hidden_bf16 = torch.randn(
                batch,
                args.seq_len,
                hidden_size,
                device=device,
                dtype=torch.bfloat16,
            )
            hidden_fp16 = hidden_bf16.to(torch.float16)

            position_ids = V8.make_position_ids(batch, args.seq_len, device)
            pe_bf16 = V8.build_position_embeddings(model, hidden_bf16, position_ids, torch.bfloat16)
            pe_fp16 = V8.build_position_embeddings(model, hidden_fp16, position_ids, torch.float16)

            bf16_ms, bf16_mode = bench_graph_or_eager(
                lambda: run_layer_once(bf16_layer, hidden_bf16, position_ids, pe_bf16),
                args.warmup,
                args.iters,
                device,
            )
            log(f"[TIME] layer={layer_idx} batch={batch} bf16_ms={bf16_ms:.6f} mode={bf16_mode}")

            pure_ms, pure_mode = bench_graph_or_eager(
                lambda: run_layer_once(pure_layer, hidden_fp16, position_ids, pe_fp16),
                args.warmup,
                args.iters,
                device,
            )
            log(f"[TIME] layer={layer_idx} batch={batch} pure_w4a4_qf_ms={pure_ms:.6f} mode={pure_mode}")

            split_fixed_ms, split_fixed_mode = bench_graph_or_eager(
                lambda: run_layer_once(split_fixed_layer, hidden_fp16, position_ids, pe_fp16),
                args.warmup,
                args.iters,
                device,
            )
            log(f"[TIME] layer={layer_idx} batch={batch} split_fixed_qf_ms={split_fixed_ms:.6f} mode={split_fixed_mode}")

            split_policy_ms, split_policy_mode = bench_graph_or_eager(
                lambda: run_layer_once(split_policy_layer, hidden_fp16, position_ids, pe_fp16),
                args.warmup,
                args.iters,
                device,
            )
            log(f"[TIME] layer={layer_idx} batch={batch} split_policy_qf_ms={split_policy_ms:.6f} mode={split_policy_mode}")

            row = {
                "model": args.model,
                "layer_idx": layer_idx,
                "num_layers": num_layers,
                "batch": batch,
                "seq_len": args.seq_len,
                "hidden_size": hidden_size,
                "bf16_ms": bf16_ms,
                "pure_w4a4_qfactory_ms": pure_ms,
                "split_fixed_ratio": args.fixed_ratio,
                "split_fixed_qfactory_ms": split_fixed_ms,
                "split_policy_qfactory_ms": split_policy_ms,
                "bf16_mode": bf16_mode,
                "pure_mode": pure_mode,
                "split_fixed_mode": split_fixed_mode,
                "split_policy_mode": split_policy_mode,
                "pure_over_bf16": pure_ms / bf16_ms if bf16_ms > 0 else None,
                "split_fixed_over_bf16": split_fixed_ms / bf16_ms if bf16_ms > 0 else None,
                "split_policy_over_bf16": split_policy_ms / bf16_ms if bf16_ms > 0 else None,
                "split_fixed_over_pure": split_fixed_ms / pure_ms if pure_ms > 0 else None,
                "split_policy_over_pure": split_policy_ms / pure_ms if pure_ms > 0 else None,
                "split_policy_over_fixed": split_policy_ms / split_fixed_ms if split_fixed_ms > 0 else None,
                "policy_speedup_over_fixed": split_fixed_ms / split_policy_ms if split_policy_ms > 0 else None,
                **ratio_sum,
            }

            latency_rows.append(row)
            log("[RESULT] " + json.dumps(row, ensure_ascii=False))

            del hidden_bf16, hidden_fp16, position_ids, pe_bf16, pe_fp16
            torch.cuda.empty_cache()

            # 每个 case 后持续落盘，防止中途失败丢结果。
            write_csv(out_dir / "layer_latency_all_v22.csv", latency_rows)
            json.dump(latency_rows, open(out_dir / "layer_latency_all_v22.json", "w"), indent=2, ensure_ascii=False)

        del bf16_layer, pure_layer, split_fixed_layer, split_policy_layer
        torch.cuda.empty_cache()

    write_csv(out_dir / "policy_ratio_summary_v22.csv", policy_rows_all)
    json.dump(policy_rows_all, open(out_dir / "policy_ratio_summary_v22.json", "w"), indent=2, ensure_ascii=False)

    write_csv(out_dir / "policy_missing_v22.csv", missing_rows)
    json.dump(missing_rows, open(out_dir / "policy_missing_v22.json", "w"), indent=2, ensure_ascii=False)

    summary_rows = []
    for batch in batches:
        rs = [r for r in latency_rows if int(r["batch"]) == batch]
        if not rs:
            continue

        def s(key):
            return sum(float(r[key]) for r in rs)

        summary = {
            "batch": batch,
            "seq_len": args.seq_len,
            "num_layers": num_layers,
            "sum_bf16_ms": s("bf16_ms"),
            "sum_pure_w4a4_qfactory_ms": s("pure_w4a4_qfactory_ms"),
            "sum_split_fixed_qfactory_ms": s("split_fixed_qfactory_ms"),
            "sum_split_policy_qfactory_ms": s("split_policy_qfactory_ms"),
        }

        summary["pure_over_bf16_sum"] = summary["sum_pure_w4a4_qfactory_ms"] / summary["sum_bf16_ms"]
        summary["split_fixed_over_bf16_sum"] = summary["sum_split_fixed_qfactory_ms"] / summary["sum_bf16_ms"]
        summary["split_policy_over_bf16_sum"] = summary["sum_split_policy_qfactory_ms"] / summary["sum_bf16_ms"]
        summary["split_fixed_over_pure_sum"] = summary["sum_split_fixed_qfactory_ms"] / summary["sum_pure_w4a4_qfactory_ms"]
        summary["split_policy_over_pure_sum"] = summary["sum_split_policy_qfactory_ms"] / summary["sum_pure_w4a4_qfactory_ms"]
        summary["split_policy_over_fixed_sum"] = summary["sum_split_policy_qfactory_ms"] / summary["sum_split_fixed_qfactory_ms"]
        summary["policy_speedup_over_fixed_sum"] = summary["sum_split_fixed_qfactory_ms"] / summary["sum_split_policy_qfactory_ms"]
        summary_rows.append(summary)

    write_csv(out_dir / "summary_by_batch_v22.csv", summary_rows)
    json.dump(summary_rows, open(out_dir / "summary_by_batch_v22.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {out_dir / 'layer_latency_all_v22.csv'}")
    log(f"[CSV] {out_dir / 'summary_by_batch_v22.csv'}")
    log(f"[CSV] {out_dir / 'policy_ratio_summary_v22.csv'}")
    log(f"[MISSING] {len(missing_rows)} missing policy linears")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
