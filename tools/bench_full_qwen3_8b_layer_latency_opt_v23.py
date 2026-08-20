import argparse
import csv
import copy
import json
import os
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import bench_full_qwen3_8b_layer_latency_policy_projected_v22b as V22B


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--policy", required=True)
    p.add_argument("--baseline_dir", required=True)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    p.add_argument("--check_first_n_layers", type=int, default=3)
    return p.parse_args()


def is_real_policy_linear(m):
    name = m.__class__.__name__
    return name == "RealPolicyLinear" or "RealPolicyLinear" in name


def ext_of(m):
    ext = getattr(m, "ext", None)
    if ext is None:
        ext = getattr(m, "main_ext", None)
    if ext is None:
        raise RuntimeError("cannot find ext/main_ext")
    return ext


def run_layer_once(layer, hidden_states, position_ids, position_embeddings):
    out = layer(
        hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    return out[0] if isinstance(out, tuple) else out


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(x):
    if x is None or x == "":
        return None
    return float(x)


def copy_output(t):
    out = torch.empty_like(t)
    out.copy_(t)
    return out


def qfactory_dense_to_ybody(linear, A_pack, body_scale, scratch, M):
    ext = ext_of(linear)
    C = V22B.qfactory_gemm_raw(
        A_pack,
        linear.B_col,
        scratch["C_body_i32"],
        M,
        linear.N,
        linear.K,
    )
    ext.scale_i32_to_fp16(
        C,
        body_scale,
        linear.w_scale,
        scratch["Y_body"],
    )
    return scratch["Y_body"]


def sparse_inplace_add_to_ybody(linear, indices, top_q, top_scale, y_body):
    ext = ext_of(linear)
    ext.sparse_top_add_rowmajor_quad_shared(
        top_q,
        indices,
        linear.B_row,
        top_scale,
        linear.w_scale,
        y_body,
        linear.K,
    )
    return y_body


def split_linear_from_prepared_inplace(linear, prep, M):
    scratch = linear.scratch_pool.get(M, linear.K, linear.N)
    y_body = qfactory_dense_to_ybody(
        linear=linear,
        A_pack=prep["A_pack"],
        body_scale=prep["body_scale"],
        scratch=scratch,
        M=M,
    )
    sparse_inplace_add_to_ybody(
        linear=linear,
        indices=prep["indices"],
        top_q=prep["top_q"],
        top_scale=prep["top_scale"],
        y_body=y_body,
    )
    return copy_output(y_body)


def patch_inplace_sparse_add(layer):
    patched = []

    for name, mod in layer.named_modules():
        if not is_real_policy_linear(mod):
            continue
        if not getattr(mod, "is_split", False):
            continue
        if not hasattr(mod, "B_col") or not hasattr(mod, "B_row"):
            continue

        def make_inplace_split(module_name):
            def qf_split_compute_inplace(
                self,
                A,
                scratch,
                B_col,
                dense_ready_event,
                dense_stream,
                sparse_stream,
            ):
                M = int(A.shape[0])

                indices, top_q, _ = self._prepare_split(A, scratch)

                y_body = qfactory_dense_to_ybody(
                    linear=self,
                    A_pack=scratch["A_pack"],
                    body_scale=scratch["body_scale"],
                    scratch=scratch,
                    M=M,
                )

                sparse_inplace_add_to_ybody(
                    linear=self,
                    indices=indices,
                    top_q=top_q,
                    top_scale=scratch["top_scale"],
                    y_body=y_body,
                )

                return copy_output(y_body)

            return qf_split_compute_inplace

        mod._split_compute = types.MethodType(make_inplace_split(name), mod)
        patched.append(name)

    return patched


def patch_mlp_gate_up_shared_prepare(layer):
    if not hasattr(layer, "mlp"):
        return {"enabled": False, "reason": "no mlp"}

    mlp = layer.mlp
    if not hasattr(mlp, "gate_proj") or not hasattr(mlp, "up_proj") or not hasattr(mlp, "down_proj"):
        return {"enabled": False, "reason": "missing gate/up/down"}

    gate = mlp.gate_proj
    up = mlp.up_proj

    if not (is_real_policy_linear(gate) and is_real_policy_linear(up)):
        return {"enabled": False, "reason": "gate/up not RealPolicyLinear"}

    if not (getattr(gate, "is_split", False) and getattr(up, "is_split", False)):
        return {"enabled": False, "reason": "gate/up not both split"}

    if int(gate.K) != int(up.K):
        return {"enabled": False, "reason": "K mismatch"}

    if int(getattr(gate, "R", -1)) != int(getattr(up, "R", -2)):
        return {
            "enabled": False,
            "reason": "R mismatch",
            "gate_R": int(getattr(gate, "R", -1)),
            "up_R": int(getattr(up, "R", -1)),
        }

    if int(getattr(gate, "R", 0)) <= 0:
        return {"enabled": False, "reason": "R <= 0"}

    original_forward = mlp.forward

    def shared_forward(self, x):
        original_shape = x.shape[:-1]
        A = x.reshape(-1, gate.K).contiguous()
        if A.dtype != torch.float16:
            A = A.to(torch.float16)

        M = int(A.shape[0])

        gate_scratch = gate.scratch_pool.get(M, gate.K, gate.N)
        indices, top_q, _ = gate._prepare_split(A, gate_scratch)

        prep = {
            "indices": indices,
            "top_q": top_q,
            "A_pack": gate_scratch["A_pack"],
            "body_scale": gate_scratch["body_scale"],
            "top_scale": gate_scratch["top_scale"],
        }

        gate_out = split_linear_from_prepared_inplace(gate, prep, M).reshape(*original_shape, gate.N)
        up_out = split_linear_from_prepared_inplace(up, prep, M).reshape(*original_shape, up.N)

        hidden = self.act_fn(gate_out) * up_out
        return self.down_proj(hidden)

    mlp.forward = types.MethodType(shared_forward, mlp)

    return {
        "enabled": True,
        "reason": "gate/up shared prepare enabled",
        "R": int(gate.R),
        "gate_ratio": float(getattr(gate, "ratio", 0.0)),
        "up_ratio": float(getattr(up, "ratio", 0.0)),
    }


def build_policy_layer(base_layer, B, main_ext, layout_ext, policy_pack_ext, policy_rows, eps, device, optimize=False):
    layer, patch_records, seed_records = V22B.patch_layer_with_policy_ratios(
        base_layer=base_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        policy_rows=policy_rows,
        eps=eps,
        device=device,
    )

    V22B.patch_qfactory_raw_backend(layer)

    opt_info = {}
    if optimize:
        opt_info["inplace_sparse_add_modules"] = patch_inplace_sparse_add(layer)
        opt_info["mlp_shared_prepare"] = patch_mlp_gate_up_shared_prepare(layer)

    return layer, patch_records, opt_info


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


def compare_tensors(a, b):
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt((d * d).mean()).item()),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = Path(args.baseline_dir)
    baseline_csv = baseline_dir / "layer_latency_all_v22.csv"
    if not baseline_csv.exists():
        raise FileNotFoundError(f"baseline projected csv not found: {baseline_csv}")

    policy_path = Path(args.policy)
    if not policy_path.exists():
        raise FileNotFoundError(f"policy not found: {policy_path}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[POLICY] {policy_path}")
    log(f"[BASELINE_CSV] {baseline_csv}")
    log(f"[BATCHES] {args.batches}")
    log(f"[DEVICE] {torch.cuda.get_device_name(device)}")
    log("[NOTE] v23 optimizations: projected policy + QFactory dense + sparse in-place add to Y_body + MLP gate/up shared prepare when R matches.")

    baseline_rows = read_csv(baseline_csv)
    baseline_map = {
        (int(r["layer_idx"]), int(r["batch"])): r
        for r in baseline_rows
    }

    raw_policy = json.load(open(policy_path))
    ratio_map = V22B.flatten_policy_ratios(raw_policy)

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V22B.V8.resolve_extensions(B, args, out_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    layers = V22B.V8.get_layers(model)
    hidden_size = V22B.V8.infer_hidden_size(model)
    num_layers = len(layers)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    log(f"[NUM_LAYERS] {num_layers}")
    log(f"[HIDDEN_SIZE] {hidden_size}")

    rows = []
    opt_infos = []

    for layer_idx, base_layer in enumerate(layers):
        policy_rows = V22B.policy_ratios_for_layer(
            base_layer,
            ratio_map,
            layer_idx,
            missing_ratio=0.0,
        )
        ratio_sum = V22B.layer_ratio_summary(policy_rows)

        log(f"\n[LAYER] {layer_idx}/{num_layers-1} " + json.dumps(ratio_sum, ensure_ascii=False))

        baseline_layer = None
        opt_layer = None

        need_check = layer_idx < args.check_first_n_layers
        if need_check:
            baseline_layer, _, _ = build_policy_layer(
                base_layer=base_layer,
                B=B,
                main_ext=main_ext,
                layout_ext=layout_ext,
                policy_pack_ext=policy_pack_ext,
                policy_rows=policy_rows,
                eps=args.eps,
                device=device,
                optimize=False,
            )

        opt_layer, patch_records, opt_info = build_policy_layer(
            base_layer=base_layer,
            B=B,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            policy_rows=policy_rows,
            eps=args.eps,
            device=device,
            optimize=True,
        )

        opt_info_row = {
            "layer_idx": layer_idx,
            **ratio_sum,
            "mlp_shared_prepare_enabled": bool(opt_info.get("mlp_shared_prepare", {}).get("enabled", False)),
            "mlp_shared_prepare_reason": opt_info.get("mlp_shared_prepare", {}).get("reason", ""),
            "inplace_sparse_add_count": len(opt_info.get("inplace_sparse_add_modules", [])),
        }
        opt_infos.append(opt_info_row)
        log("[OPT_INFO] " + json.dumps(opt_info_row, ensure_ascii=False))

        for batch in batches:
            log(f"[CASE] layer={layer_idx} batch={batch} seq_len={args.seq_len}")

            hidden = torch.randn(
                batch,
                args.seq_len,
                hidden_size,
                device=device,
                dtype=torch.float16,
            )
            position_ids = V22B.V8.make_position_ids(batch, args.seq_len, device)
            pe = V22B.V8.build_position_embeddings(model, hidden, position_ids, torch.float16)

            correctness = {}
            if need_check:
                with torch.inference_mode():
                    y_base = run_layer_once(baseline_layer, hidden, position_ids, pe)
                    y_opt = run_layer_once(opt_layer, hidden, position_ids, pe)
                    torch.cuda.synchronize(device)
                    correctness = compare_tensors(y_base, y_opt)
                    log("[CHECK] " + json.dumps({
                        "layer": layer_idx,
                        "batch": batch,
                        **correctness,
                    }, ensure_ascii=False))

            opt_ms, opt_mode = bench_graph_or_eager(
                lambda: run_layer_once(opt_layer, hidden, position_ids, pe),
                args.warmup,
                args.iters,
                device,
            )

            base = baseline_map.get((layer_idx, batch), {})
            base_policy_ms = f(base.get("split_policy_qfactory_ms"))
            base_pure_ms = f(base.get("pure_w4a4_qfactory_ms"))
            base_fixed_ms = f(base.get("split_fixed_qfactory_ms"))
            base_bf16_ms = f(base.get("bf16_ms"))

            row = {
                "model": args.model,
                "layer_idx": layer_idx,
                "batch": batch,
                "seq_len": args.seq_len,
                "hidden_size": hidden_size,

                "baseline_bf16_ms": base_bf16_ms,
                "baseline_pure_w4a4_qfactory_ms": base_pure_ms,
                "baseline_split_fixed_qfactory_ms": base_fixed_ms,
                "baseline_split_projected_qfactory_ms": base_policy_ms,

                "opt_split_projected_qfactory_ms": opt_ms,
                "opt_mode": opt_mode,

                "opt_over_baseline_projected": opt_ms / base_policy_ms if base_policy_ms else None,
                "opt_speedup_over_baseline_projected": base_policy_ms / opt_ms if base_policy_ms else None,
                "opt_over_pure": opt_ms / base_pure_ms if base_pure_ms else None,
                "opt_over_fixed": opt_ms / base_fixed_ms if base_fixed_ms else None,
                "opt_over_bf16": opt_ms / base_bf16_ms if base_bf16_ms else None,

                "check_max_abs": correctness.get("max_abs", ""),
                "check_mean_abs": correctness.get("mean_abs", ""),
                "check_rmse": correctness.get("rmse", ""),

                **ratio_sum,
                "mlp_shared_prepare_enabled": opt_info_row["mlp_shared_prepare_enabled"],
                "mlp_shared_prepare_reason": opt_info_row["mlp_shared_prepare_reason"],
                "inplace_sparse_add_count": opt_info_row["inplace_sparse_add_count"],
            }

            rows.append(row)
            log("[RESULT] " + json.dumps(row, ensure_ascii=False))

            write_csv(out_dir / "layer_latency_opt_v23.csv", rows)
            json.dump(rows, open(out_dir / "layer_latency_opt_v23.json", "w"), indent=2, ensure_ascii=False)

            del hidden, position_ids, pe
            torch.cuda.empty_cache()

        del opt_layer
        if baseline_layer is not None:
            del baseline_layer
        torch.cuda.empty_cache()

    write_csv(out_dir / "opt_info_v23.csv", opt_infos)
    json.dump(opt_infos, open(out_dir / "opt_info_v23.json", "w"), indent=2, ensure_ascii=False)

    summary_rows = []
    for batch in batches:
        rs = [r for r in rows if int(r["batch"]) == batch]

        def s(k):
            return sum(float(r[k]) for r in rs if r[k] not in ("", None))

        sum_base = s("baseline_split_projected_qfactory_ms")
        sum_opt = s("opt_split_projected_qfactory_ms")
        sum_pure = s("baseline_pure_w4a4_qfactory_ms")
        sum_fixed = s("baseline_split_fixed_qfactory_ms")
        sum_bf16 = s("baseline_bf16_ms")

        summary = {
            "batch": batch,
            "num_layers": len(rs),
            "sum_bf16_ms": sum_bf16,
            "sum_pure_w4a4_qfactory_ms": sum_pure,
            "sum_split_fixed_qfactory_ms": sum_fixed,
            "sum_baseline_projected_ms": sum_base,
            "sum_opt_projected_ms": sum_opt,
            "opt_over_baseline_projected": sum_opt / sum_base if sum_base else None,
            "opt_speedup_over_baseline_projected": sum_base / sum_opt if sum_opt else None,
            "baseline_projected_over_pure": sum_base / sum_pure if sum_pure else None,
            "opt_projected_over_pure": sum_opt / sum_pure if sum_pure else None,
            "opt_projected_over_fixed": sum_opt / sum_fixed if sum_fixed else None,
            "opt_projected_over_bf16": sum_opt / sum_bf16 if sum_bf16 else None,
        }
        summary_rows.append(summary)
        log("[SUMMARY] " + json.dumps(summary, ensure_ascii=False))

    write_csv(out_dir / "summary_opt_v23.csv", summary_rows)
    json.dump(summary_rows, open(out_dir / "summary_opt_v23.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {out_dir / 'layer_latency_opt_v23.csv'}")
    log(f"[CSV] {out_dir / 'summary_opt_v23.csv'}")
    log(f"[CSV] {out_dir / 'opt_info_v23.csv'}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
