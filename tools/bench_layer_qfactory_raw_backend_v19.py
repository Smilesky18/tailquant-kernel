import argparse
import copy
import csv
import json
import os
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8
from qfactory.kernels.gemm_w4a4 import gemm_int4_int4_nt


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--ratio", type=float, default=0.05)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    return p.parse_args()


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


def compare_tensors(a, b):
    d = (a.float() - b.float()).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "rmse": float(torch.sqrt((d * d).mean()).item()),
    }


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

        patched.append({
            "name": name,
            "K": int(mod.K),
            "N": int(mod.N),
            "is_split": bool(getattr(mod, "is_split", False)),
            "R": int(getattr(mod, "R", 0)),
            "ratio": float(getattr(mod, "ratio", 0.0)),
        })

    log("[PATCH_QFACTORY_RAW_BACKEND] " + json.dumps(patched, indent=2))
    return patched


def bench_graph(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

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
    return ms


def build_layer_pair(base_layer, B, main_ext, layout_ext, policy_pack_ext, mode, ratio, eps, device):
    current = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    current, current_records = V8.patch_layer_with_real_policy(
        layer=current,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode=mode,
        ratio=ratio,
        eps=eps,
        device=device,
    )
    current.to(device=device).eval()

    qf = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    qf, qf_records = V8.patch_layer_with_real_policy(
        layer=qf,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode=mode,
        ratio=ratio,
        eps=eps,
        device=device,
    )
    qf.to(device=device).eval()
    patch_qfactory_raw_backend(qf)

    return current, qf, current_records, qf_records


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[BATCHES] {args.batches}")
    log(f"[RATIO] {args.ratio}")
    log(f"[QFACTORY_ARCH] {os.environ.get('QFACTORY_ARCH')}")
    log(f"[QFACTORY_CACHE_DIR] {os.environ.get('QFACTORY_CACHE_DIR')}")
    log("[NOTE] Compare current backend vs QFactory raw A4W4 backend for Pure and Split layer.")

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
    base_layer = layers[args.layer_idx]

    log("[BUILD] pure layers")
    pure_current, pure_qf, pure_cur_rec, pure_qf_rec = build_layer_pair(
        base_layer,
        B,
        main_ext,
        layout_ext,
        policy_pack_ext,
        mode="pure",
        ratio=0.0,
        eps=args.eps,
        device=device,
    )

    log("[BUILD] split layers")
    split_current, split_qf, split_cur_rec, split_qf_rec = build_layer_pair(
        base_layer,
        B,
        main_ext,
        layout_ext,
        policy_pack_ext,
        mode="dual_policy",
        ratio=args.ratio,
        eps=args.eps,
        device=device,
    )

    rows = []
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    for batch in batches:
        log(f"\n[CASE] batch={batch} seq_len={args.seq_len}")

        hidden = torch.randn(
            batch,
            args.seq_len,
            hidden_size,
            device=device,
            dtype=torch.float16,
        )
        position_ids = V8.make_position_ids(batch, args.seq_len, device)
        pe = V8.build_position_embeddings(model, hidden, position_ids, torch.float16)

        # JIT compile QFactory kernels before graph capture
        log("[JIT_WARMUP] pure_qfactory")
        _ = run_layer_once(pure_qf, hidden, position_ids, pe)
        torch.cuda.synchronize(device)

        log("[JIT_WARMUP] split_qfactory")
        _ = run_layer_once(split_qf, hidden, position_ids, pe)
        torch.cuda.synchronize(device)

        with torch.no_grad():
            pure_ref = run_layer_once(pure_current, hidden, position_ids, pe)
            pure_new = run_layer_once(pure_qf, hidden, position_ids, pe)
            split_ref = run_layer_once(split_current, hidden, position_ids, pe)
            split_new = run_layer_once(split_qf, hidden, position_ids, pe)
            torch.cuda.synchronize(device)

        pure_diff = compare_tensors(pure_ref, pure_new)
        split_diff = compare_tensors(split_ref, split_new)

        log("[CHECK_PURE] " + json.dumps(pure_diff, ensure_ascii=False))
        log("[CHECK_SPLIT] " + json.dumps(split_diff, ensure_ascii=False))

        timings = {}

        timings["pure_current_ms"] = bench_graph(
            lambda: run_layer_once(pure_current, hidden, position_ids, pe),
            args.warmup,
            args.iters,
            device,
        )
        log(f"[TIME] pure_current {timings['pure_current_ms']:.6f} ms")

        timings["pure_qfactory_raw_ms"] = bench_graph(
            lambda: run_layer_once(pure_qf, hidden, position_ids, pe),
            args.warmup,
            args.iters,
            device,
        )
        log(f"[TIME] pure_qfactory_raw {timings['pure_qfactory_raw_ms']:.6f} ms")

        timings["split_current_ms"] = bench_graph(
            lambda: run_layer_once(split_current, hidden, position_ids, pe),
            args.warmup,
            args.iters,
            device,
        )
        log(f"[TIME] split_current {timings['split_current_ms']:.6f} ms")

        timings["split_qfactory_raw_ms"] = bench_graph(
            lambda: run_layer_once(split_qf, hidden, position_ids, pe),
            args.warmup,
            args.iters,
            device,
        )
        log(f"[TIME] split_qfactory_raw {timings['split_qfactory_raw_ms']:.6f} ms")

        row = {
            "model": args.model,
            "layer_idx": args.layer_idx,
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            **timings,
            "pure_qfactory_over_current": timings["pure_qfactory_raw_ms"] / timings["pure_current_ms"],
            "split_qfactory_over_current": timings["split_qfactory_raw_ms"] / timings["split_current_ms"],
            "pure_current_over_split_current": timings["split_current_ms"] / timings["pure_current_ms"],
            "pure_qfactory_over_split_qfactory": timings["split_qfactory_raw_ms"] / timings["pure_qfactory_raw_ms"],
            "pure_max_abs_diff": pure_diff["max_abs"],
            "pure_mean_abs_diff": pure_diff["mean_abs"],
            "pure_rmse_diff": pure_diff["rmse"],
            "split_max_abs_diff": split_diff["max_abs"],
            "split_mean_abs_diff": split_diff["mean_abs"],
            "split_rmse_diff": split_diff["rmse"],
        }

        rows.append(row)
        log("[RESULT] " + json.dumps(row, indent=2, ensure_ascii=False))

    csv_path = out_dir / "layer_qfactory_raw_backend_v19.csv"
    json_path = out_dir / "layer_qfactory_raw_backend_v19.json"

    fields = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)

    meta = {
        "pure_current_patch_records": pure_cur_rec,
        "split_current_patch_records": split_cur_rec,
        "note": "QFactory raw A4W4 replaces only dense W4A4 GEMM. Existing pack and scale_i32_to_fp16 are kept for correctness and FP16 output.",
    }
    json.dump(meta, open(out_dir / "layer_qfactory_raw_backend_v19_meta.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
