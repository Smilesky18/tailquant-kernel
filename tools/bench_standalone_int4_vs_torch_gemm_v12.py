import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    p.add_argument("--shapes", default="2048,4096,4096;2048,4096,12288;2048,12288,4096;8192,4096,4096;8192,4096,12288;8192,12288,4096")
    return p.parse_args()


def time_graph(fn, warmup, iters, device):
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
    return float(start.elapsed_time(end) / iters)


def time_eager(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


def parse_shapes(s):
    out = []
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        M, K, N = [int(x) for x in item.split(",")]
        out.append((M, K, N))
    return out


def tops(M, K, N, ms):
    # dense matmul equivalent ops
    ops = 2.0 * M * K * N
    return ops / (ms / 1000.0) / 1e12


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability(device)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        log(f"[ARCH] TORCH_CUDA_ARCH_LIST={os.environ['TORCH_CUDA_ARCH_LIST']}")

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[DEVICE] {device}")
    log(f"[WARMUP] {args.warmup}")
    log(f"[ITERS] {args.iters}")
    log(f"[SHAPES] {args.shapes}")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    BASE = B.BASE
    cutlass = BASE.find_cutlass_path(args.cutlass_path)
    ext = BASE.load_ext(cutlass, verbose=False)
    log(f"[EXT] {ext}")
    log(f"[CUTLASS] {cutlass}")

    rows = []

    for M, K, N in parse_shapes(args.shapes):
        log(f"\n[CASE] M={M} K={K} N={N}")

        torch.cuda.empty_cache()

        A_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
        W_kn_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
        W_nk_fp16 = W_kn_fp16.t().contiguous()

        A_bf16 = A_fp16.to(torch.bfloat16)
        W_kn_bf16 = W_kn_fp16.to(torch.bfloat16)

        # packed INT4 buffers
        A_pack = torch.empty((M * K + 1) // 2, device=device, dtype=torch.uint8)
        B_col = torch.empty((K * N + 1) // 2, device=device, dtype=torch.uint8)
        a_scale = torch.empty(M, device=device, dtype=torch.float32)
        w_scale = torch.empty(N, device=device, dtype=torch.float32)
        C_i32 = torch.empty(M, N, device=device, dtype=torch.int32)
        Y_fp16 = torch.empty(M, N, device=device, dtype=torch.float16)

        # pack once outside timing
        ext.pack_weight_colmajor_s4(
            W_nk_fp16.t().contiguous(),
            B_col,
            w_scale,
            args.eps,
        )
        ext.pack_a_full_s4(
            A_fp16,
            A_pack,
            a_scale,
            args.eps,
        )
        torch.cuda.synchronize(device)

        def torch_fp16_mm():
            return torch.mm(A_fp16, W_kn_fp16)

        def torch_bf16_mm():
            return torch.mm(A_bf16, W_kn_bf16)

        def int4_gemm_only():
            ext.cutlass_s4_gemm(
                A_pack,
                B_col,
                C_i32,
                M,
                N,
                K,
            )

        def int4_scale_only():
            ext.scale_i32_to_fp16(
                C_i32,
                a_scale,
                w_scale,
                Y_fp16,
            )

        def int4_gemm_plus_scale():
            ext.cutlass_s4_gemm(
                A_pack,
                B_col,
                C_i32,
                M,
                N,
                K,
            )
            ext.scale_i32_to_fp16(
                C_i32,
                a_scale,
                w_scale,
                Y_fp16,
            )

        def int4_online_total():
            ext.pack_a_full_s4(
                A_fp16,
                A_pack,
                a_scale,
                args.eps,
            )
            ext.cutlass_s4_gemm(
                A_pack,
                B_col,
                C_i32,
                M,
                N,
                K,
            )
            ext.scale_i32_to_fp16(
                C_i32,
                a_scale,
                w_scale,
                Y_fp16,
            )

        # torch.mm 有时 graph/cublas workspace 行为不稳定，先 graph，失败就 eager
        timings = {}
        for name, fn in [
            ("torch_fp16_mm", torch_fp16_mm),
            ("torch_bf16_mm", torch_bf16_mm),
            ("int4_gemm_only", int4_gemm_only),
            ("int4_scale_only", int4_scale_only),
            ("int4_gemm_plus_scale", int4_gemm_plus_scale),
            ("int4_online_total", int4_online_total),
        ]:
            try:
                ms = time_graph(fn, args.warmup, args.iters, device)
                mode = "graph"
            except Exception as e:
                log(f"[WARN] graph failed for {name}: {e!r}; fallback eager")
                ms = time_eager(fn, args.warmup, args.iters, device)
                mode = "eager"
            timings[name + "_ms"] = ms
            timings[name + "_timing"] = mode
            log(f"[TIME] {name} {ms:.6f} ms mode={mode}")

        row = {
            "M": M,
            "K": K,
            "N": N,
            **timings,
            "torch_fp16_tops": tops(M, K, N, timings["torch_fp16_mm_ms"]),
            "torch_bf16_tops": tops(M, K, N, timings["torch_bf16_mm_ms"]),
            "int4_gemm_only_tops_equiv": tops(M, K, N, timings["int4_gemm_only_ms"]),
            "int4_gemm_plus_scale_tops_equiv": tops(M, K, N, timings["int4_gemm_plus_scale_ms"]),
            "int4_online_total_tops_equiv": tops(M, K, N, timings["int4_online_total_ms"]),
            "int4_gemm_over_torch_bf16": timings["int4_gemm_only_ms"] / timings["torch_bf16_mm_ms"],
            "int4_total_over_torch_bf16": timings["int4_online_total_ms"] / timings["torch_bf16_mm_ms"],
            "int4_gemm_over_torch_fp16": timings["int4_gemm_only_ms"] / timings["torch_fp16_mm_ms"],
            "int4_total_over_torch_fp16": timings["int4_online_total_ms"] / timings["torch_fp16_mm_ms"],
        }
        rows.append(row)
        log("[RESULT] " + json.dumps(row, indent=2))

        del A_fp16, W_kn_fp16, W_nk_fp16, A_bf16, W_kn_bf16
        del A_pack, B_col, a_scale, w_scale, C_i32, Y_fp16
        torch.cuda.empty_cache()

    csv_path = out_dir / "standalone_int4_vs_torch_gemm_v12.csv"
    json_path = out_dir / "standalone_int4_vs_torch_gemm_v12.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
