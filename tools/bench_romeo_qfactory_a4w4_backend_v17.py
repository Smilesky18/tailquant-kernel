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
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument(
        "--shapes",
        default="2048,4096,4096;2048,4096,12288;2048,12288,4096;8192,4096,4096;8192,4096,12288;8192,12288,4096",
    )
    return p.parse_args()


def parse_shapes(s):
    out = []
    for item in s.split(";"):
        item = item.strip()
        if item:
            M, K, N = [int(x) for x in item.split(",")]
            out.append((M, K, N))
    return out


def tops(M, K, N, ms):
    return (2.0 * M * K * N) / (ms / 1000.0) / 1e12


def time_graph_or_eager(fn, warmup, iters, device):
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
        return float(start.elapsed_time(end) / iters), "graph"
    except Exception as e:
        log(f"[WARN] graph failed: {repr(e)}; fallback eager")
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end) / iters), "eager"


def try_load_current_ext():
    try:
        import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
        BASE = B.BASE
        cutlass = BASE.find_cutlass_path("/data/yzy/quarot-gpt-2/third_party/cutlass")
        ext = BASE.load_ext(cutlass, verbose=False)
        log(f"[CURRENT_EXT] {ext}")
        return ext
    except Exception as e:
        log(f"[CURRENT_EXT_SKIP] {repr(e)}")
        return None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[DEVICE] {torch.cuda.get_device_name(device)}")
    log(f"[CAPABILITY] {torch.cuda.get_device_capability(device)}")
    log(f"[QFACTORY_ARCH] {os.environ.get('QFACTORY_ARCH')}")
    log(f"[QFACTORY_CACHE_DIR] {os.environ.get('QFACTORY_CACHE_DIR')}")
    log("[NOTE] RoMeo/QFactory backend benchmark: raw A4W4 int32 and fused per-channel A4W4->bf16")

    from qfactory.kernels.gemm_w4a4 import gemm_int4_int4_nt
    from qfactory.kernels.gemm_w4a4_mixed_precision import gemm_mixed_nt_perchannel

    current_ext = try_load_current_ext()

    rows = []

    for M, K, N in parse_shapes(args.shapes):
        log(f"\n[CASE] M={M} K={K} N={N}")
        torch.cuda.empty_cache()

        A_pack = torch.randint(0, 256, (M, K // 2), device=device, dtype=torch.uint8).contiguous()
        B_pack = torch.randint(0, 256, (N, K // 2), device=device, dtype=torch.uint8).contiguous()

        A_scale_bf16 = torch.ones(M, device=device, dtype=torch.bfloat16)
        B_scale_bf16 = torch.ones(N, device=device, dtype=torch.bfloat16)
        Y_bf16 = torch.empty(M, N, device=device, dtype=torch.bfloat16)

        C_i32_qfactory = torch.empty(M, N, device=device, dtype=torch.int32)

        A_scale_f32 = torch.ones(M, device=device, dtype=torch.float32)
        B_scale_f32 = torch.ones(N, device=device, dtype=torch.float32)
        C_i32_current = torch.empty(M, N, device=device, dtype=torch.int32)
        Y_fp16_current = torch.empty(M, N, device=device, dtype=torch.float16)

        def qfactory_raw_a4w4():
            return gemm_int4_int4_nt(A_pack, B_pack, C_i32_qfactory)

        def qfactory_fused_a4w4():
            return gemm_mixed_nt_perchannel(
                A_pack,
                A_scale_bf16,
                B_pack,
                B_scale_bf16,
                Y_bf16,
                name="a4w4",
            )

        timings = {}

        # 第一次调用会触发 JIT compile + tune，单独预热并同步。
        log("[JIT_WARMUP] qfactory_raw_a4w4")
        qfactory_raw_a4w4()
        torch.cuda.synchronize(device)

        log("[JIT_WARMUP] qfactory_fused_a4w4")
        qfactory_fused_a4w4()
        torch.cuda.synchronize(device)

        ms, mode = time_graph_or_eager(qfactory_raw_a4w4, args.warmup, args.iters, device)
        timings["qfactory_raw_a4w4_ms"] = ms
        timings["qfactory_raw_a4w4_timing"] = mode
        log(f"[TIME] qfactory_raw_a4w4 {ms:.6f} ms mode={mode}")

        ms, mode = time_graph_or_eager(qfactory_fused_a4w4, args.warmup, args.iters, device)
        timings["qfactory_fused_a4w4_bf16_ms"] = ms
        timings["qfactory_fused_a4w4_bf16_timing"] = mode
        log(f"[TIME] qfactory_fused_a4w4_bf16 {ms:.6f} ms mode={mode}")

        if current_ext is not None:
            def current_gemm_only():
                current_ext.cutlass_s4_gemm(
                    A_pack,
                    B_pack,
                    C_i32_current,
                    M,
                    N,
                    K,
                )

            def current_gemm_plus_scale():
                current_ext.cutlass_s4_gemm(
                    A_pack,
                    B_pack,
                    C_i32_current,
                    M,
                    N,
                    K,
                )
                current_ext.scale_i32_to_fp16(
                    C_i32_current,
                    A_scale_f32,
                    B_scale_f32,
                    Y_fp16_current,
                )

            ms, mode = time_graph_or_eager(current_gemm_only, args.warmup, args.iters, device)
            timings["current_cutlass_s4_gemm_ms"] = ms
            timings["current_cutlass_s4_gemm_timing"] = mode
            log(f"[TIME] current_cutlass_s4_gemm {ms:.6f} ms mode={mode}")

            ms, mode = time_graph_or_eager(current_gemm_plus_scale, args.warmup, args.iters, device)
            timings["current_cutlass_s4_gemm_plus_scale_ms"] = ms
            timings["current_cutlass_s4_gemm_plus_scale_timing"] = mode
            log(f"[TIME] current_cutlass_s4_gemm_plus_scale {ms:.6f} ms mode={mode}")

        row = {
            "M": M,
            "K": K,
            "N": N,
            **timings,
        }

        for k, v in list(timings.items()):
            if k.endswith("_ms"):
                row[k.replace("_ms", "_tops_equiv")] = tops(M, K, N, v)

        if "current_cutlass_s4_gemm_plus_scale_ms" in timings:
            row["qfactory_fused_over_current_total"] = (
                timings["qfactory_fused_a4w4_bf16_ms"] / timings["current_cutlass_s4_gemm_plus_scale_ms"]
            )
            row["qfactory_raw_over_current_gemm"] = (
                timings["qfactory_raw_a4w4_ms"] / timings["current_cutlass_s4_gemm_ms"]
            )

        rows.append(row)
        log("[RESULT] " + json.dumps(row, indent=2, ensure_ascii=False))

        del A_pack, B_pack, A_scale_bf16, B_scale_bf16, Y_bf16
        del C_i32_qfactory, A_scale_f32, B_scale_f32, C_i32_current, Y_fp16_current
        torch.cuda.empty_cache()

    csv_path = out_dir / "romeo_qfactory_a4w4_backend_v17.csv"
    json_path = out_dir / "romeo_qfactory_a4w4_backend_v17.json"

    fields = sorted({k for r in rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
