import argparse
import csv
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--cutlass_path", default="/data/yzy/quarot-gpt-2/third_party/cutlass")
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
            out.append(tuple(int(x) for x in item.split(",")))
    return out


def tops(M, K, N, ms):
    return (2.0 * M * K * N) / (ms / 1000.0) / 1e12


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


def patch_source(src: str, tb, warp, inst=(8, 8, 32), sm="Sm80"):
    text = src

    text, n1 = re.subn(
        r"using\s+SmArch\s*=\s*cutlass::arch::Sm[0-9]+;",
        f"using SmArch = cutlass::arch::{sm};",
        text,
    )
    text, n2 = re.subn(
        r"using\s+ThreadblockShape\s*=\s*cutlass::gemm::GemmShape<\s*\d+\s*,\s*\d+\s*,\s*\d+\s*>;",
        f"using ThreadblockShape = cutlass::gemm::GemmShape<{tb[0]}, {tb[1]}, {tb[2]}>;",
        text,
    )
    text, n3 = re.subn(
        r"using\s+WarpShape\s*=\s*cutlass::gemm::GemmShape<\s*\d+\s*,\s*\d+\s*,\s*\d+\s*>;",
        f"using WarpShape = cutlass::gemm::GemmShape<{warp[0]}, {warp[1]}, {warp[2]}>;",
        text,
    )
    text, n4 = re.subn(
        r"using\s+InstructionShape\s*=\s*cutlass::gemm::GemmShape<\s*\d+\s*,\s*\d+\s*,\s*\d+\s*>;",
        f"using InstructionShape = cutlass::gemm::GemmShape<{inst[0]}, {inst[1]}, {inst[2]}>;",
        text,
    )

    if min(n1, n2, n3, n4) == 0:
        raise RuntimeError(f"source patch failed: n1={n1} n2={n2} n3={n3} n4={n4}")

    return text


def load_baseline_ext(cutlass_path, device):
    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

    BASE = B.BASE
    cutlass = BASE.find_cutlass_path(cutlass_path)
    ext = BASE.load_ext(cutlass, verbose=False)
    return ext, Path(ext.__file__).parent, cutlass


def compile_variant(name, src_dir, work_dir, cutlass_path, tb, warp, inst=(8, 8, 32), sm="Sm80"):
    vdir = work_dir / name
    vdir.mkdir(parents=True, exist_ok=True)

    cuda_src = (src_dir / "cuda.cu").read_text()
    main_src = (src_dir / "main.cpp").read_text()

    cuda_patched = patch_source(cuda_src, tb=tb, warp=warp, inst=inst, sm=sm)

    cuda_path = vdir / "cuda.cu"
    main_path = vdir / "main.cpp"
    cuda_path.write_text(cuda_patched)
    main_path.write_text(main_src)

    build_dir = vdir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    extra_cflags = [
        "-O3",
        "-std=c++17",
    ]
    extra_cuda_cflags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-DCUTLASS_ARCH_MMA_SM80_SUPPORTED=1",
    ]

    ext = load(
        name=name,
        sources=[str(main_path), str(cuda_path)],
        extra_include_paths=[
            str(Path(cutlass_path) / "include"),
            str(Path(cutlass_path) / "tools/util/include"),
        ],
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=False,
    )
    return ext, vdir


def bench_variant(ext, M, K, N, warmup, iters, eps, device):
    torch.cuda.empty_cache()

    A_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    W_kn_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)
    W_T = W_kn_fp16.contiguous()

    A_bf16 = A_fp16.to(torch.bfloat16)
    W_bf16 = W_kn_fp16.to(torch.bfloat16)

    A_pack = torch.empty((M * K + 1) // 2, device=device, dtype=torch.uint8)
    B_col = torch.empty((K * N + 1) // 2, device=device, dtype=torch.uint8)
    a_scale = torch.empty(M, device=device, dtype=torch.float32)
    w_scale = torch.empty(N, device=device, dtype=torch.float32)
    C_i32 = torch.empty(M, N, device=device, dtype=torch.int32)
    Y_fp16 = torch.empty(M, N, device=device, dtype=torch.float16)

    # W_T logical [K,N], A logical [M,K]
    ext.pack_weight_colmajor_s4(W_T, B_col, w_scale, eps)
    ext.pack_a_full_s4(A_fp16, A_pack, a_scale, eps)
    torch.cuda.synchronize(device)

    def torch_bf16_mm():
        return torch.mm(A_bf16, W_bf16)

    def int4_gemm():
        ext.cutlass_s4_gemm(A_pack, B_col, C_i32, M, N, K)

    def int4_total():
        ext.cutlass_s4_gemm(A_pack, B_col, C_i32, M, N, K)
        ext.scale_i32_to_fp16(C_i32, a_scale, w_scale, Y_fp16)

    bf16_ms = time_graph(torch_bf16_mm, warmup, iters, device)
    gemm_ms = time_graph(int4_gemm, warmup, iters, device)
    total_ms = time_graph(int4_total, warmup, iters, device)

    row = {
        "M": M,
        "K": K,
        "N": N,
        "torch_bf16_mm_ms": bf16_ms,
        "int4_gemm_only_ms": gemm_ms,
        "int4_gemm_plus_scale_ms": total_ms,
        "torch_bf16_tops": tops(M, K, N, bf16_ms),
        "int4_gemm_tops_equiv": tops(M, K, N, gemm_ms),
        "int4_total_tops_equiv": tops(M, K, N, total_ms),
        "int4_gemm_over_bf16": gemm_ms / bf16_ms,
        "int4_total_over_bf16": total_ms / bf16_ms,
    }

    del A_fp16, W_kn_fp16, W_T, A_bf16, W_bf16
    del A_pack, B_col, a_scale, w_scale, C_i32, Y_fp16
    torch.cuda.empty_cache()

    return row


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        major, minor = torch.cuda.get_device_capability(device)
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[DEVICE] {torch.cuda.get_device_name(device)}")
    log(f"[CAPABILITY] {torch.cuda.get_device_capability(device)}")
    log(f"[TORCH_CUDA_ARCH_LIST] {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
    log(f"[OUT] {out_dir}")
    log(f"[WORK] {work_dir}")

    baseline_ext, src_dir, cutlass = load_baseline_ext(args.cutlass_path, device)
    log(f"[BASELINE_EXT_DIR] {src_dir}")
    log(f"[CUTLASS] {cutlass}")

    variants = [
        ("baseline_tb128x128x128_w64x64x128", (128, 128, 128), (64, 64, 128)),
        ("tb128x256x128_w64x64x128", (128, 256, 128), (64, 64, 128)),
        ("tb256x128x128_w64x64x128", (256, 128, 128), (64, 64, 128)),
        ("tb128x128x256_w64x64x128", (128, 128, 256), (64, 64, 128)),
        ("tb64x128x128_w32x64x128", (64, 128, 128), (32, 64, 128)),
        ("tb128x64x128_w64x32x128", (128, 64, 128), (64, 32, 128)),
    ]

    shapes = parse_shapes(args.shapes)
    rows = []
    compile_rows = []

    for name, tb, warp in variants:
        safe_name = "s4gemm_v14_" + re.sub(r"[^A-Za-z0-9_]", "_", name)
        log(f"\n[COMPILE] {name} tb={tb} warp={warp}")

        try:
            if name.startswith("baseline_"):
                ext = baseline_ext
                vdir = src_dir
                compile_ok = True
                compile_error = ""
            else:
                ext, vdir = compile_variant(
                    safe_name,
                    src_dir=src_dir,
                    work_dir=work_dir,
                    cutlass_path=cutlass,
                    tb=tb,
                    warp=warp,
                    inst=(8, 8, 32),
                    sm="Sm80",
                )
                compile_ok = True
                compile_error = ""

            log(f"[COMPILE_OK] {name} dir={vdir}")
        except Exception as e:
            compile_ok = False
            compile_error = repr(e)
            log(f"[COMPILE_FAIL] {name} error={compile_error}")
            compile_rows.append({
                "variant": name,
                "tb": str(tb),
                "warp": str(warp),
                "compile_ok": False,
                "compile_error": compile_error,
            })
            continue

        compile_rows.append({
            "variant": name,
            "tb": str(tb),
            "warp": str(warp),
            "compile_ok": True,
            "compile_error": "",
        })

        for M, K, N in shapes:
            log(f"[BENCH] variant={name} M={M} K={K} N={N}")
            try:
                r = bench_variant(
                    ext=ext,
                    M=M,
                    K=K,
                    N=N,
                    warmup=args.warmup,
                    iters=args.iters,
                    eps=args.eps,
                    device=device,
                )
                r.update({
                    "variant": name,
                    "tb": str(tb),
                    "warp": str(warp),
                    "compile_ok": True,
                    "runtime_ok": True,
                    "runtime_error": "",
                })
                rows.append(r)
                log("[RESULT] " + json.dumps(r, ensure_ascii=False))
            except Exception as e:
                r = {
                    "variant": name,
                    "tb": str(tb),
                    "warp": str(warp),
                    "M": M,
                    "K": K,
                    "N": N,
                    "compile_ok": True,
                    "runtime_ok": False,
                    "runtime_error": repr(e),
                }
                rows.append(r)
                log("[RUNTIME_FAIL] " + json.dumps(r, ensure_ascii=False))

    csv_path = out_dir / "cutlass_s4_gemm_tile_autotune_v14.csv"
    json_path = out_dir / "cutlass_s4_gemm_tile_autotune_v14.json"
    compile_csv = out_dir / "cutlass_s4_gemm_tile_autotune_compile_v14.csv"

    if rows:
        fields = sorted({k for r in rows for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)

    if compile_rows:
        fields = list(compile_rows[0].keys())
        with open(compile_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(compile_rows)

    # best per shape
    valid = [r for r in rows if r.get("runtime_ok") and "int4_gemm_only_ms" in r]
    best_rows = []
    for M, K, N in shapes:
        cand = [r for r in valid if r["M"] == M and r["K"] == K and r["N"] == N]
        if not cand:
            continue
        best = min(cand, key=lambda x: x["int4_gemm_only_ms"])
        best_rows.append(best)
        log(
            "[BEST] "
            + json.dumps({
                "M": M,
                "K": K,
                "N": N,
                "variant": best["variant"],
                "int4_gemm_only_ms": best["int4_gemm_only_ms"],
                "torch_bf16_mm_ms": best["torch_bf16_mm_ms"],
                "int4_gemm_over_bf16": best["int4_gemm_over_bf16"],
                "int4_gemm_tops_equiv": best["int4_gemm_tops_equiv"],
            }, ensure_ascii=False)
        )

    best_csv = out_dir / "cutlass_s4_gemm_tile_autotune_best_v14.csv"
    if best_rows:
        fields = sorted({k for r in best_rows for k in r.keys()})
        with open(best_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(best_rows)

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[COMPILE_CSV] {compile_csv}")
    log(f"[BEST_CSV] {best_csv}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
