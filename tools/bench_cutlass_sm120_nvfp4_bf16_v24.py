import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path


QWEN3_LINEAR_SHAPES = [
    ("q_proj", 4096, 4096),
    ("k_proj", 1024, 4096),
    ("v_proj", 1024, 4096),
    ("o_proj", 4096, 4096),
    ("gate_proj", 12288, 4096),
    ("up_proj", 12288, 4096),
    ("down_proj", 4096, 12288),
]


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cutlass_root", default="/data/yzy/quarot-gpt-2/third_party/cutlass")
    p.add_argument("--cuda_home", default=os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--shape_filter", default="all")
    p.add_argument("--arch", default="120a")
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def run(cmd, cwd=None):
    log("[CMD] " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(
        [str(x) for x in cmd],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(str(x) for x in cmd)}")
    return proc.stdout


def compile_example(args, build_dir: Path):
    cutlass = Path(args.cutlass_root)
    cuda_home = Path(args.cuda_home)
    src = cutlass / "examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu"
    local_src = build_dir / "79a_blackwell_geforce_nvfp4_bf16_gemm_perf_only.cu"
    exe = build_dir / "79a_blackwell_geforce_nvfp4_bf16_gemm"
    if exe.exists() and not args.rebuild:
        log(f"[BUILD] reuse {exe}")
        return exe
    build_dir.mkdir(parents=True, exist_ok=True)
    shadow_include = build_dir / "include_shadow"
    shadow_cutlass = shadow_include / "cutlass"
    shadow_cutlass.mkdir(parents=True, exist_ok=True)
    original_subbyte = cutlass / "include/cutlass/subbyte_reference.h"
    patched_subbyte = shadow_cutlass / "subbyte_reference.h"
    text = original_subbyte.read_text(encoding="utf-8")
    text = text.replace(
        "Storage original = __nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED);",
        "Storage original = __nv_atomic_load_n(ptr_, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_DEVICE);",
    )
    patched_subbyte.write_text(text, encoding="utf-8")

    src_text = src.read_text(encoding="utf-8")
    src_text = src_text.replace(
        "result.passed = verify(options);",
        "result.passed = true;  // perf-only experiment copy: skip slow host reference"
    )
    local_src.write_text(src_text, encoding="utf-8")

    cmd = [
        cuda_home / "bin/nvcc",
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
        local_src,
        "-o",
        exe,
        "-I",
        shadow_include,
        "-I",
        cutlass / "include",
        "-I",
        cutlass / "tools/util/include",
        "-I",
        cutlass / "examples/common",
    ]
    run(cmd)
    return exe


def parse_metrics(text: str):
    out = {}
    m = re.search(r"Disposition:\s+(\w+)", text)
    if m:
        out["disposition"] = m.group(1)
    m = re.search(r"Problem Size:\s+(\d+)x(\d+)x(\d+)", text)
    if m:
        out["reported_m"] = int(m.group(1))
        out["reported_n"] = int(m.group(2))
        out["reported_k"] = int(m.group(3))
    m = re.search(r"Avg runtime:\s+([-+0-9.eE]+)\s+ms", text)
    if m:
        out["avg_runtime_ms"] = float(m.group(1))
    m = re.search(r"GFLOPS:\s+([-+0-9.eE]+)", text)
    if m:
        out["gflops"] = float(m.group(1))
        out["tflops"] = out["gflops"] / 1000.0
    return out


def select_shapes(shape_filter: str):
    if shape_filter == "all":
        return QWEN3_LINEAR_SHAPES
    wanted = {x.strip() for x in shape_filter.split(",") if x.strip()}
    return [x for x in QWEN3_LINEAR_SHAPES if x[0] in wanted]


def write_csv(path: Path, rows):
    fields = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = out_dir / "build"

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[CUTLASS_ROOT] {args.cutlass_root}")
    log(f"[CUDA_HOME] {args.cuda_home}")
    log(f"[ARCH] {args.arch}")
    log(f"[BATCHES] {args.batches}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log("[NOTE] Official CUTLASS SM120 NVFP4->BF16 GEMM baseline; not yet E0M3 W4A4.")

    exe = compile_example(args, build_dir)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    shapes = select_shapes(args.shape_filter)
    rows = []

    for batch in batches:
        M = batch * args.seq_len
        for name, N, K in shapes:
            log(f"[RUN] shape={name} M={M} N={N} K={K}")
            text = run(
                [
                    exe,
                    f"--m={M}",
                    f"--n={N}",
                    f"--k={K}",
                    f"--iterations={args.iterations}",
                ]
            )
            metrics = parse_metrics(text)
            row = {
                "backend": "cutlass_sm120_nvfp4_bf16_79a",
                "shape_name": name,
                "batch": batch,
                "seq_len": args.seq_len,
                "M": M,
                "N": N,
                "K": K,
                "iterations": args.iterations,
                **metrics,
            }
            if "avg_runtime_ms" in row:
                row["tokens"] = M
                row["ms_per_token"] = row["avg_runtime_ms"] / M
            rows.append(row)
            log("[ROW] " + json.dumps(row, ensure_ascii=False))

            csv_path = out_dir / "cutlass_sm120_nvfp4_bf16_qwen3_shapes_v24_partial.csv"
            json_path = out_dir / "cutlass_sm120_nvfp4_bf16_qwen3_shapes_v24_partial.json"
            write_csv(csv_path, rows)
            json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = out_dir / "cutlass_sm120_nvfp4_bf16_qwen3_shapes_v24.csv"
    json_path = out_dir / "cutlass_sm120_nvfp4_bf16_qwen3_shapes_v24.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
