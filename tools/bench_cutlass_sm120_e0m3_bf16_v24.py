import argparse
import csv
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path

from bench_cutlass_sm120_nvfp4_bf16_v24 import (
    compile_example,
    parse_metrics,
    QWEN3_LINEAR_SHAPES,
    select_shapes,
    write_csv,
)


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cutlass_root", default="/data/yzy/quarot-gpt-2/third_party/cutlass")
    p.add_argument("--cuda_home", default=os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    p.add_argument("--out_dir", required=True)
    p.add_argument("--base_build_dir", default="")
    p.add_argument("--batches", default="16,64,256")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--shape_filter", default="all")
    p.add_argument("--arch", default="120a")
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def run(cmd):
    log("[CMD] " + " ".join(str(x) for x in cmd))
    proc = subprocess.run(
        [str(x) for x in cmd],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed rc={proc.returncode}: {' '.join(str(x) for x in cmd)}")
    return proc.stdout


def patch_e0m3(binary: Path, cuobjdump: Path, patched: Path):
    sass = run([cuobjdump, "--dump-sass", binary])
    pattern = re.compile(
        r"OMMA\.SF\.16864\.F32\.E2M1\.E2M1\.UE4M3\.4X[^\n]*"
        r"/\* 0x([0-9a-fA-F]{16}) \*/\s*\n\s*"
        r"/\* 0x([0-9a-fA-F]{16}) \*/"
    )
    encodings = sorted({(int(a, 16), int(b, 16)) for a, b in pattern.findall(sass)})
    if not encodings:
        raise RuntimeError("No E2M1 x E2M1 OMMA instructions found in CUTLASS binary")

    data = binary.read_bytes()
    out = data
    replacements = 0
    patched_words = []
    for first, second in encodings:
        if ((second >> 14) & 0b11) != 0:
            raise RuntimeError(f"Unexpected nonzero E2M1 selector bits: {second:#x}")
        new_second = second | (0b11 << 14)
        needle = struct.pack("<QQ", first, second)
        repl = struct.pack("<QQ", first, new_second)
        count = out.count(needle)
        if count == 0:
            raise RuntimeError(f"Cannot find OMMA bytes {first:#x} {second:#x}")
        out = out.replace(needle, repl)
        replacements += count
        patched_words.append(f"0x{new_second:016x}")

    patched.write_bytes(out)
    os.chmod(patched, binary.stat().st_mode)
    patched_sass = run([cuobjdump, "--dump-sass", patched])
    e0m3_lines = [line for line in patched_sass.splitlines() if "OMMA.SF.16864" in line]
    return {
        "source_binary": str(binary),
        "patched_binary": str(patched),
        "encoding_count": len(encodings),
        "replacements": replacements,
        "patched_words": patched_words,
        "omma_lines": e0m3_lines[:20],
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_dir = Path(args.base_build_dir) if args.base_build_dir else out_dir / "build"

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[OUT] {out_dir}")
    log(f"[BASE_BUILD_DIR] {build_dir}")
    log("[NOTE] CUTLASS 79a perf-only binary patched from E2M1xE2M1 to E0M3xE0M3.")

    base_exe = compile_example(args, build_dir)
    patched_exe = build_dir / "79a_blackwell_geforce_e0m3_bf16_gemm_patched"
    patch_info = patch_e0m3(
        base_exe,
        Path(args.cuda_home) / "bin" / "cuobjdump",
        patched_exe,
    )
    (out_dir / "cutlass_sm120_e0m3_patch_info_v24.json").write_text(
        json.dumps(patch_info, indent=2),
        encoding="utf-8",
    )
    log("[PATCH_INFO] " + json.dumps(patch_info, ensure_ascii=False))

    rows = []
    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    shapes = select_shapes(args.shape_filter)
    for batch in batches:
        M = batch * args.seq_len
        for name, N, K in shapes:
            log(f"[RUN] shape={name} M={M} N={N} K={K}")
            text = run(
                [
                    patched_exe,
                    f"--m={M}",
                    f"--n={N}",
                    f"--k={K}",
                    f"--iterations={args.iterations}",
                ]
            )
            metrics = parse_metrics(text)
            row = {
                "backend": "cutlass_sm120_e0m3_bf16_79a_patched",
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
            write_csv(out_dir / "cutlass_sm120_e0m3_bf16_qwen3_shapes_v24_partial.csv", rows)
            (out_dir / "cutlass_sm120_e0m3_bf16_qwen3_shapes_v24_partial.json").write_text(
                json.dumps(rows, indent=2),
                encoding="utf-8",
            )

    csv_path = out_dir / "cutlass_sm120_e0m3_bf16_qwen3_shapes_v24.csv"
    json_path = out_dir / "cutlass_sm120_e0m3_bf16_qwen3_shapes_v24.json"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
