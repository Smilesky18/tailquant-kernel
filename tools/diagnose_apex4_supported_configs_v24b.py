import os
import re
import json
import csv
import traceback
from pathlib import Path

import torch


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


def dump_context(path, pattern, out_lines, radius=8):
    lines = Path(path).read_text(errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            lo = max(0, i - radius)
            hi = min(len(lines), i + radius + 1)
            out_lines.append(f"\n===== {path}:{i+1} =====")
            for j in range(lo, hi):
                out_lines.append(f"{j+1:05d}: {lines[j]}")


def qfactory_uint8_smoke(out_dir):
    from qfactory.kernels.gemm_w4a4 import gemm_int4_int4_nt

    rows = []
    shapes = [
        (2048, 4096, 4096, "q_or_o_b16"),
        (2048, 4096, 1024, "k_or_v_b16"),
        (2048, 4096, 12288, "gate_or_up_b16"),
        (2048, 12288, 4096, "down_b16"),
        (8192, 4096, 4096, "q_or_o_b64"),
        (8192, 4096, 1024, "k_or_v_b64"),
        (8192, 4096, 12288, "gate_or_up_b64"),
        (8192, 12288, 4096, "down_b64"),
    ]

    def bench(fn, warmup=5, iters=30):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            fn()
        ed.record()
        torch.cuda.synchronize()
        return float(st.elapsed_time(ed) / iters)

    for M, K, N, tag in shapes:
        row = {"tag": tag, "M": M, "K": K, "N": N}
        try:
            A = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
            B = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
            C = torch.empty((M, N), device="cuda", dtype=torch.int32)

            def fn():
                return gemm_int4_int4_nt(A, B, C)

            ms = bench(fn)
            row["qfactory_raw_uint8_ms"] = ms
            log(f"[QFACTORY_UINT8_OK] {tag} M={M} K={K} N={N} ms={ms:.6f}")
        except Exception as e:
            row["qfactory_error"] = repr(e)
            log(f"[QFACTORY_UINT8_FAILED] {tag} {repr(e)}")
            traceback.print_exc()

        rows.append(row)
        write_csv(out_dir / "qfactory_uint8_smoke_v24b.csv", rows)

    return rows


def main():
    apex4 = Path(os.environ["APEX4"])
    out_dir = Path(os.environ["OUT"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"[APEX4] {apex4}")
    log(f"[OUT] {out_dir}")
    log(f"[DEVICE] {torch.cuda.get_device_name(0)}")
    log(f"[TORCH] {torch.__version__} cuda={torch.version.cuda}")

    files = [
        apex4 / "kernels/csrc/__init__.py",
        apex4 / "kernels/csrc/pybind.cpp",
        apex4 / "kernels/csrc/apex4_gemm_w4a4_group.cu",
        apex4 / "kernels/csrc/apex4_gemm_w4a4_channel.cu",
        apex4 / "kernels/test_w4a4.py",
        apex4 / "kernels/README.md",
    ]

    ctx = []
    for f in files:
        if f.exists():
            dump_context(f, r"No kernel implementation|thread_k|thread_n|groupsize|w4a4_mul|run_problem|apex4_gemm", ctx, radius=6)

    (out_dir / "apex4_dispatch_context_v24b.txt").write_text("\n".join(ctx))
    log(f"[DISPATCH_CONTEXT] {out_dir / 'apex4_dispatch_context_v24b.txt'}")

    # 粗略抽取源码里出现过的 thread_k/thread_n/groupsize 常量。
    src = "\n".join([f.read_text(errors="ignore") for f in files if f.exists()])
    constants = {
        "thread_k_values": sorted(set(int(x) for x in re.findall(r"thread_k\s*[=<>!]+\s*(\d+)", src))),
        "thread_n_values": sorted(set(int(x) for x in re.findall(r"thread_n\s*[=<>!]+\s*(\d+)", src))),
        "groupsize_values": sorted(set(int(x) for x in re.findall(r"groupsize\s*[=<>!]+\s*(-?\d+)", src))),
    }
    (out_dir / "apex4_extracted_constants_v24b.json").write_text(json.dumps(constants, indent=2, ensure_ascii=False))
    log("[EXTRACTED_CONSTANTS] " + json.dumps(constants, ensure_ascii=False))

    # 运行官方 test_w4a4.py --help 和源码入口，不要求成功；只收集它默认用哪些配置。
    import subprocess, sys

    test_py = apex4 / "kernels/test_w4a4.py"
    for cmd_name, cmd in [
        ("help", [sys.executable, str(test_py), "--help"]),
        ("default", [sys.executable, str(test_py)]),
    ]:
        log(f"[RUN_OFFICIAL_{cmd_name.upper()}] {' '.join(cmd)}")
        p = subprocess.run(
            cmd,
            cwd=str(apex4 / "kernels"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=600,
        )
        (out_dir / f"official_test_w4a4_{cmd_name}_v24b.log").write_text(p.stdout)
        log(f"[OFFICIAL_{cmd_name.upper()}_RC] {p.returncode}")

    # 修复 QFactory uint8 对照，确认不是 backend 本身坏了。
    qfactory_uint8_smoke(out_dir)

    log("[DONE]")


if __name__ == "__main__":
    main()
