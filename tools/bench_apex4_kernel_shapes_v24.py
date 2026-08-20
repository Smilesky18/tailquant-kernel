import argparse
import csv
import json
import os
import sys
import time
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


def bench_fn(fn, warmup=5, iters=50):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def qfactory_raw_bench(M, K, N, warmup, iters):
    from qfactory.kernels.gemm_w4a4 import gemm_int4_int4_nt

    A = torch.randint(-128, 127, (M, K // 2), device="cuda", dtype=torch.int8)
    B = torch.randint(-128, 127, (N, K // 2), device="cuda", dtype=torch.int8)
    C = torch.empty((M, N), device="cuda", dtype=torch.int32)

    def fn():
        ret = gemm_int4_int4_nt(A, B, C)
        return ret

    ms = bench_fn(fn, warmup=warmup, iters=iters)
    return ms


def apex4_run_problem(apex_kernels_dir, M, K, N, thread_k, thread_n, groupsize):
    # 复用 APEX4 官方 test_w4a4.py 里的数据生成、pack、correctness、timing逻辑。
    # 这样避免我们猜它的私有 packing layout。
    sys.path.insert(0, str(apex_kernels_dir))
    import test_w4a4 as T

    T.CSV_FILE = str(Path(os.environ.get("APEX4_OUT_DIR", ".")) / "apex4_official_run_problem_raw.csv")
    tester = T.Test()

    t0 = time.time()
    tester.run_problem(M, N, K, thread_k, thread_n, groupsize)
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apex4", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    os.environ["APEX4_OUT_DIR"] = str(out)

    apex_kernels_dir = Path(args.apex4) / "kernels"

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[APEX4] {args.apex4}")
    log(f"[APEX4_KERNELS] {apex_kernels_dir}")
    log(f"[OUT] {out}")
    log(f"[DEVICE] {torch.cuda.get_device_name(0)}")
    log(f"[TORCH] {torch.__version__} cuda={torch.version.cuda}")

    sys.path.insert(0, str(apex_kernels_dir))

    rows = []
    import_results = {}

    try:
        import csrc
        import_results["csrc_import"] = "ok"
        import_results["csrc_dir"] = [x for x in dir(csrc) if "w4a4" in x or "quant" in x or "compress" in x]
        log("[CSRC_IMPORT] " + json.dumps(import_results, ensure_ascii=False))
    except Exception as e:
        import_results["csrc_import"] = "failed"
        import_results["error"] = repr(e)
        log("[CSRC_IMPORT_FAILED] " + repr(e))
        traceback.print_exc()

    # Qwen3-8B layer linear GEMM shapes, M=batch*seq.
    shapes = [
        # batch16 seq128
        (2048, 4096, 4096, "q_or_o_b16"),
        (2048, 4096, 1024, "k_or_v_b16"),
        (2048, 4096, 12288, "gate_or_up_b16"),
        (2048, 12288, 4096, "down_b16"),

        # batch64 seq128
        (8192, 4096, 4096, "q_or_o_b64"),
        (8192, 4096, 1024, "k_or_v_b64"),
        (8192, 4096, 12288, "gate_or_up_b64"),
        (8192, 12288, 4096, "down_b64"),
    ]

    for M, K, N, tag in shapes:
        log(f"\n[QFACTORY_CASE] {tag} M={M} K={K} N={N}")
        row = {
            "tag": tag,
            "M": M,
            "K": K,
            "N": N,
        }

        try:
            qf_ms = qfactory_raw_bench(M, K, N, args.warmup, args.iters)
            row["qfactory_raw_ms"] = qf_ms
            log(f"[QFACTORY_TIME] {tag} {qf_ms:.6f} ms")
        except Exception as e:
            row["qfactory_error"] = repr(e)
            log("[QFACTORY_FAILED] " + repr(e))
            traceback.print_exc()

        # APEX4 官方 kernel默认测试函数很可能是 decode/small-M 风格；
        # 这里也尝试 Qwen prefill M，失败则记录，不中断。
        for thread_k, thread_n, groupsize in [
            (128, 128, 128),
            (64, 256, 128),
            (128, 128, -1),
        ]:
            key = f"apex4_run_problem_tk{thread_k}_tn{thread_n}_g{groupsize}"
            try:
                log(f"[APEX4_CASE] {tag} M={M} N={N} K={K} thread_k={thread_k} thread_n={thread_n} groupsize={groupsize}")
                total_wall_ms = apex4_run_problem(apex_kernels_dir, M, K, N, thread_k, thread_n, groupsize)
                row[key + "_wall_ms"] = total_wall_ms
                log(f"[APEX4_DONE] {tag} {key} wall={total_wall_ms:.3f} ms")
            except Exception as e:
                row[key + "_error"] = repr(e)
                log(f"[APEX4_FAILED] {tag} {key} error={repr(e)}")
                traceback.print_exc()

        rows.append(row)
        write_csv(out / "apex4_qwen_shape_probe_v24.csv", rows)
        json.dump(rows, open(out / "apex4_qwen_shape_probe_v24.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {out / 'apex4_qwen_shape_probe_v24.csv'}")
    log(f"[JSON] {out / 'apex4_qwen_shape_probe_v24.json'}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
