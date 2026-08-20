#!/usr/bin/env python3
"""Evaluate practical COMET/RoMeo-inspired optimizations for the split scheme.

This standalone smoke focuses on the split sparse-correction side. It does not
modify production kernels. It covers:

1. permutation-free split layout diagnostics
2. fused-epilogue feasibility status
3. sparse-friendly idx/layout variants, including uint16 idx
4. shape/R/kernel autotune for sparse correction
5. CUDA Graph replay timing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

import torch

ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for p in (TOOLS, ROOT):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import quarot.functional  # noqa: E402
from split_sparse_idx16_ext_v1 import load_split_sparse_idx16_ext_v1  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "experiments/kernel_quant/layer_latency_split_v1/results/split_kernel_optimization_5points_v1/smoke.json"))
    p.add_argument("--shapes", default="128,4096,4096;2048,4096,4096;2048,12288,4096;2048,4096,12288")
    p.add_argument("--Rs", default="21,30,63")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--verbose_compile", action="store_true")
    return p.parse_args()


def parse_shapes(s: str) -> list[tuple[int, int, int]]:
    out = []
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        m, n, k = [int(x) for x in item.split(",")]
        out.append((m, n, k))
    return out


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def pack_i4_from_q(q: torch.Tensor) -> torch.Tensor:
    return quarot.functional.pack_i4(q.contiguous()).contiguous()


def make_data(M: int, N: int, K: int, R: int, device: torch.device):
    B_q = torch.randint(-8, 8, (N, K), dtype=torch.int8, device=device)
    B_row_pack = pack_i4_from_q(B_q.t().contiguous())
    idx = torch.empty((M, R), dtype=torch.int32, device=device)
    for m in range(M):
        idx[m] = torch.randperm(K, device=device, dtype=torch.int32)[:R]
    sorted_pair = torch.sort(idx, dim=1)
    idx_sorted = sorted_pair.values.contiguous()
    sort_order = sorted_pair.indices
    top_q = torch.randint(-8, 8, (M, R), dtype=torch.int8, device=device)
    top_q_sorted = torch.gather(top_q, 1, sort_order).contiguous()
    top_scale = torch.empty(M, dtype=torch.float32, device=device).uniform_(0.001, 0.02)
    w_scale = torch.empty(N, dtype=torch.float32, device=device).uniform_(0.001, 0.02)
    out = torch.empty((M, N), dtype=torch.float16, device=device)
    return {
        "B_row_pack": B_row_pack,
        "idx": idx.contiguous(),
        "idx_sorted": idx_sorted,
        "top_q": top_q.contiguous(),
        "top_q_sorted": top_q_sorted,
        "top_scale": top_scale.contiguous(),
        "w_scale": w_scale.contiguous(),
        "out": out,
    }


def time_cuda(fn: Callable[[], None], warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn()
    ed.record()
    torch.cuda.synchronize(device)
    return float(st.elapsed_time(ed) / max(1, iters))


def time_graph(fn: Callable[[], None], warmup: int, iters: int, device: torch.device) -> float | None:
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize(device)
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            graph.replay()
        ed.record()
        torch.cuda.synchronize(device)
        return float(st.elapsed_time(ed) / max(1, iters))
    except Exception:
        return None


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).abs()
    denom = torch.linalg.vector_norm(a.float()).clamp_min(1e-12)
    return {
        "max_abs": float(diff.max().item()),
        "mse": float(torch.mean(diff * diff).item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
    }


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    ext = load_split_sparse_idx16_ext_v1(verbose=args.verbose_compile)

    kernels = {
        "i32_c4_b128": (ext.sparse_i32_c4_b128, "idx"),
        "i32_c8_b128": (ext.sparse_i32_c8_b128, "idx"),
        "i32_c8_b256": (ext.sparse_i32_c8_b256, "idx"),
        "i32_c16_b128": (ext.sparse_i32_c16_b128, "idx"),
        "u16_c4_b128": (ext.sparse_u16_c4_b128, "idx_u16"),
        "u16_c8_b128": (ext.sparse_u16_c8_b128, "idx_u16"),
        "u16_c8_b256": (ext.sparse_u16_c8_b256, "idx_u16"),
        "u16_c16_b128": (ext.sparse_u16_c16_b128, "idx_u16"),
    }

    report = {
        "device": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "point1_permutation_free_diagnostic": {
            "status": "implemented_by_existing_split_prepare",
            "evidence": [
                "Existing fused top-R prepare writes A_pack with top-R entries zeroed and stores top_q/idx as compact [M,R] side buffers.",
                "This avoids gathering top-R into a separate high-precision dense GEMM or permuting token rows.",
                "This smoke keeps that layout and only evaluates side-buffer/layout refinements.",
            ],
        },
        "point2_true_fused_epilogue": {
            "status": "skipped_due_to_scope_after_prior_blocker",
            "reason": "True fusion must modify QuaRot/CUTLASS GEMM epilogue to consume body_scale/top_q/idx/B_row/w_scale before the final store. Prior standalone CUTLASS output-op attempts were blocked by the standard epilogue interface, so this run focuses on prerequisites and measurements.",
            "next_action": "Implement inside the v58/v61 QuaRot-style GEMM epilogue, not as a second sparse-add kernel.",
        },
        "results": [],
    }

    for M, N, K in parse_shapes(args.shapes):
        if K > 65535:
            continue
        for R in parse_ints(args.Rs):
            if R <= 0 or R >= K:
                continue
            data = make_data(M, N, K, R, device)
            idx_u16 = torch.empty_like(data["idx"], dtype=torch.int16)
            idx_sorted_u16 = torch.empty_like(data["idx_sorted"], dtype=torch.int16)

            compress_ms = time_cuda(lambda: ext.idx_i32_to_u16(data["idx"], idx_u16), args.warmup, args.iters, device)
            ext.idx_i32_to_u16(data["idx"], idx_u16)
            ext.idx_i32_to_u16(data["idx_sorted"], idx_sorted_u16)
            torch.cuda.synchronize(device)

            def run_kernel(fn, idx_tensor, top_q_tensor=None):
                if top_q_tensor is None:
                    top_q_tensor = data["top_q"]
                fn(top_q_tensor, idx_tensor, data["B_row_pack"], data["top_scale"], data["w_scale"], data["out"])

            # Reference for correctness.
            kernels["i32_c8_b128"][0](data["top_q"], data["idx"], data["B_row_pack"], data["top_scale"], data["w_scale"], data["out"])
            torch.cuda.synchronize(device)
            ref = data["out"].clone()

            rows = []
            for name, (fn, idx_key) in kernels.items():
                idx_tensor = idx_u16 if idx_key == "idx_u16" else data["idx"]
                ms = time_cuda(lambda fn=fn, idx_tensor=idx_tensor: run_kernel(fn, idx_tensor), args.warmup, args.iters, device)
                graph_ms = time_graph(lambda fn=fn, idx_tensor=idx_tensor: run_kernel(fn, idx_tensor), args.warmup, args.iters, device)
                fn(data["top_q"], idx_tensor, data["B_row_pack"], data["top_scale"], data["w_scale"], data["out"])
                torch.cuda.synchronize(device)
                rows.append({
                    "kernel": name,
                    "idx_layout": idx_key,
                    "idx_order": "random",
                    "event_ms": ms,
                    "graph_ms": graph_ms,
                    "graph_speedup_percent": ((ms - graph_ms) / ms * 100.0) if graph_ms is not None and ms > 0 else None,
                    "correctness_vs_i32_c8_b128": compare(ref, data["out"]),
                })

            # Sorting idx is a layout proxy: it improves B_row spatial locality without changing storage format.
            sorted_tests = {
                "i32_c8_b128_sorted_idx": (ext.sparse_i32_c8_b128, data["idx_sorted"]),
                "u16_c8_b128_sorted_idx": (ext.sparse_u16_c8_b128, idx_sorted_u16),
            }
            for name, (fn, idx_tensor) in sorted_tests.items():
                ms = time_cuda(lambda fn=fn, idx_tensor=idx_tensor: run_kernel(fn, idx_tensor, data["top_q_sorted"]), args.warmup, args.iters, device)
                graph_ms = time_graph(lambda fn=fn, idx_tensor=idx_tensor: run_kernel(fn, idx_tensor, data["top_q_sorted"]), args.warmup, args.iters, device)
                fn(data["top_q_sorted"], idx_tensor, data["B_row_pack"], data["top_scale"], data["w_scale"], data["out"])
                torch.cuda.synchronize(device)
                rows.append({
                    "kernel": name,
                    "idx_layout": "u16" if name.startswith("u16") else "i32",
                    "idx_order": "sorted_per_row",
                    "event_ms": ms,
                    "graph_ms": graph_ms,
                    "graph_speedup_percent": ((ms - graph_ms) / ms * 100.0) if graph_ms is not None and ms > 0 else None,
                    "correctness_vs_i32_c8_b128": compare(ref, data["out"]),
                })

            best = min(rows, key=lambda r: r["event_ms"])
            baseline = next(r for r in rows if r["kernel"] == "i32_c8_b128" and r["idx_order"] == "random")
            report["results"].append({
                "shape": {"M": M, "N": N, "K": K, "R": R},
                "idx_i32_to_u16_compress_ms": compress_ms,
                "baseline_i32_c8_b128_event_ms": baseline["event_ms"],
                "best_event": best,
                "best_speedup_over_baseline_percent": (baseline["event_ms"] - best["event_ms"]) / baseline["event_ms"] * 100.0,
                "all_kernel_rows": rows,
            })

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = out_path.with_suffix(".md")
    lines = ["# Split Kernel Optimization 5 Points Smoke v1", "", f"Device: {report['device']}", "", "## Summary", ""]
    for rec in report["results"]:
        s = rec["shape"]
        b = rec["baseline_i32_c8_b128_event_ms"]
        best = rec["best_event"]
        lines.append(
            f"- M{s['M']} N{s['N']} K{s['K']} R{s['R']}: baseline {b:.6f} ms, "
            f"best {best['kernel']} {best['event_ms']:.6f} ms, speedup {rec['best_speedup_over_baseline_percent']:.2f}%, "
            f"idx16 compress {rec['idx_i32_to_u16_compress_ms']:.6f} ms"
        )
    lines += [
        "",
        "## Applied Points",
        "",
        "1. Permutation-free split layout is kept: body pack is dense and top-R is side-buffered as top_q/idx.",
        "2. True QuaRot GEMM epilogue fusion is recorded as the remaining hard step; this smoke avoids adopting a second sparse-add path as final design.",
        "3. Sparse-friendly refinements tested here: uint16 idx and per-row sorted idx/top_q pairs for better B-row locality.",
        "4. Autotune space tested here: cols/thread in {4,8,16}, block threads in {128,256}, idx width in {i32,u16}, R and shape grid.",
        "5. CUDA Graph replay timing is reported per kernel where capture succeeds.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(out_path), "md": str(md), "num_results": len(report["results"])}, indent=2))


if __name__ == "__main__":
    main()
