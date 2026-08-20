"""Run Llama2-7B mixed-policy v61 latency comparison.

This runner is intentionally standalone. It reuses the existing v61 baseline
and RoMeO-prepare bench wrappers, but does not modify them.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
RESULTS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/results"

PYTHON = Path(os.environ.get("KQ_PYTHON", "/data/yzy/miniconda3/envs/romeo_sm120/bin/python"))
BASELINE_SCRIPT = TOOLS / "bench_prefill_bf16_romeoquarotdense_split_total_v61_sharedprepare_fp16rotate.py"
NEW_SCRIPT = TOOLS / "bench_prefill_bf16_romeoquarotdense_split_total_v61_romeorot_prepare.py"

MODEL = "meta-llama/Llama-2-7b-hf"
MODEL_SHORT = "llama2_7b"
ROTATION_CONFIG = "/data/yzy/quarot/llama2-7b-hf.csv"
POLICY = (
    RESULTS
    / "oldv6_grouped_capped_sharedlambda_pareto_romeorot_v1/llama2_7b/"
    / "llama2_7b_romeorot_sharedlambda_lambda_0p08/policy.json"
)
OUT_ROOT = RESULTS / "oldv6_grouped_capped_sharedlambda_latency_romeorot_v61_compare_v1" / MODEL_SHORT / "mixed_0p08"


@dataclass
class Job:
    kind: str
    variants: str
    gpu: int | None = None
    proc: subprocess.Popen | None = None
    log_file: object | None = None

    @property
    def script(self) -> Path:
        return BASELINE_SCRIPT if self.kind == "v61_baseline" else NEW_SCRIPT

    @property
    def label(self) -> str:
        return f"{MODEL_SHORT}_{self.kind}_mixed_0p08"

    @property
    def out_dir(self) -> Path:
        return OUT_ROOT / self.kind

    @property
    def csv_path(self) -> Path:
        return self.out_dir / f"{self.label}_prefill_layer_total_v53.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "new", "both"], default="both")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batches", default="16,64,256")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def env_for_gpu(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PATH"] = "/data/yzy/miniconda3/envs/romeo_sm120/bin:/usr/local/cuda/bin:/usr/bin:/bin"
    env["PYTHONPATH"] = ":".join(
        [
            "/data/yzy/RoMeo",
            str(ROOT),
            str(TOOLS),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(":")
    env["KQ_PROJECT_ROOT"] = str(ROOT)
    return env


def read_policy_summary(path: Path) -> dict:
    policy = json.loads(path.read_text())
    summary = policy.get("summary", {})
    return {
        "ratio": summary.get("mac_weighted_projected_ratio", ""),
        "split": summary.get("split_module_count", ""),
        "sumR": summary.get("sum_R", ""),
    }


def launch(job: Job, gpu: int, args: argparse.Namespace) -> None:
    job.gpu = gpu
    job.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.out_dir / f"{job.label}.log"
    job.log_file = log_path.open("w", buffering=1)
    cmd = [
        str(PYTHON),
        str(job.script),
        "--model",
        MODEL,
        "--label",
        job.label,
        "--policy",
        str(POLICY),
        "--rotation_config",
        ROTATION_CONFIG,
        "--seq_len",
        str(args.seq_len),
        "--batches",
        args.batches,
        "--layers",
        args.layers,
        "--variants",
        job.variants,
        "--device",
        "cuda:0",
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--out_dir",
        str(job.out_dir),
        "--local_files_only",
    ]
    print(f"[LAUNCH] gpu={gpu} kind={job.kind} variants={job.variants}", flush=True)
    job.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env_for_gpu(gpu), stdout=job.log_file, stderr=subprocess.STDOUT)


def sum_csv(path: Path) -> dict[str, float]:
    sums = {"bf16": 0.0, "romeo": 0.0, "split": 0.0}
    with path.open("r", newline="") as f:
        for rec in csv.DictReader(f):
            sums["bf16"] += float(rec.get("bf16_ms") or 0.0)
            sums["romeo"] += float(rec.get("romeo_ms") or 0.0)
            sums["split"] += float(rec.get("split_ms") or 0.0)
    return sums


def summarize(jobs: list[Job]) -> Path:
    row = {
        "model": MODEL_SHORT,
        "policy": str(POLICY),
        **read_policy_summary(POLICY),
    }
    for job in jobs:
        if not job.csv_path.exists():
            row[f"{job.kind}_status"] = "missing_csv"
            continue
        sums = sum_csv(job.csv_path)
        row[f"{job.kind}_status"] = "ok"
        row[f"{job.kind}_bf16_ms"] = f"{sums['bf16']:.6f}"
        row[f"{job.kind}_romeo_ms"] = f"{sums['romeo']:.6f}"
        row[f"{job.kind}_split_ms"] = f"{sums['split']:.6f}"
    base_split = float(row.get("v61_baseline_split_ms") or 0.0)
    new_split = float(row.get("v61_romeorot_prepare_split_ms") or 0.0)
    if base_split > 0 and new_split > 0:
        row["delta_new_minus_baseline_ms"] = f"{new_split - base_split:.6f}"
        row["speedup_baseline_over_new"] = f"{base_split / new_split:.6f}"
    else:
        row["delta_new_minus_baseline_ms"] = ""
        row["speedup_baseline_over_new"] = ""

    fields = [
        "model",
        "ratio",
        "split",
        "sumR",
        "policy",
        "v61_baseline_status",
        "v61_baseline_bf16_ms",
        "v61_baseline_romeo_ms",
        "v61_baseline_split_ms",
        "v61_romeorot_prepare_status",
        "v61_romeorot_prepare_split_ms",
        "delta_new_minus_baseline_ms",
        "speedup_baseline_over_new",
    ]
    out = OUT_ROOT / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
    return out


def main() -> None:
    args = parse_args()
    jobs: list[Job] = []
    if args.mode in ("baseline", "both"):
        jobs.append(Job("v61_baseline", "bf16,romeo,split"))
    if args.mode in ("new", "both"):
        jobs.append(Job("v61_romeorot_prepare", "split"))
    if args.skip_existing:
        jobs = [job for job in jobs if not job.csv_path.exists()]

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if len(gpus) < len(jobs):
        raise RuntimeError(f"Need at least {len(jobs)} GPUs for requested jobs, got {gpus}")

    running: list[Job] = []
    completed: list[Job] = []
    failed: list[Job] = []
    for job, gpu in zip(jobs, gpus):
        launch(job, gpu, args)
        running.append(job)

    while running:
        time.sleep(2)
        for job in list(running):
            assert job.proc is not None
            rc = job.proc.poll()
            if rc is None:
                continue
            if job.log_file is not None:
                job.log_file.close()
            running.remove(job)
            if rc == 0:
                completed.append(job)
                print(f"[DONE] kind={job.kind} gpu={job.gpu}", flush=True)
            else:
                failed.append(job)
                print(f"[FAILED] kind={job.kind} gpu={job.gpu} rc={rc}", flush=True)

    summary = summarize(completed + failed)
    print(f"[SUMMARY] {summary}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
