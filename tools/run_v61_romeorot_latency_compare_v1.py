"""Run v61 baseline vs RoMeO-rotation prepare latency comparison.

This runner only invokes standalone wrapper scripts and writes results/logs
under the requested output root. It does not modify original benchmark files.
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

LAMBDAS = ["0p08", "0p16", "0p32", "0p64", "1p28", "2p56"]


@dataclass(frozen=True)
class ModelSpec:
    short: str
    model: str
    rotation_config: str
    policy_root: Path
    label_prefix: str


MODELS = [
    ModelSpec(
        short="qwen3_8b",
        model="Qwen/Qwen3-8B",
        rotation_config="/data/yzy/quarot/qwen3-8B_layer_all.csv",
        policy_root=RESULTS / "oldv6_grouped_capped_sharedlambda_pareto_romeorot_v1/qwen3_8b",
        label_prefix="qwen3_romeorot_sharedlambda_lambda_",
    ),
    ModelSpec(
        short="llama3_8b",
        model="meta-llama/Meta-Llama-3-8B",
        rotation_config="/data/yzy/quarot/llama3-8B_layer.csv",
        policy_root=RESULTS / "oldv6_grouped_capped_sharedlambda_pareto_romeorot_v1/llama3_8b",
        label_prefix="llama3_romeorot_sharedlambda_lambda_",
    ),
]


@dataclass
class Job:
    kind: str
    model: ModelSpec
    lam: str
    gpu: int | None = None
    proc: subprocess.Popen | None = None
    log_file: object | None = None

    @property
    def script(self) -> Path:
        return BASELINE_SCRIPT if self.kind == "v61_baseline" else NEW_SCRIPT

    @property
    def label(self) -> str:
        return f"{self.model.short}_{self.kind}_lambda_{self.lam}"

    @property
    def policy(self) -> Path:
        policy_dir = self.model.policy_root / f"{self.model.label_prefix}{self.lam}"
        return policy_dir / "policy.json"

    @property
    def out_dir(self) -> Path:
        return RESULTS / "oldv6_grouped_capped_sharedlambda_latency_romeorot_v61_compare_v1" / self.model.short / self.lam / self.kind

    @property
    def csv_path(self) -> Path:
        return self.out_dir / f"{self.label}_prefill_layer_total_v53.csv"

    @property
    def meta_path(self) -> Path:
        return self.out_dir / f"{self.label}_prefill_layer_total_meta_v53.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "new", "both"], default="both")
    parser.add_argument("--max_parallel", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
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
    return env


def read_policy_summary(path: Path) -> dict:
    with path.open("r") as f:
        policy = json.load(f)
    summary = policy.get("summary", {})
    return {
        "ratio": summary.get("mac_weighted_projected_ratio", summary.get("ratio", "")),
        "split": summary.get("split_module_count", ""),
        "sumR": summary.get("sum_R", summary.get("sumR", "")),
    }


def launch(job: Job, gpu: int, args: argparse.Namespace):
    job.gpu = gpu
    job.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.out_dir / f"{job.label}.log"
    job.log_file = log_path.open("w", buffering=1)
    cmd = [
        str(PYTHON),
        str(job.script),
        "--model",
        job.model.model,
        "--label",
        job.label,
        "--policy",
        str(job.policy),
        "--rotation_config",
        job.model.rotation_config,
        "--seq_len",
        str(args.seq_len),
        "--batches",
        args.batches,
        "--layers",
        args.layers,
        "--variants",
        "split",
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
    print(f"[LAUNCH] gpu={gpu} kind={job.kind} model={job.model.short} lambda={job.lam}", flush=True)
    job.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env_for_gpu(gpu), stdout=job.log_file, stderr=subprocess.STDOUT)


def summarize(jobs: list[Job]) -> Path:
    summary_root = RESULTS / "oldv6_grouped_capped_sharedlambda_latency_romeorot_v61_compare_v1"
    rows_by_key: dict[tuple[str, str], dict] = {}
    for job in jobs:
        key = (job.model.short, job.lam)
        row = rows_by_key.setdefault(
            key,
            {
                "model": job.model.short,
                "lambda": job.lam,
                **read_policy_summary(job.policy),
            },
        )
        if not job.csv_path.exists():
            row[f"{job.kind}_status"] = "missing_csv"
            continue
        sums: dict[str, float] = {}
        with job.csv_path.open("r", newline="") as f:
            for rec in csv.DictReader(f):
                batch = str(rec.get("batch", ""))
                sums[batch] = sums.get(batch, 0.0) + float(rec.get("split_ms", 0.0))
        for batch in ["16", "64", "256"]:
            row[f"{job.kind}_b{batch}_ms"] = f"{sums.get(batch, 0.0):.6f}"
        row[f"{job.kind}_status"] = "ok"

    for row in rows_by_key.values():
        for batch in ["16", "64", "256"]:
            old = float(row.get(f"v61_baseline_b{batch}_ms", "0") or 0)
            new = float(row.get(f"v61_romeorot_prepare_b{batch}_ms", "0") or 0)
            if old > 0 and new > 0:
                row[f"delta_b{batch}_ms"] = f"{new - old:.6f}"
                row[f"speedup_b{batch}"] = f"{old / new:.6f}"
            else:
                row[f"delta_b{batch}_ms"] = ""
                row[f"speedup_b{batch}"] = ""

    fields = [
        "model",
        "lambda",
        "ratio",
        "split",
        "sumR",
        "v61_baseline_status",
        "v61_baseline_b16_ms",
        "v61_baseline_b64_ms",
        "v61_baseline_b256_ms",
        "v61_romeorot_prepare_status",
        "v61_romeorot_prepare_b16_ms",
        "v61_romeorot_prepare_b64_ms",
        "v61_romeorot_prepare_b256_ms",
        "delta_b16_ms",
        "speedup_b16",
        "delta_b64_ms",
        "speedup_b64",
        "delta_b256_ms",
        "speedup_b256",
    ]
    out = summary_root / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(rows_by_key):
            writer.writerow({field: rows_by_key[key].get(field, "") for field in fields})
    return out


def main():
    args = parse_args()
    kinds = []
    if args.mode in ("baseline", "both"):
        kinds.append("v61_baseline")
    if args.mode in ("new", "both"):
        kinds.append("v61_romeorot_prepare")
    jobs = [Job(kind=kind, model=model, lam=lam) for kind in kinds for model in MODELS for lam in LAMBDAS]
    if args.skip_existing:
        jobs = [job for job in jobs if not job.csv_path.exists()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    free = list(gpus)
    running: list[Job] = []
    pending = list(jobs)
    completed: list[Job] = []
    failed: list[Job] = []

    while pending or running:
        while pending and free and len(running) < args.max_parallel:
            gpu = free.pop(0)
            job = pending.pop(0)
            launch(job, gpu, args)
            running.append(job)
        time.sleep(2)
        still = []
        for job in running:
            assert job.proc is not None
            rc = job.proc.poll()
            if rc is None:
                still.append(job)
                continue
            if job.log_file is not None:
                job.log_file.close()
            free.append(int(job.gpu))
            if rc == 0:
                print(f"[DONE] gpu={job.gpu} kind={job.kind} model={job.model.short} lambda={job.lam}", flush=True)
                completed.append(job)
            else:
                print(f"[FAIL] rc={rc} gpu={job.gpu} kind={job.kind} model={job.model.short} lambda={job.lam}", flush=True)
                failed.append(job)
        running = still

    out = summarize([*completed, *failed])
    print(f"[SUMMARY] {out}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
