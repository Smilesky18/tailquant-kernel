#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run and merge per-layer latency for the oldv6 mixed RoMeO route.

The merged CSV contains full-layer BF16, full-layer RoMeO, and v61
RoMeO-prepare split timings for each layer and batch. Original benchmark
scripts are invoked unchanged.
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
ROMEO_PREPARE_SCRIPT = TOOLS / "bench_prefill_bf16_romeoquarotdense_split_total_v61_romeorot_prepare.py"

DEFAULT_BATCHES = "2,4,8,16,64,256"


@dataclass(frozen=True)
class ModelSpec:
    short: str
    model: str
    rotation_config: str
    label_prefix: str
    default_lambda: str


MODELS = {
    "qwen3_8b": ModelSpec("qwen3_8b", "Qwen/Qwen3-8B", "/data/yzy/quarot/qwen3-8B_layer_all.csv", "qwen3_romeorot_sharedlambda", "2p56"),
    "llama3_8b": ModelSpec("llama3_8b", "meta-llama/Meta-Llama-3-8B", "/data/yzy/quarot/llama3-8B_layer.csv", "llama3_romeorot_sharedlambda", "1p28"),
    "llama2_7b": ModelSpec("llama2_7b", "meta-llama/Llama-2-7b-hf", "/data/yzy/quarot/llama2-7b-hf.csv", "llama2_7b_romeorot_sharedlambda", "0p08"),
    "llama2_13b": ModelSpec("llama2_13b", "meta-llama/Llama-2-13b-hf", "/data/yzy/quarot/llama-2-13b-hf.csv", "llama2_13b_romeorot_sharedlambda", "0p08"),
    "llama31_8b": ModelSpec("llama31_8b", "meta-llama/Llama-3.1-8B", "/data/yzy/quarot/llama3.1-8B.csv", "llama31_8b_romeorot_sharedlambda", "0p08"),
}


@dataclass
class Job:
    spec: ModelSpec
    lam: str
    kind: str
    variants: str
    gpu: int = -1
    proc: subprocess.Popen | None = None
    log_file: object | None = None

    @property
    def script(self) -> Path:
        return BASELINE_SCRIPT if self.kind == "baseline_full" else ROMEO_PREPARE_SCRIPT

    @property
    def label(self) -> str:
        return f"{self.spec.short}_{self.kind}_mixed_{self.lam}_b2_4_8_16_64_256"

    @property
    def out_dir(self) -> Path:
        return out_root() / self.spec.short / self.lam / self.kind

    @property
    def csv_path(self) -> Path:
        return self.out_dir / f"{self.label}_prefill_layer_total_v53.csv"


def out_root() -> Path:
    return RESULTS / "oldv6_mixed_romeorot_latency_perlayer_v1"


def policy_label_candidates(spec: ModelSpec, lam: str) -> list[str]:
    return [
        f"{spec.label_prefix}_lambda_{lam}",
        f"{spec.label_prefix}_lambda__lambda_{lam}",
        f"{spec.label_prefix}{lam}",
    ]


def policy_path(spec: ModelSpec, lam: str) -> Path:
    root = RESULTS / "oldv6_grouped_capped_sharedlambda_pareto_romeorot_v1" / spec.short
    for label in policy_label_candidates(spec, lam):
        path = root / label / "policy.json"
        if path.exists():
            return path
    return root / policy_label_candidates(spec, lam)[0] / "policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen3_8b,llama3_8b,llama2_7b,llama2_13b,llama31_8b")
    parser.add_argument("--lambdas", default="")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max_parallel", type=int, default=8)
    parser.add_argument("--batches", default=DEFAULT_BATCHES)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
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


def launch(job: Job, gpu: int, args: argparse.Namespace) -> None:
    job.gpu = gpu
    job.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.out_dir / f"{job.label}.log"
    job.log_file = log_path.open("w", buffering=1)
    cmd = [
        str(PYTHON),
        str(job.script),
        "--model",
        job.spec.model,
        "--label",
        job.label,
        "--policy",
        str(policy_path(job.spec, job.lam)),
        "--rotation_config",
        job.spec.rotation_config,
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
    print(f"[LAUNCH] gpu={gpu} model={job.spec.short} lambda={job.lam} kind={job.kind}", flush=True)
    job.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env_for_gpu(gpu), stdout=job.log_file, stderr=subprocess.STDOUT)


def read_policy_summary(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    return {
        "ratio": summary.get("mac_weighted_projected_ratio", summary.get("ratio", "")),
        "split_module_count": summary.get("split_module_count", ""),
        "sum_R": summary.get("sum_R", summary.get("sumR", "")),
    }


def read_csv_by_key(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    rows: dict[tuple[int, int], dict[str, str]] = {}
    with path.open("r", newline="") as f:
        for rec in csv.DictReader(f):
            key = (int(rec["layer_idx"]), int(rec["batch"]))
            rows[key] = rec
    return rows


def merge_model(spec: ModelSpec, lam: str) -> Path:
    base_job = Job(spec, lam, "baseline_full", "bf16,romeo")
    split_job = Job(spec, lam, "v61_romeorot_prepare", "split")
    base_rows = read_csv_by_key(base_job.csv_path)
    split_rows = read_csv_by_key(split_job.csv_path)
    policy = policy_path(spec, lam)
    policy_summary = read_policy_summary(policy)
    merged = out_root() / spec.short / lam / f"{spec.short}_mixed_{lam}_perlayer_b2_4_8_16_64_256.csv"
    merged.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_short",
        "lambda",
        "layer_idx",
        "batch",
        "seq_len",
        "bf16_ms",
        "romeo_ms",
        "v61_romeorot_prepare_split_ms",
        "split_over_bf16",
        "split_over_romeo",
        "ratio",
        "split_module_count",
        "sum_R",
        "policy_file",
        "baseline_csv",
        "v61_romeorot_prepare_csv",
    ]
    with merged.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(set(base_rows) | set(split_rows)):
            b = base_rows.get(key, {})
            s = split_rows.get(key, {})
            bf16 = float(b.get("bf16_ms") or 0.0)
            romeo = float(b.get("romeo_ms") or 0.0)
            split = float(s.get("split_ms") or 0.0)
            row = {
                "model_short": spec.short,
                "lambda": lam,
                "layer_idx": key[0],
                "batch": key[1],
                "seq_len": b.get("seq_len") or s.get("seq_len") or "",
                "bf16_ms": f"{bf16:.6f}" if bf16 else "",
                "romeo_ms": f"{romeo:.6f}" if romeo else "",
                "v61_romeorot_prepare_split_ms": f"{split:.6f}" if split else "",
                "split_over_bf16": f"{split / bf16:.6f}" if split and bf16 else "",
                "split_over_romeo": f"{split / romeo:.6f}" if split and romeo else "",
                **policy_summary,
                "policy_file": str(policy),
                "baseline_csv": str(base_job.csv_path),
                "v61_romeorot_prepare_csv": str(split_job.csv_path),
            }
            writer.writerow(row)
    return merged


def write_total_summary(merged_paths: list[Path]) -> Path:
    rows = []
    for path in merged_paths:
        sums: dict[tuple[str, str], dict[str, float]] = {}
        with path.open("r", newline="") as f:
            for rec in csv.DictReader(f):
                key = (rec["model_short"], rec["batch"])
                item = sums.setdefault(key, {"bf16": 0.0, "romeo": 0.0, "split": 0.0})
                item["bf16"] += float(rec.get("bf16_ms") or 0.0)
                item["romeo"] += float(rec.get("romeo_ms") or 0.0)
                item["split"] += float(rec.get("v61_romeorot_prepare_split_ms") or 0.0)
        for (model, batch), item in sorted(sums.items()):
            rows.append(
                {
                    "model": model,
                    "batch": batch,
                    "bf16_total_ms": f"{item['bf16']:.6f}",
                    "romeo_total_ms": f"{item['romeo']:.6f}",
                    "v61_romeorot_prepare_split_total_ms": f"{item['split']:.6f}",
                    "split_over_bf16": f"{item['split'] / item['bf16']:.6f}" if item["bf16"] else "",
                    "split_over_romeo": f"{item['split'] / item['romeo']:.6f}" if item["romeo"] else "",
                }
            )
    out = out_root() / "summary_total_by_model_batch.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "batch", "bf16_total_ms", "romeo_total_ms", "v61_romeorot_prepare_split_total_ms", "split_over_bf16", "split_over_romeo"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out


def main() -> None:
    args = parse_args()
    model_keys = [x.strip() for x in args.models.split(",") if x.strip()]
    lambdas = {}
    if args.lambdas:
        for item in args.lambdas.split(","):
            model, lam = item.split(":", 1)
            lambdas[model] = lam
    specs = [MODELS[k] for k in model_keys]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    jobs: list[Job] = []
    for spec in specs:
        lam = lambdas.get(spec.short, spec.default_lambda)
        if not policy_path(spec, lam).exists():
            raise FileNotFoundError(policy_path(spec, lam))
        jobs.extend(
            [
                Job(spec, lam, "baseline_full", "bf16,romeo"),
                Job(spec, lam, "v61_romeorot_prepare", "split"),
            ]
        )
    if args.skip_existing:
        jobs = [job for job in jobs if not job.csv_path.exists()]

    pending = list(jobs)
    running: list[Job] = []
    completed: list[Job] = []
    failed: list[Job] = []
    max_parallel = max(1, min(args.max_parallel, len(gpus)))
    while pending or running:
        while pending and len(running) < max_parallel:
            gpu = gpus[len(running) % len(gpus)]
            job = pending.pop(0)
            launch(job, gpu, args)
            running.append(job)
        time.sleep(5)
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
                print(f"[DONE] model={job.spec.short} lambda={job.lam} kind={job.kind}", flush=True)
            else:
                failed.append(job)
                print(f"[FAILED] model={job.spec.short} lambda={job.lam} kind={job.kind} rc={rc}", flush=True)

    if failed:
        raise RuntimeError(f"{len(failed)} latency jobs failed")

    merged_paths = []
    for spec in specs:
        lam = lambdas.get(spec.short, spec.default_lambda)
        merged = merge_model(spec, lam)
        merged_paths.append(merged)
        print(f"[MERGED] {merged}", flush=True)
    total = write_total_summary(merged_paths)
    print(f"[TOTAL SUMMARY] {total}", flush=True)


if __name__ == "__main__":
    main()
