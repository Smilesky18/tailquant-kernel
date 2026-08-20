#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the oldv6-base mixed RoMeO-rotation policy route.

Route:
  1. oldv6 low-ratio base search
  2. RoMeO-rotation grouped-capped search, capped by the oldv6 base policy
  3. shared-lambda Pareto postprocess
  4. RoMeO-rotation GPTQ full PPL eval, deduplicated by policy content

This is a standalone orchestration script. It does not modify the underlying
research scripts or model code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

BASE_SEARCH = TOOLS / "calibrate_per_linear_v6_lowratio_v1.py"
GROUPED_CAPPED_ROMEO = TOOLS / "calibrate_per_linear_v6_lowratio_grouped_capped_romeorot_v1.py"
SHAREDLAMBDA = TOOLS / "make_grouped_capped_sharedlambda_pareto_policies_v1.py"
PPL_EVAL = TOOLS / "eval_policy_v6_weightmode_fp16_romeorot_smoke_v1.py"

LAMBDA_TAGS = ["0p08", "0p16", "0p32", "0p64", "1p28", "2p56"]
LAMBDA_VALUES = "0.08,0.16,0.32,0.64,1.28,2.56"


@dataclass(frozen=True)
class ModelSpec:
    short: str
    model: str
    rotation_config: str
    label_prefix: str


MODELS = {
    "qwen3_8b": ModelSpec("qwen3_8b", "Qwen/Qwen3-8B", "/data/yzy/quarot/qwen3-8B_layer_all.csv", "qwen3_romeorot_sharedlambda"),
    "llama3_8b": ModelSpec("llama3_8b", "meta-llama/Meta-Llama-3-8B", "/data/yzy/quarot/llama3-8B_layer.csv", "llama3_romeorot_sharedlambda"),
    "llama2_7b": ModelSpec("llama2_7b", "meta-llama/Llama-2-7b-hf", "/data/yzy/quarot/llama2-7b-hf.csv", "llama2_7b_romeorot_sharedlambda"),
    "llama2_13b": ModelSpec("llama2_13b", "meta-llama/Llama-2-13b-hf", "/data/yzy/quarot/llama-2-13b-hf.csv", "llama2_13b_romeorot_sharedlambda"),
    "llama31_8b": ModelSpec("llama31_8b", "meta-llama/Llama-3.1-8B", "/data/yzy/quarot/llama3.1-8B.csv", "llama31_8b_romeorot_sharedlambda"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="llama2_13b,llama31_8b")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--search_only", action="store_true")
    parser.add_argument("--ppl_only", action="store_true")
    parser.add_argument("--max_parallel_ppl", type=int, default=4)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
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
            str(ROOT / "kernel_quant/scripts"),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(":")
    env["KQ_PROJECT_ROOT"] = str(ROOT)
    return env


def base_dir(spec: ModelSpec) -> Path:
    return RESULTS / "oldv6_lowratio_search_v2" / spec.short / "split_lambda_0p08_lowratio"


def capped_dir(spec: ModelSpec) -> Path:
    return RESULTS / "oldv6_grouped_capped_search_romeorot_v1" / spec.short / "split_lambda_0p08_grouped_capped_romeorot"


def pareto_dir(spec: ModelSpec) -> Path:
    return RESULTS / "oldv6_grouped_capped_sharedlambda_pareto_romeorot_v1" / spec.short


def ppl_root(spec: ModelSpec) -> Path:
    return RESULTS / "oldv6_grouped_capped_sharedlambda_ppl_romeorot_v1" / spec.short


def run_checked(cmd: list[str], env: dict[str, str], log_path: Path, cwd: Path = ROOT) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    with log_path.open("w", buffering=1) as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}; log={log_path}")


def run_search_for_model(spec: ModelSpec, gpu: int, args: argparse.Namespace) -> None:
    env = env_for_gpu(gpu)
    base = base_dir(spec)
    capped = capped_dir(spec)
    pareto = pareto_dir(spec)

    if not (args.skip_existing and (base / "policy.json").exists()):
        run_checked(
            [
                str(PYTHON),
                str(BASE_SEARCH),
                "--model",
                spec.model,
                "--dataset",
                "wikitext2",
                "--out_dir",
                str(base),
                "--rotation_config",
                spec.rotation_config,
                "--nsamples",
                str(args.nsamples),
                "--seqlen",
                str(args.seqlen),
                "--seed",
                str(args.seed),
                "--steps",
                "50",
                "--eval_every",
                "5",
                "--capture_rows",
                "1024",
                "--train_rows",
                "128",
                "--val_rows",
                "256",
                "--train_out_channels",
                "256",
                "--val_out_channels",
                "256",
                "--lr",
                "0.05",
                "--grad_clip",
                "5.0",
                "--ratio_lambda",
                "0.08",
                "--init_ratio",
                "0.04",
                "--max_ratio",
                "0.1",
                "--init_activation_percentile",
                "99.75",
                "--init_weight_percentile",
                "99.9",
                "--min_percentile",
                "0.96",
                "--mask_temp_start",
                "0.02",
                "--mask_temp_end",
                "0.004",
                "--quantile_temp_start",
                "0.003",
                "--quantile_temp_end",
                "0.0008",
                "--recon_tolerance_rel",
                "0.02",
                "--recon_tolerance_abs",
                "1e-5",
            ],
            env,
            base / "run_oldv6_base.log",
        )
    else:
        print(f"[SKIP] {spec.short} oldv6 base exists", flush=True)

    if not (args.skip_existing and (capped / "policy.json").exists()):
        env_capped = dict(env)
        env_capped["RATIO_CAP_POLICY"] = str(base / "policy.json")
        run_checked(
            [
                str(PYTHON),
                str(GROUPED_CAPPED_ROMEO),
                "--model",
                spec.model,
                "--dataset",
                "wikitext2",
                "--out_dir",
                str(capped),
                "--rotation_config",
                spec.rotation_config,
                "--nsamples",
                str(args.nsamples),
                "--seqlen",
                str(args.seqlen),
                "--seed",
                str(args.seed),
                "--steps",
                "50",
                "--eval_every",
                "5",
                "--capture_rows",
                "1024",
                "--train_rows",
                "128",
                "--val_rows",
                "256",
                "--train_out_channels",
                "256",
                "--val_out_channels",
                "256",
                "--lr",
                "0.05",
                "--grad_clip",
                "5.0",
                "--ratio_lambda",
                "0.08",
                "--init_ratio",
                "0.04",
                "--max_ratio",
                "0.1",
                "--init_activation_percentile",
                "99.75",
                "--init_weight_percentile",
                "99.9",
                "--min_percentile",
                "0.96",
                "--mask_temp_start",
                "0.02",
                "--mask_temp_end",
                "0.004",
                "--quantile_temp_start",
                "0.003",
                "--quantile_temp_end",
                "0.0008",
                "--recon_tolerance_rel",
                "0.02",
                "--recon_tolerance_abs",
                "1e-5",
            ],
            env_capped,
            capped / "run_grouped_capped_romeorot.log",
        )
    else:
        print(f"[SKIP] {spec.short} grouped-capped RoMeO exists", flush=True)

    if not (args.skip_existing and (pareto / "pareto_summary.csv").exists()):
        run_checked(
            [
                str(PYTHON),
                str(SHAREDLAMBDA),
                "--base_policy",
                str(capped / "policy.json"),
                "--out_root",
                str(pareto),
                "--lambda_values",
                LAMBDA_VALUES,
                "--min_R_zero",
                "17",
                "--label_prefix",
                spec.label_prefix,
            ],
            env,
            pareto / "make_sharedlambda.log",
        )
    else:
        print(f"[SKIP] {spec.short} sharedlambda pareto exists", flush=True)


def policy_label_candidates(spec: ModelSpec, lam: str) -> list[str]:
    return [
        f"{spec.label_prefix}_lambda_{lam}",
        f"{spec.label_prefix}_lambda__lambda_{lam}",
        f"{spec.label_prefix}{lam}",
    ]


def policy_path(spec: ModelSpec, lam: str) -> Path:
    for label in policy_label_candidates(spec, lam):
        path = pareto_dir(spec) / label / "policy.json"
        if path.exists():
            return path
    return pareto_dir(spec) / policy_label_candidates(spec, lam)[0] / "policy.json"


def policy_hash(path: Path) -> str:
    data = json.loads(path.read_text())
    return hashlib.sha256(json.dumps(data.get("modules", data), sort_keys=True).encode("utf-8")).hexdigest()


def read_policy_summary(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    return {
        "ratio": summary.get("mac_weighted_projected_ratio", summary.get("ratio", "")),
        "split": summary.get("split_module_count", ""),
        "sumR": summary.get("sum_R", summary.get("sumR", "")),
    }


@dataclass
class PplJob:
    spec: ModelSpec
    lam: str
    canonical_lam: str
    gpu: int
    proc: subprocess.Popen | None = None
    log_file: object | None = None

    @property
    def out_dir(self) -> Path:
        return ppl_root(self.spec) / f"{self.spec.label_prefix}_lambda_{self.lam}_gptq"

    @property
    def result(self) -> Path:
        return self.out_dir / "result.json"


def launch_ppl(job: PplJob, args: argparse.Namespace) -> None:
    job.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.out_dir / "run_gptq_full.log"
    job.log_file = log_path.open("w", buffering=1)
    cmd = [
        str(PYTHON),
        str(PPL_EVAL),
        "--policy",
        str(policy_path(job.spec, job.lam)),
        "--out_dir",
        str(job.out_dir),
        "--model",
        job.spec.model,
        "--dataset",
        "wikitext103",
        "--cal_dataset",
        "wikitext2",
        "--n_windows",
        "128",
        "--seqlen",
        "2048",
        "--gptq_seqlen",
        "2048",
        "--nsamples",
        "128",
        "--seed",
        str(args.seed),
        "--rotation_config",
        job.spec.rotation_config,
        "--weight_method",
        "gptq",
    ]
    print(f"[PPL LAUNCH] gpu={job.gpu} model={job.spec.short} lambda={job.lam}", flush=True)
    job.proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env_for_gpu(job.gpu), stdout=job.log_file, stderr=subprocess.STDOUT)


def load_result(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def run_ppl_for_model(spec: ModelSpec, gpus: list[int], args: argparse.Namespace) -> Path:
    unique_by_hash: dict[str, str] = {}
    reuse: dict[str, str] = {}
    jobs: list[PplJob] = []
    for lam in LAMBDA_TAGS:
        p = policy_path(spec, lam)
        if not p.exists():
            print(f"[MISS] policy missing: {p}", flush=True)
            continue
        h = policy_hash(p)
        if h in unique_by_hash:
            reuse[lam] = unique_by_hash[h]
            print(f"[DEDUP] {spec.short} lambda={lam} same_as={unique_by_hash[h]}", flush=True)
            continue
        unique_by_hash[h] = lam
        reuse[lam] = lam
        out = ppl_root(spec) / f"{spec.label_prefix}_lambda_{lam}_gptq" / "result.json"
        if args.skip_existing and out.exists():
            print(f"[SKIP] {spec.short} lambda={lam} PPL exists", flush=True)
            continue
        jobs.append(PplJob(spec, lam, lam, -1))

    running: list[PplJob] = []
    pending = list(jobs)
    failed: list[PplJob] = []
    max_parallel = max(1, min(args.max_parallel_ppl, len(gpus)))
    while pending or running:
        while pending and len(running) < max_parallel:
            gpu = gpus[len(running) % len(gpus)]
            job = pending.pop(0)
            job.gpu = gpu
            launch_ppl(job, args)
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
            if rc != 0:
                failed.append(job)
                print(f"[PPL FAILED] {job.spec.short} lambda={job.lam} rc={rc}", flush=True)
            else:
                print(f"[PPL DONE] {job.spec.short} lambda={job.lam}", flush=True)

    summary_path = ppl_root(spec) / "summary_mixed_romeorot_gptq_full.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "lambda", "canonical_lambda", "ratio", "split", "sumR", "ppl", "mean_ce", "status", "policy", "result"]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for lam in LAMBDA_TAGS:
            p = policy_path(spec, lam)
            if not p.exists():
                writer.writerow({"model": spec.short, "lambda": lam, "status": "missing_policy"})
                continue
            canonical = reuse.get(lam, lam)
            rpath = ppl_root(spec) / f"{spec.label_prefix}_lambda_{canonical}_gptq" / "result.json"
            result = load_result(rpath)
            row = {
                "model": spec.short,
                "lambda": lam,
                "canonical_lambda": canonical,
                **read_policy_summary(p),
                "ppl": result.get("ppl", ""),
                "mean_ce": result.get("mean_ce", ""),
                "status": "ok" if result else "missing_result",
                "policy": str(p),
                "result": str(rpath),
            }
            writer.writerow(row)
    if failed:
        raise RuntimeError(f"{len(failed)} PPL jobs failed for {spec.short}; summary={summary_path}")
    return summary_path


def main() -> None:
    args = parse_args()
    specs = [MODELS[x.strip()] for x in args.models.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise RuntimeError("No GPUs specified")

    if not args.ppl_only:
        for i, spec in enumerate(specs):
            run_search_for_model(spec, gpus[i % len(gpus)], args)

    if not args.search_only:
        for i, spec in enumerate(specs):
            summary = run_ppl_for_model(spec, gpus[i % len(gpus):] + gpus[: i % len(gpus)], args)
            print(f"[PPL SUMMARY] {summary}", flush=True)


if __name__ == "__main__":
    main()
