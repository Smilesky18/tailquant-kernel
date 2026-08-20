import argparse
import copy
import csv
import json
import os
import time
import types
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8


def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def run_layer_once(layer, hidden_states, position_ids, position_embeddings):
    out = layer(
        hidden_states,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
    )
    return out[0] if isinstance(out, tuple) else out


def is_real_policy_linear(m):
    name = m.__class__.__name__
    return name == "RealPolicyLinear" or "RealPolicyLinear" in name


def event_time_ms(fn, device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    ret = fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)), ret


def install_pure_stage_profiler(layer, stage_records):
    patched = []

    for name, mod in layer.named_modules():
        if not is_real_policy_linear(mod):
            continue

        if not hasattr(mod, "ext"):
            continue
        if not hasattr(mod, "B_col"):
            continue

        def make_profiled_pure(module_name):
            def profiled_pure(self, A, scratch):
                M = int(A.shape[0])
                device = A.device

                output = torch.empty(
                    (M, self.N),
                    dtype=torch.float16,
                    device=device,
                )

                def pack_fn():
                    self.ext.pack_a_full_s4(
                        A,
                        scratch["A_pack"],
                        scratch["a_scale"],
                        self.eps,
                    )

                pack_ms, _ = event_time_ms(pack_fn, device)

                def gemm_fn():
                    self.ext.cutlass_s4_gemm(
                        scratch["A_pack"],
                        self.B_col,
                        scratch["C_i32"],
                        M,
                        self.N,
                        self.K,
                    )

                gemm_ms, _ = event_time_ms(gemm_fn, device)

                def scale_fn():
                    self.ext.scale_i32_to_fp16(
                        scratch["C_i32"],
                        scratch["a_scale"],
                        self.w_scale,
                        output,
                    )

                scale_ms, _ = event_time_ms(scale_fn, device)

                total_ms = pack_ms + gemm_ms + scale_ms

                stage_records.append({
                    "module": module_name,
                    "M": M,
                    "K": int(self.K),
                    "N": int(self.N),
                    "pack_a_full_s4_ms": pack_ms,
                    "cutlass_s4_gemm_ms": gemm_ms,
                    "scale_i32_to_fp16_ms": scale_ms,
                    "sum_stages_ms": total_ms,
                    "pack_pct": pack_ms / total_ms if total_ms > 0 else 0.0,
                    "gemm_pct": gemm_ms / total_ms if total_ms > 0 else 0.0,
                    "scale_pct": scale_ms / total_ms if total_ms > 0 else 0.0,
                })
                return output

            return profiled_pure

        mod._pure = types.MethodType(make_profiled_pure(name), mod)
        patched.append(name)

    log("[PATCHED_PURE_STAGE_PROFILER] " + json.dumps(patched, indent=2))
    return patched


def summarize(records):
    agg = {}
    for r in records:
        key = r["module"]
        a = agg.setdefault(key, {
            "module": key,
            "count": 0,
            "M": r["M"],
            "K": r["K"],
            "N": r["N"],
            "pack_a_full_s4_ms": 0.0,
            "cutlass_s4_gemm_ms": 0.0,
            "scale_i32_to_fp16_ms": 0.0,
            "sum_stages_ms": 0.0,
        })
        a["count"] += 1
        for k in ["pack_a_full_s4_ms", "cutlass_s4_gemm_ms", "scale_i32_to_fp16_ms", "sum_stages_ms"]:
            a[k] += float(r[k])

    rows = []
    for a in agg.values():
        c = max(a["count"], 1)
        for k in ["pack_a_full_s4_ms", "cutlass_s4_gemm_ms", "scale_i32_to_fp16_ms", "sum_stages_ms"]:
            a[k] /= c
        total = a["sum_stages_ms"]
        a["pack_pct"] = a["pack_a_full_s4_ms"] / total if total > 0 else 0.0
        a["gemm_pct"] = a["cutlass_s4_gemm_ms"] / total if total > 0 else 0.0
        a["scale_pct"] = a["scale_i32_to_fp16_ms"] / total if total > 0 else 0.0
        rows.append(a)

    rows.sort(key=lambda x: -x["sum_stages_ms"])
    return rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[BATCHES] {args.batches}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log("[NOTE] Pure W4A4 stage profiling: pack_a_full_s4 / cutlass_s4_gemm / scale_i32_to_fp16")

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V8.resolve_extensions(B, args, out_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    layers = V8.get_layers(model)
    hidden_size = V8.infer_hidden_size(model)
    base_layer = layers[args.layer_idx]

    pure_layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    pure_layer, records_patch = V8.patch_layer_with_real_policy(
        layer=pure_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="pure",
        ratio=0.0,
        eps=args.eps,
        device=device,
    )
    pure_layer.to(device=device).eval()

    stage_records = []
    install_pure_stage_profiler(pure_layer, stage_records)

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    all_summary = []

    for batch in batches:
        log(f"\n[CASE] batch={batch} seq_len={args.seq_len}")
        hidden = torch.randn(
            batch,
            args.seq_len,
            hidden_size,
            device=device,
            dtype=torch.float16,
        )
        position_ids = V8.make_position_ids(batch, args.seq_len, device)
        pe = V8.build_position_embeddings(model, hidden, position_ids, torch.float16)

        for _ in range(args.warmup):
            _ = run_layer_once(pure_layer, hidden, position_ids, pe)
        torch.cuda.synchronize(device)

        stage_records.clear()

        for _ in range(args.runs):
            _ = run_layer_once(pure_layer, hidden, position_ids, pe)
        torch.cuda.synchronize(device)

        batch_records = [dict(r, batch=batch, seq_len=args.seq_len) for r in stage_records]
        summary = summarize(batch_records)
        for r in summary:
            r["batch"] = batch
            r["seq_len"] = args.seq_len
        all_summary.extend(summary)

        raw_json = out_dir / f"pure_stage_records_b{batch}.json"
        sum_json = out_dir / f"pure_stage_summary_b{batch}.json"
        sum_csv = out_dir / f"pure_stage_summary_b{batch}.csv"

        json.dump(batch_records, open(raw_json, "w"), indent=2, ensure_ascii=False)
        json.dump(summary, open(sum_json, "w"), indent=2, ensure_ascii=False)

        with open(sum_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

        log(f"[SUMMARY_CSV] {sum_csv}")
        log(f"[SUMMARY_JSON] {sum_json}")
        log(f"[SUMMARY_TOP_b{batch}]")
        for r in summary:
            log(json.dumps(r, ensure_ascii=False))

    all_csv = out_dir / "pure_stage_summary_all_v11.csv"
    all_json = out_dir / "pure_stage_summary_all_v11.json"

    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary)

    json.dump(all_summary, open(all_json, "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {all_csv}")
    log(f"[JSON] {all_json}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
