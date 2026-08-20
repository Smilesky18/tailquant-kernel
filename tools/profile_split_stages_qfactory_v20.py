import argparse
import copy
import csv
import json
import os
import time
import types
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8

from qfactory.kernels.gemm_w4a4 import gemm_int4_int4_nt


def qfactory_gemm_raw(A_pack, B_pack, C_i32, M, N, K):
    A2 = A_pack.view(M, K // 2).contiguous()
    B2 = B_pack.view(N, K // 2).contiguous()
    C2 = C_i32.view(M, N).contiguous()
    ret = gemm_int4_int4_nt(A2, B2, C2)
    if isinstance(ret, torch.Tensor):
        return ret.view(M, N)
    return C2



def log(msg):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--ratio", type=float, default=0.05)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
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


def event_ms(fn, device):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    start.record()
    ret = fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)), ret


def install_split_stage_profiler(layer, records):
    patched = []

    for name, mod in layer.named_modules():
        if not is_real_policy_linear(mod):
            continue
        if not getattr(mod, "is_split", False):
            continue

        def make_profiled_split_compute(module_name):
            def profiled_split_compute(
                self,
                A,
                scratch,
                B_col,
                dense_ready_event,
                dense_stream,
                sparse_stream,
            ):
                M = int(A.shape[0])
                device = A.device
                ext = getattr(self, "ext", None)
                if ext is None:
                    ext = getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError("cannot find ext/main_ext")

                prepare_ms, prep_ret = event_ms(
                    lambda: self._prepare_split(A, scratch),
                    device,
                )
                indices, top_q, _ = prep_ret

                def dense_body_fn():
                    C = qfactory_gemm_raw(
                        scratch["A_pack"],
                        B_col,
                        scratch["C_body_i32"],
                        M,
                        self.N,
                        self.K,
                    )
                    ext.scale_i32_to_fp16(
                        C,
                        scratch["body_scale"],
                        self.w_scale,
                        scratch["Y_body"],
                    )

                dense_ms, _ = event_ms(dense_body_fn, device)

                def sparse_fn():
                    scratch["Y_sparse"].zero_()
                    ext.sparse_top_add_rowmajor_quad_shared(
                        top_q,
                        indices,
                        self.B_row,
                        scratch["top_scale"],
                        self.w_scale,
                        scratch["Y_sparse"],
                        self.K,
                    )

                sparse_ms, _ = event_ms(sparse_fn, device)

                output = torch.empty(
                    (M, self.N),
                    dtype=torch.float16,
                    device=device,
                )

                merge_ms, _ = event_ms(
                    lambda: torch.add(
                        scratch["Y_body"],
                        scratch["Y_sparse"],
                        out=output,
                    ),
                    device,
                )

                total = prepare_ms + dense_ms + sparse_ms + merge_ms
                records.append({
                    "module": module_name,
                    "M": M,
                    "K": int(self.K),
                    "N": int(self.N),
                    "R": int(self.R),
                    "ratio": float(self.ratio),
                    "prepare_split_ms": prepare_ms,
                    "dense_body_ms": dense_ms,
                    "sparse_correction_ms": sparse_ms,
                    "merge_ms": merge_ms,
                    "sum_serial_ms": total,
                    "prepare_pct": prepare_ms / total if total > 0 else 0.0,
                    "dense_pct": dense_ms / total if total > 0 else 0.0,
                    "sparse_pct": sparse_ms / total if total > 0 else 0.0,
                    "merge_pct": merge_ms / total if total > 0 else 0.0,
                })

                return output

            return profiled_split_compute

        mod._split_compute = types.MethodType(make_profiled_split_compute(name), mod)
        patched.append(name)

    log("[PATCHED_SPLIT_STAGE_PROFILER] " + json.dumps(patched, indent=2))
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
            "R": r["R"],
            "ratio": r["ratio"],
            "prepare_split_ms": 0.0,
            "dense_body_ms": 0.0,
            "sparse_correction_ms": 0.0,
            "merge_ms": 0.0,
            "sum_serial_ms": 0.0,
        })
        a["count"] += 1
        for k in ["prepare_split_ms", "dense_body_ms", "sparse_correction_ms", "merge_ms", "sum_serial_ms"]:
            a[k] += float(r[k])

    rows = []
    for a in agg.values():
        c = max(a["count"], 1)
        for k in ["prepare_split_ms", "dense_body_ms", "sparse_correction_ms", "merge_ms", "sum_serial_ms"]:
            a[k] /= c
        total = a["sum_serial_ms"]
        a["prepare_pct"] = a["prepare_split_ms"] / total if total > 0 else 0.0
        a["dense_pct"] = a["dense_body_ms"] / total if total > 0 else 0.0
        a["sparse_pct"] = a["sparse_correction_ms"] / total if total > 0 else 0.0
        a["merge_pct"] = a["merge_ms"] / total if total > 0 else 0.0
        rows.append(a)

    rows.sort(key=lambda x: -x["sum_serial_ms"])
    return rows


def summarize_total(summary_rows):
    total = {
        "module": "__TOTAL__",
        "count": sum(int(r["count"]) for r in summary_rows),
        "M": "",
        "K": "",
        "N": "",
        "R": "",
        "ratio": "",
        "prepare_split_ms": sum(float(r["prepare_split_ms"]) for r in summary_rows),
        "dense_body_ms": sum(float(r["dense_body_ms"]) for r in summary_rows),
        "sparse_correction_ms": sum(float(r["sparse_correction_ms"]) for r in summary_rows),
        "merge_ms": sum(float(r["merge_ms"]) for r in summary_rows),
        "sum_serial_ms": sum(float(r["sum_serial_ms"]) for r in summary_rows),
    }
    s = total["sum_serial_ms"]
    total["prepare_pct"] = total["prepare_split_ms"] / s if s > 0 else 0.0
    total["dense_pct"] = total["dense_body_ms"] / s if s > 0 else 0.0
    total["sparse_pct"] = total["sparse_correction_ms"] / s if s > 0 else 0.0
    total["merge_pct"] = total["merge_ms"] / s if s > 0 else 0.0
    return total


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[LAYER] {args.layer_idx}")
    log(f"[RATIO] {args.ratio}")
    log(f"[BATCHES] {args.batches}")
    log("[NOTE] Split stage profile with QFactory raw A4W4 dense backend: prepare / dense_body / sparse_correction / merge. Timed serially for attribution.")

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

    split_layer = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    split_layer, patch_records = V8.patch_layer_with_real_policy(
        layer=split_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="dual_policy",
        ratio=args.ratio,
        eps=args.eps,
        device=device,
    )
    split_layer.to(device=device).eval()

    stage_records = []
    install_split_stage_profiler(split_layer, stage_records)

    batches = [int(x) for x in args.batches.split(",") if x.strip()]
    all_rows = []

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
            _ = run_layer_once(split_layer, hidden, position_ids, pe)
        torch.cuda.synchronize(device)

        stage_records.clear()

        for _ in range(args.runs):
            _ = run_layer_once(split_layer, hidden, position_ids, pe)
        torch.cuda.synchronize(device)

        batch_records = [dict(r, batch=batch, seq_len=args.seq_len) for r in stage_records]
        summary = summarize(batch_records)
        total_row = summarize_total(summary)
        total_row["batch"] = batch
        total_row["seq_len"] = args.seq_len
        summary_with_total = summary + [total_row]

        for r in summary:
            r["batch"] = batch
            r["seq_len"] = args.seq_len

        all_rows.extend(summary_with_total)

        raw_json = out_dir / f"split_stage_records_b{batch}.json"
        sum_json = out_dir / f"split_stage_summary_b{batch}.json"
        sum_csv = out_dir / f"split_stage_summary_b{batch}.csv"

        json.dump(batch_records, open(raw_json, "w"), indent=2, ensure_ascii=False)
        json.dump(summary_with_total, open(sum_json, "w"), indent=2, ensure_ascii=False)

        fields = sorted({k for r in summary_with_total for k in r.keys()})
        with open(sum_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary_with_total)

        log(f"[SUMMARY_CSV] {sum_csv}")
        log(f"[SUMMARY_JSON] {sum_json}")
        log(f"[TOTAL_b{batch}] " + json.dumps(total_row, ensure_ascii=False))
        log(f"[TOP_MODULES_b{batch}]")
        for r in summary[:7]:
            log(json.dumps(r, ensure_ascii=False))

    all_csv = out_dir / "split_stage_summary_all_v18.csv"
    all_json = out_dir / "split_stage_summary_all_v18.json"

    fields = sorted({k for r in all_rows for k in r.keys()})
    with open(all_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    json.dump(all_rows, open(all_json, "w"), indent=2, ensure_ascii=False)
    json.dump(patch_records, open(out_dir / "split_patch_records_v18.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {all_csv}")
    log(f"[JSON] {all_json}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
