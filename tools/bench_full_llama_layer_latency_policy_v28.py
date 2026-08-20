import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

import bench_full_qwen3_8b_layer_latency_policy_projected_v22b as V22B


TARGET_LINEAR_NAMES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    return p.parse_args()


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("cannot locate transformer layers")


def get_hidden_size(model):
    cfg = model.config
    for name in ["hidden_size", "n_embd", "d_model"]:
        if hasattr(cfg, name):
            return int(getattr(cfg, name))
    raise RuntimeError("cannot locate hidden size")


def make_position_embeddings(model, hidden_states, position_ids):
    base = model.model if hasattr(model, "model") else model
    rotary = getattr(base, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return None


def run_layer(layer, hidden_states, position_ids, position_embeddings):
    # Transformers 版本不同，Llama/Qwen 的 layer forward 签名略有差异。
    kwargs_list = []
    if position_embeddings is not None:
        kwargs_list.append({
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "use_cache": False,
            "output_attentions": False,
        })
    kwargs_list.append({
        "position_ids": position_ids,
        "use_cache": False,
        "output_attentions": False,
    })
    kwargs_list.append({
        "position_ids": position_ids,
    })
    kwargs_list.append({})

    last_err = None
    for kwargs in kwargs_list:
        try:
            out = layer(hidden_states, **kwargs)
            if isinstance(out, tuple):
                return out[0]
            return out
        except TypeError as e:
            last_err = e
            continue
    raise last_err


def bench(fn, warmup, iters, device):
    with torch.inference_mode():
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
        return float(st.elapsed_time(ed) / iters)


def make_all_zero_policy_rows(base_layer, layer_idx):
    rows = []
    for module_name in TARGET_LINEAR_NAMES:
        m = base_layer
        ok = True
        for p in module_name.split("."):
            if not hasattr(m, p):
                ok = False
                break
            m = getattr(m, p)
        if not ok:
            continue
        rows.append({
            "layer_idx": layer_idx,
            "module": module_name,
            "name": f"model.layers.{layer_idx}.{module_name}",
            "ratio": 0.0,
            "ratio_continuous": 0.0,
            "ratio_projected": 0.0,
            "activation_percentile": 1.0,
            "weight_percentile": 1.0,
        })
    return rows


def layer_ratio_summary_from_rows(rows):
    if hasattr(V22B, "layer_ratio_summary"):
        return V22B.layer_ratio_summary(rows)
    ratios = [float(r.get("ratio_projected", r.get("ratio", 0.0))) for r in rows]
    return {
        "policy_avg_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "policy_max_ratio": max(ratios) if ratios else 0.0,
        "policy_nonzero_linears": sum(1 for r in ratios if r > 0),
        "policy_missing_linears": 0,
    }


def make_policy_rows(base_layer, ratio_map, layer_idx):
    if hasattr(V22B, "policy_ratios_for_layer"):
        return V22B.policy_ratios_for_layer(
            base_layer,
            ratio_map,
            layer_idx,
            missing_ratio=0.0,
        )

    rows = []
    for module_name in TARGET_LINEAR_NAMES:
        matched = None
        for k, v in ratio_map.items():
            if f"layers.{layer_idx}.{module_name}" in k:
                matched = float(v)
                break
        if matched is None:
            matched = 0.0
        rows.append({
            "layer_idx": layer_idx,
            "module": module_name,
            "name": f"model.layers.{layer_idx}.{module_name}",
            "ratio": matched,
            "ratio_continuous": matched,
            "ratio_projected": matched,
            "activation_percentile": 1.0,
            "weight_percentile": 1.0,
        })
    return rows


def patch_layer(base_layer, B, main_ext, layout_ext, policy_pack_ext, policy_rows, eps, device):
    return V22B.patch_layer_with_policy_ratios(
        base_layer=base_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        policy_rows=policy_rows,
        eps=eps,
        device=device,
    )


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not os.path.isfile(args.policy):
        raise FileNotFoundError(f"policy not found: {args.policy}")

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[LABEL] {args.label}")
    log(f"[MODEL] {args.model}")
    log(f"[POLICY] {args.policy}")
    log(f"[OUT] {out}")
    log(f"[DEVICE] {torch.cuda.get_device_name(device)}")
    log(f"[BATCHES] {args.batches}")
    log(f"[SEQ_LEN] {args.seq_len}")

    raw_policy = json.load(open(args.policy))
    ratio_map = V22B.flatten_policy_ratios(raw_policy)

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, policy_pack_ext = V22B.V8.resolve_extensions(B, args, out)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval()

    layers = get_layers(model)
    hidden_size = get_hidden_size(model)
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

    rows = []
    summary_rows = []

    for layer_idx, base_layer in enumerate(layers):
        policy_rows = make_policy_rows(base_layer, ratio_map, layer_idx)
        ratio_summary = layer_ratio_summary_from_rows(policy_rows)

        log("")
        log(f"[LAYER] {layer_idx} " + json.dumps(ratio_summary, ensure_ascii=False))

        bf16_layer = copy.deepcopy(base_layer).to(device).eval()

        pure_policy_rows = make_all_zero_policy_rows(base_layer, layer_idx)
        pure_layer, pure_patch_records, _ = patch_layer(
            base_layer=base_layer,
            B=B,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            policy_rows=pure_policy_rows,
            eps=args.eps,
            device=device,
        )

        split_layer, split_patch_records, _ = patch_layer(
            base_layer=base_layer,
            B=B,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            policy_rows=policy_rows,
            eps=args.eps,
            device=device,
        )

        for batch in batches:
            position_ids = torch.arange(args.seq_len, device=device).unsqueeze(0).repeat(batch, 1)
            h_bf16 = torch.randn((batch, args.seq_len, hidden_size), device=device, dtype=torch.bfloat16)
            h_fp16 = h_bf16.to(torch.float16)

            pos_bf16 = make_position_embeddings(model, h_bf16, position_ids)
            pos_fp16 = None
            if pos_bf16 is not None:
                pos_fp16 = tuple(x.to(torch.float16) for x in pos_bf16) if isinstance(pos_bf16, tuple) else pos_bf16.to(torch.float16)

            def fn_bf16():
                return run_layer(bf16_layer, h_bf16, position_ids, pos_bf16)

            def fn_pure():
                return run_layer(pure_layer, h_fp16, position_ids, pos_fp16)

            def fn_split():
                return run_layer(split_layer, h_fp16, position_ids, pos_fp16)

            bf16_ms = bench(fn_bf16, args.warmup, args.iters, device)
            pure_ms = bench(fn_pure, args.warmup, args.iters, device)
            split_ms = bench(fn_split, args.warmup, args.iters, device)

            row = {
                "label": args.label,
                "model": args.model,
                "layer_idx": layer_idx,
                "batch": batch,
                "seq_len": args.seq_len,
                "bf16_ms": bf16_ms,
                "pure_w4a4_qfactory_ms": pure_ms,
                "split_policy_qfactory_ms": split_ms,
                "pure_over_bf16": pure_ms / bf16_ms if bf16_ms > 0 else 0.0,
                "split_over_bf16": split_ms / bf16_ms if bf16_ms > 0 else 0.0,
                "split_over_pure": split_ms / pure_ms if pure_ms > 0 else 0.0,
                **ratio_summary,
            }
            rows.append(row)

            log("[RESULT] " + json.dumps(row, ensure_ascii=False))
            write_csv(out / f"{args.label}_layer_latency_v28.csv", rows)
            json.dump(rows, open(out / f"{args.label}_layer_latency_v28.json", "w"), indent=2, ensure_ascii=False)

        del bf16_layer, pure_layer, split_layer
        torch.cuda.empty_cache()

    for batch in batches:
        rs = [r for r in rows if int(r["batch"]) == batch]
        if not rs:
            continue

        def s(k):
            return sum(float(r[k]) for r in rs)

        summary = {
            "label": args.label,
            "model": args.model,
            "batch": batch,
            "seq_len": args.seq_len,
            "num_layers": len(rs),
            "sum_bf16_ms": s("bf16_ms"),
            "sum_pure_w4a4_qfactory_ms": s("pure_w4a4_qfactory_ms"),
            "sum_split_policy_qfactory_ms": s("split_policy_qfactory_ms"),
        }
        summary["pure_over_bf16"] = summary["sum_pure_w4a4_qfactory_ms"] / summary["sum_bf16_ms"]
        summary["split_over_bf16"] = summary["sum_split_policy_qfactory_ms"] / summary["sum_bf16_ms"]
        summary["split_over_pure"] = summary["sum_split_policy_qfactory_ms"] / summary["sum_pure_w4a4_qfactory_ms"]
        summary["avg_policy_avg_ratio"] = sum(float(r.get("policy_avg_ratio", 0.0)) for r in rs) / len(rs)
        summary["avg_policy_mac_weighted_ratio"] = sum(float(r.get("policy_mac_weighted_ratio", 0.0)) for r in rs) / len(rs)
        summary["avg_policy_max_ratio"] = sum(float(r.get("policy_max_ratio", 0.0)) for r in rs) / len(rs)

        summary_rows.append(summary)

    write_csv(out / f"{args.label}_summary_v28.csv", summary_rows)
    json.dump(summary_rows, open(out / f"{args.label}_summary_v28.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {out / f'{args.label}_layer_latency_v28.csv'}")
    log(f"[CSV] {out / f'{args.label}_summary_v28.csv'}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
