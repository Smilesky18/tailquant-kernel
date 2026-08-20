import argparse
import copy
import csv
import json
import math
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
    p.add_argument("--layers", default="all")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--cutlass_path", default=os.environ.get("CUTLASS_PATH", "/data/yzy/quarot-gpt-2/third_party/cutlass"))
    return p.parse_args()


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise RuntimeError("cannot locate model.model.layers")


def get_hidden_size(model):
    cfg = model.config
    for name in ["hidden_size", "n_embd", "d_model"]:
        if hasattr(cfg, name):
            return int(getattr(cfg, name))
    raise RuntimeError("cannot locate hidden size")


def get_submodule(root, name):
    m = root
    for p in name.split("."):
        if not hasattr(m, p):
            return None
        m = getattr(m, p)
    return m


def pick_float(d, keys, default):
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)


def normalize_ratio(x):
    x = float(x)
    if x > 1.0 and x <= 100.0:
        x /= 100.0
    return max(0.0, min(1.0, x))


def normalize_percentile(x):
    x = float(x)
    # policy 里有的可能写 99 / 99.9，有的可能写 0.99 / 0.999。
    if x > 1.0 and x <= 100.0:
        x /= 100.0
    return max(0.0, min(1.0, x))


def flatten_policy_entries(policy_obj):
    """
    尽量鲁棒地从 policy.json 中抓取每个 Linear 的完整字段：
      ratio_projected / ratio_continuous / ratio
      activation_percentile / weight_percentile
    """
    entries = {}

    def maybe_record(d, path):
        if not isinstance(d, dict):
            return

        text = " ".join(str(x) for x in path)
        name = d.get("name") or d.get("module") or d.get("linear") or d.get("key") or ""
        full_text = text + " " + str(name)

        hit_module = None
        hit_layer = None

        for mod in TARGET_LINEAR_NAMES:
            if mod in full_text:
                hit_module = mod
                break

        # 常见路径里含 layers.12.xxx；否则尝试从字段读 layer_idx。
        import re
        m = re.search(r"layers\.(\d+)\.", full_text)
        if m:
            hit_layer = int(m.group(1))
        elif "layer_idx" in d:
            try:
                hit_layer = int(d["layer_idx"])
            except Exception:
                hit_layer = None
        elif "layer" in d:
            try:
                hit_layer = int(d["layer"])
            except Exception:
                hit_layer = None

        ratio_keys = [
            "ratio_projected",
            "projected_ratio",
            "ratio",
            "ratio_continuous",
            "continuous_ratio",
            "r",
        ]

        has_ratio = any(k in d for k in ratio_keys)
        has_pct = any(k in d for k in [
            "activation_percentile",
            "act_percentile",
            "body_percentile",
            "a_percentile",
            "weight_percentile",
            "w_percentile",
        ])

        if hit_module is not None and hit_layer is not None and (has_ratio or has_pct):
            ratio_projected = pick_float(d, ["ratio_projected", "projected_ratio", "ratio"], 0.0)
            ratio_continuous = pick_float(d, ["ratio_continuous", "continuous_ratio", "ratio"], ratio_projected)
            ratio = pick_float(d, ["ratio", "ratio_projected", "projected_ratio"], ratio_projected)

            act_pct = pick_float(
                d,
                [
                    "activation_percentile",
                    "act_percentile",
                    "a_percentile",
                    "body_percentile",
                    "body_activation_percentile",
                    "input_percentile",
                    "percentile",
                ],
                1.0,
            )
            w_pct = pick_float(
                d,
                [
                    "weight_percentile",
                    "w_percentile",
                    "weight_clip_percentile",
                    "w_clip_percentile",
                    "weight_percent",
                ],
                1.0,
            )

            entries[(hit_layer, hit_module)] = {
                "name": str(name) if name else f"model.layers.{hit_layer}.{hit_module}",
                "module": hit_module,
                "layer_idx": hit_layer,
                "ratio": normalize_ratio(ratio),
                "ratio_continuous": normalize_ratio(ratio_continuous),
                "ratio_projected": normalize_ratio(ratio_projected),
                "activation_percentile": normalize_percentile(act_pct),
                "act_percentile": normalize_percentile(act_pct),
                "body_percentile": normalize_percentile(act_pct),
                "weight_percentile": normalize_percentile(w_pct),
                "w_percentile": normalize_percentile(w_pct),
                "raw_policy_keys": ",".join(sorted(d.keys())),
            }

    def walk(x, path):
        if isinstance(x, dict):
            maybe_record(x, path)
            for k, v in x.items():
                walk(v, path + [str(k)])
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + [str(i)])

    walk(policy_obj, [])
    return entries


def make_policy_rows(base_layer, entries, layer_idx):
    rows = []
    for module_name in TARGET_LINEAR_NAMES:
        mod = get_submodule(base_layer, module_name)
        if mod is None or not hasattr(mod, "weight"):
            continue

        K = int(mod.weight.shape[1])
        r = entries.get((layer_idx, module_name), None)

        if r is None:
            row = {
                "layer_idx": layer_idx,
                "module": module_name,
                "name": f"model.layers.{layer_idx}.{module_name}",
                "ratio": 0.0,
                "ratio_continuous": 0.0,
                "ratio_projected": 0.0,
                "activation_percentile": 1.0,
                "act_percentile": 1.0,
                "body_percentile": 1.0,
                "weight_percentile": 1.0,
                "w_percentile": 1.0,
                "missing_policy": 1,
            }
        else:
            row = dict(r)
            row["missing_policy"] = 0

        row["R"] = int(math.ceil(K * float(row["ratio_projected"])))
        row["K"] = K
        row["N"] = int(mod.weight.shape[0])
        row["mode"] = "split" if row["R"] > 0 else "pure"
        rows.append(row)

    return rows


def make_pure_rows_like_policy(policy_rows):
    pure_rows = []
    for r in policy_rows:
        x = dict(r)
        x["ratio"] = 0.0
        x["ratio_continuous"] = 0.0
        x["ratio_projected"] = 0.0
        x["R"] = 0
        x["mode"] = "pure"
        # 关键：percentile 继续沿用 policy，不要默认丢掉。
        pure_rows.append(x)
    return pure_rows


def layer_ratio_summary(policy_rows):
    if not policy_rows:
        return {
            "policy_avg_ratio": 0.0,
            "policy_mac_weighted_ratio": 0.0,
            "policy_max_ratio": 0.0,
            "policy_nonzero_linears": 0,
            "policy_missing_linears": 0,
        }

    ratios = [float(r.get("ratio_projected", 0.0)) for r in policy_rows]
    macs = [float(r.get("K", 0)) * float(r.get("N", 0)) for r in policy_rows]
    denom = sum(macs) if sum(macs) > 0 else 1.0

    return {
        "policy_avg_ratio": sum(ratios) / len(ratios),
        "policy_mac_weighted_ratio": sum(r * m for r, m in zip(ratios, macs)) / denom,
        "policy_max_ratio": max(ratios) if ratios else 0.0,
        "policy_nonzero_linears": sum(1 for r in ratios if r > 0),
        "policy_missing_linears": sum(int(r.get("missing_policy", 0)) for r in policy_rows),
    }


def ensure_scratch_has_pure_keys(layer, device):
    """
    修复 ratio=0 pure branch 的 scratch KeyError: 'a_scale'。
    某些 dual-policy scratch pool 只准备 split keys，ratio=0 走 _pure 时需要 A_pack/a_scale/C_i32。
    """
    patched = 0

    for mod in layer.modules():
        if not hasattr(mod, "scratch_pool"):
            continue
        if not hasattr(mod, "K") or not hasattr(mod, "N"):
            continue

        sp = mod.scratch_pool
        if getattr(sp, "_v28c_a_scale_patch", False):
            continue

        old_get = sp.get

        def new_get(M, K, N, old_get=old_get, mod=mod):
            scratch = old_get(M, K, N)

            K0 = int(getattr(mod, "K", K))
            N0 = int(getattr(mod, "N", N))

            if "A_pack" not in scratch:
                scratch["A_pack"] = torch.empty((M, K0 // 2), device=device, dtype=torch.uint8)
            if "a_scale" not in scratch:
                scratch["a_scale"] = torch.empty((M,), device=device, dtype=torch.float16)
            if "C_i32" not in scratch:
                scratch["C_i32"] = torch.empty((M, N0), device=device, dtype=torch.int32)

            return scratch

        sp.get = new_get
        sp._v28c_a_scale_patch = True
        patched += 1

    return patched


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


def patch_layer(base_layer, B, main_ext, layout_ext, policy_pack_ext, policy_rows, eps, device):
    layer, patch_records, seed_records = V22B.patch_layer_with_policy_ratios(
        base_layer=base_layer,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        policy_rows=policy_rows,
        eps=eps,
        device=device,
    )
    patched = ensure_scratch_has_pure_keys(layer, device)
    return layer, patch_records, seed_records, patched


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
    log(f"[LAYERS] {args.layers}")

    raw_policy = json.load(open(args.policy))
    entries = flatten_policy_entries(raw_policy)
    log(f"[POLICY_ENTRIES] {len(entries)}")

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

    if args.layers == "all":
        layer_ids = list(range(len(layers)))
    else:
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]

    rows = []
    policy_debug_rows = []

    for layer_idx in layer_ids:
        base_layer = layers[layer_idx]

        policy_rows = make_policy_rows(base_layer, entries, layer_idx)
        pure_policy_rows = make_pure_rows_like_policy(policy_rows)
        ratio_summary = layer_ratio_summary(policy_rows)

        log("")
        log(f"[LAYER] {layer_idx} " + json.dumps(ratio_summary, ensure_ascii=False))

        for r in policy_rows:
            dbg = {
                "label": args.label,
                "layer_idx": layer_idx,
                "module": r["module"],
                "K": r["K"],
                "N": r["N"],
                "ratio_projected": r["ratio_projected"],
                "ratio_continuous": r["ratio_continuous"],
                "activation_percentile": r["activation_percentile"],
                "weight_percentile": r["weight_percentile"],
                "R": r["R"],
                "mode": r["mode"],
                "missing_policy": r["missing_policy"],
            }
            policy_debug_rows.append(dbg)
            log("[POLICY_LINEAR] " + json.dumps(dbg, ensure_ascii=False))

        write_csv(out / f"{args.label}_policy_debug_v28c.csv", policy_debug_rows)

        bf16_layer = copy.deepcopy(base_layer).to(device).eval()

        pure_layer, pure_patch_records, _, pure_scratch_patched = patch_layer(
            base_layer=base_layer,
            B=B,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            policy_rows=pure_policy_rows,
            eps=args.eps,
            device=device,
        )

        split_layer, split_patch_records, _, split_scratch_patched = patch_layer(
            base_layer=base_layer,
            B=B,
            main_ext=main_ext,
            layout_ext=layout_ext,
            policy_pack_ext=policy_pack_ext,
            policy_rows=policy_rows,
            eps=args.eps,
            device=device,
        )

        log(f"[SCRATCH_PATCH] pure={pure_scratch_patched} split={split_scratch_patched}")

        for batch in batches:
            position_ids = torch.arange(args.seq_len, device=device).unsqueeze(0).repeat(batch, 1)
            h_bf16 = torch.randn((batch, args.seq_len, hidden_size), device=device, dtype=torch.bfloat16)
            h_fp16 = h_bf16.to(torch.float32)  # v28c: pack_a_full_s4 expects Float input

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

            write_csv(out / f"{args.label}_layer_latency_v28c.csv", rows)
            json.dump(rows, open(out / f"{args.label}_layer_latency_v28c.json", "w"), indent=2, ensure_ascii=False)

        del bf16_layer, pure_layer, split_layer
        torch.cuda.empty_cache()

    summary_rows = []
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

    write_csv(out / f"{args.label}_summary_v28c.csv", summary_rows)
    json.dump(summary_rows, open(out / f"{args.label}_summary_v28c.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {out / f'{args.label}_policy_debug_v28c.csv'}")
    log(f"[CSV] {out / f'{args.label}_layer_latency_v28c.csv'}")
    log(f"[CSV] {out / f'{args.label}_summary_v28c.csv'}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
