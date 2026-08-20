import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--policy", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=8)
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--max_layers", default="all")
    p.add_argument("--max_rows_per_module", type=int, default=2048)
    return p.parse_args()


def flatten_policy_ratios(obj):
    ratio_map = {}

    def norm(v):
        try:
            x = float(v)
        except Exception:
            return None
        if x > 1 and x <= 100:
            x /= 100.0
        return max(0.0, min(1.0, x))

    def walk(x, path):
        if isinstance(x, dict):
            if "ratio_projected" in x:
                name = x.get("name") or x.get("module") or x.get("linear") or ".".join(path)
                r = norm(x["ratio_projected"])
                if r is not None:
                    ratio_map[str(name)] = r
            for k, v in x.items():
                walk(v, path + [str(k)])
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, path + [str(i)])

    walk(obj, [])
    return ratio_map


def lookup_ratio(ratio_map, layer_idx, module_name):
    candidates = [
        f"modules.model.layers.{layer_idx}.{module_name}",
        f"model.layers.{layer_idx}.{module_name}",
        f"layers.{layer_idx}.{module_name}",
        f"{layer_idx}.{module_name}",
    ]
    for c in candidates:
        if c in ratio_map:
            return ratio_map[c], c

    for k, v in ratio_map.items():
        if f"model.layers.{layer_idx}.{module_name}" in k:
            return float(v), k
        if k.endswith(f"layers.{layer_idx}.{module_name}"):
            return float(v), k
    return 0.0, ""


def get_module(root, name):
    m = root
    for p in name.split("."):
        m = getattr(m, p)
    return m


def percentile(xs, q):
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = int(round((len(ys) - 1) * q))
    return float(ys[idx])


def update_metric_list(acc, key, vals):
    if not vals:
        return
    s = acc[key]
    s["sum"] += float(sum(vals))
    s["count"] += int(len(vals))
    s["min"] = min(s.get("min", float("inf")), float(min(vals)))
    s["max"] = max(s.get("max", float("-inf")), float(max(vals)))
    # 只保留少量样本用于 percentile，避免内存爆。
    store = s.setdefault("samples", [])
    budget = 4096
    if len(store) < budget:
        remain = budget - len(store)
        store.extend(float(x) for x in vals[:remain])


def metric_mean(acc, key):
    s = acc.get(key, {})
    if not s or s.get("count", 0) == 0:
        return 0.0
    return s["sum"] / s["count"]


def compute_metrics_from_idx(idx_np, R, K, max_rows):
    # idx_np: [M, R]
    M = int(idx_np.shape[0])
    if M <= 1 or R <= 0:
        return {}

    if M > max_rows:
        idx_np = idx_np[:max_rows]
        M = max_rows

    sets = [set(map(int, row)) for row in idx_np]
    out = defaultdict(lambda: {"sum": 0.0, "count": 0, "min": float("inf"), "max": float("-inf"), "samples": []})

    # prev1 prediction
    vals = []
    exact = []
    for i in range(1, M):
        inter = len(sets[i].intersection(sets[i - 1]))
        rec = inter / R
        vals.append(rec)
        exact.append(1.0 if inter == R else 0.0)
    update_metric_list(out, "prev1_recall", vals)
    update_metric_list(out, "prev1_exact", exact)

    # history union prediction
    for W in [2, 4, 8, 16, 32]:
        vals = []
        exact = []
        union_sizes = []
        for i in range(1, M):
            lo = max(0, i - W)
            u = set()
            for j in range(lo, i):
                u.update(sets[j])
            inter = len(sets[i].intersection(u))
            vals.append(inter / R)
            exact.append(1.0 if inter == R else 0.0)
            union_sizes.append(len(u))
        update_metric_list(out, f"hist{W}_recall", vals)
        update_metric_list(out, f"hist{W}_exact", exact)
        update_metric_list(out, f"hist{W}_union_over_R", [x / R for x in union_sizes])

    # block union upper bound for dense tail GEMM
    for B in [4, 8, 16, 32, 64]:
        union_over_R = []
        density = []
        for st in range(0, M, B):
            block = sets[st: st + B]
            if not block:
                continue
            u = set()
            for s in block:
                u.update(s)
            u_size = max(1, len(u))
            bsz = len(block)
            union_over_R.append(u_size / R)
            density.append((bsz * R) / (bsz * u_size))
        update_metric_list(out, f"block{B}_union_over_R", union_over_R)
        update_metric_list(out, f"block{B}_tail_density", density)

    # global frequency predictor in this collected segment
    cnt = Counter()
    for s in sets:
        cnt.update(s)
    ranked = [x for x, _ in cnt.most_common()]
    for mul in [1, 2, 4, 8]:
        topU = set(ranked[: min(K, mul * R)])
        vals = []
        for s in sets:
            vals.append(len(s.intersection(topU)) / R)
        update_metric_list(out, f"freq_top{mul}R_recall", vals)

    return out


class StatBook:
    def __init__(self):
        self.rows_seen = defaultdict(int)
        self.freq = defaultdict(Counter)
        self.metrics = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0, "min": float("inf"), "max": float("-inf"), "samples": []}))
        self.meta = {}

    def update(self, key, idx_np, R, K, meta, max_rows):
        self.meta[key] = meta
        self.rows_seen[key] += int(idx_np.shape[0])
        m = compute_metrics_from_idx(idx_np, R, K, max_rows=max_rows)

        for mk, mv in m.items():
            dst = self.metrics[key][mk]
            dst["sum"] += mv["sum"]
            dst["count"] += mv["count"]
            dst["min"] = min(dst["min"], mv["min"])
            dst["max"] = max(dst["max"], mv["max"])
            store = dst.setdefault("samples", [])
            if len(store) < 4096:
                remain = 4096 - len(store)
                store.extend(mv.get("samples", [])[:remain])

        # frequency over all rows
        for row in idx_np[:max_rows]:
            self.freq[key].update(map(int, row))

    def to_rows(self):
        rows = []
        for key, meta in sorted(self.meta.items()):
            row = dict(meta)
            row["rows_seen"] = self.rows_seen[key]

            for mk, mv in sorted(self.metrics[key].items()):
                c = mv.get("count", 0)
                if c > 0:
                    row[mk + "_mean"] = mv["sum"] / c
                    row[mk + "_min"] = mv["min"]
                    row[mk + "_max"] = mv["max"]
                    samples = mv.get("samples", [])
                    row[mk + "_p50"] = percentile(samples, 0.50)
                    row[mk + "_p90"] = percentile(samples, 0.90)
                else:
                    row[mk + "_mean"] = 0.0

            cnt = self.freq[key]
            R = int(meta["R"])
            total = max(1, sum(cnt.values()))
            for mul in [1, 2, 4, 8]:
                top = cnt.most_common(mul * R)
                row[f"freq_top{mul}R_unique"] = len(top)
                row[f"freq_top{mul}R_mass"] = sum(v for _, v in top) / total
            rows.append(row)
        return rows


def build_batches(args, tokenizer):
    seq_len = args.seq_len
    total_tokens = args.num_batches * args.batch_size * seq_len + seq_len

    text = ""
    if args.dataset.lower() in {"wikitext2", "wikitext"}:
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="test",
                cache_dir=os.environ.get("HF_DATASETS_CACHE", None),
            )
            chunks = [x["text"] for x in ds if len(x.get("text", "")) > 50]
            text = "\n\n".join(chunks)
            log(f"[DATASET] loaded wikitext2 chunks={len(chunks)}")
        except Exception as e:
            log(f"[WARN] failed to load local wikitext2, fallback synthetic text: {repr(e)}")

    if not text:
        text = (
            "Large language model quantization requires robust activation handling. "
            "We profile row-wise tail index locality for W4A4 split inference. "
            "This fallback text is repeated to create token sequences. "
        ) * 4096

    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() < total_tokens:
        reps = int(math.ceil(total_tokens / ids.numel()))
        ids = ids.repeat(reps)

    batches = []
    ptr = 0
    for _ in range(args.num_batches):
        cur = []
        for _ in range(args.batch_size):
            cur.append(ids[ptr: ptr + seq_len])
            ptr += seq_len
        batches.append(torch.stack(cur, dim=0))
    return batches


def main():
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    log(f"[MODEL] {args.model}")
    log(f"[POLICY] {args.policy}")
    log(f"[OUT] {out}")
    log(f"[DEVICE] {torch.cuda.get_device_name(device)}")
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[BATCH_SIZE] {args.batch_size}")
    log(f"[NUM_BATCHES] {args.num_batches}")

    policy = json.load(open(args.policy))
    ratio_map = flatten_policy_ratios(policy)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device).eval()

    layers = model.model.layers
    num_layers = len(layers)

    if args.max_layers == "all":
        layer_ids = list(range(num_layers))
    else:
        layer_ids = [int(x) for x in args.max_layers.split(",") if x.strip()]

    book = StatBook()
    hooks = []

    for layer_idx in layer_ids:
        layer = layers[layer_idx]
        for module_name in TARGET_LINEAR_NAMES:
            mod = get_module(layer, module_name)
            K = int(mod.weight.shape[1])
            N = int(mod.weight.shape[0])
            ratio, policy_key = lookup_ratio(ratio_map, layer_idx, module_name)
            R = int(math.ceil(K * ratio))
            meta = {
                "layer_idx": layer_idx,
                "module": module_name,
                "K": K,
                "N": N,
                "ratio_projected": ratio,
                "R": R,
                "policy_key": policy_key,
            }
            key = f"{layer_idx}:{module_name}"

            def make_hook(kkey, kk, rr, mmeta):
                def hook(module, inputs):
                    if rr <= 0:
                        return
                    A = inputs[0].detach()
                    A2 = A.reshape(-1, A.shape[-1])
                    # topk on abs activation; sorted=False is enough for set overlap.
                    idx = torch.topk(A2.abs(), k=rr, dim=-1, largest=True, sorted=False).indices
                    idx_np = idx.to("cpu", non_blocking=False).numpy()
                    book.update(kkey, idx_np, rr, kk, mmeta, args.max_rows_per_module)
                return hook

            hooks.append(mod.register_forward_pre_hook(make_hook(key, K, R, meta)))

    batches = build_batches(args, tokenizer)

    log(f"[HOOKS] {len(hooks)}")
    log(f"[LAYERS] {layer_ids[:8]} ... total={len(layer_ids)}")

    for bi, batch in enumerate(batches):
        input_ids = batch.to(device)
        log(f"[FORWARD] batch_id={bi} shape={tuple(input_ids.shape)}")
        with torch.inference_mode():
            _ = model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize(device)

        rows = book.to_rows()
        write_csv(out / "tail_locality_by_linear_v25.csv", rows)
        json.dump(rows, open(out / "tail_locality_by_linear_v25.json", "w"), indent=2, ensure_ascii=False)

    for h in hooks:
        h.remove()

    rows = book.to_rows()
    write_csv(out / "tail_locality_by_linear_v25.csv", rows)
    json.dump(rows, open(out / "tail_locality_by_linear_v25.json", "w"), indent=2, ensure_ascii=False)

    # module-family summary
    fam = defaultdict(list)
    for r in rows:
        fam[r["module"]].append(r)

    summary = []
    for module, rs in sorted(fam.items()):
        def avg(field):
            vals = [float(x.get(field, 0.0)) for x in rs if x.get(field, "") != ""]
            return sum(vals) / len(vals) if vals else 0.0

        summary.append({
            "module": module,
            "num_layers": len(rs),
            "avg_ratio_projected": avg("ratio_projected"),
            "avg_R": avg("R"),
            "avg_prev1_recall": avg("prev1_recall_mean"),
            "avg_hist4_recall": avg("hist4_recall_mean"),
            "avg_hist8_recall": avg("hist8_recall_mean"),
            "avg_hist16_recall": avg("hist16_recall_mean"),
            "avg_block8_union_over_R": avg("block8_union_over_R_mean"),
            "avg_block16_union_over_R": avg("block16_union_over_R_mean"),
            "avg_block32_union_over_R": avg("block32_union_over_R_mean"),
            "avg_block16_tail_density": avg("block16_tail_density_mean"),
            "avg_freq_top2R_recall": avg("freq_top2R_recall_mean"),
            "avg_freq_top4R_recall": avg("freq_top4R_recall_mean"),
        })

    write_csv(out / "tail_locality_summary_by_module_v25.csv", summary)
    json.dump(summary, open(out / "tail_locality_summary_by_module_v25.json", "w"), indent=2, ensure_ascii=False)

    # global quick interpretation table
    quick = []
    for r in rows:
        hist8 = float(r.get("hist8_recall_mean", 0.0))
        block16 = float(r.get("block16_union_over_R_mean", 999.0))
        prev1 = float(r.get("prev1_recall_mean", 0.0))
        recommendation = "low_locality"
        if hist8 >= 0.95 and block16 <= 4.0:
            recommendation = "prediction_dense_tail_promising"
        elif hist8 >= 0.90:
            recommendation = "prediction_promising"
        elif block16 <= 4.0:
            recommendation = "block_union_dense_tail_promising"

        quick.append({
            "layer_idx": r["layer_idx"],
            "module": r["module"],
            "ratio_projected": r["ratio_projected"],
            "R": r["R"],
            "prev1_recall": prev1,
            "hist8_recall": hist8,
            "block16_union_over_R": block16,
            "recommendation": recommendation,
        })

    write_csv(out / "tail_locality_recommendation_v25.csv", quick)
    json.dump(quick, open(out / "tail_locality_recommendation_v25.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {out / 'tail_locality_by_linear_v25.csv'}")
    log(f"[CSV] {out / 'tail_locality_summary_by_module_v25.csv'}")
    log(f"[CSV] {out / 'tail_locality_recommendation_v25.csv'}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
