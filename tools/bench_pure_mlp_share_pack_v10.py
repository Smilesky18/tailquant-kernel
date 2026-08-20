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


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batches", default="16,64")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
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


@torch.no_grad()
def bench_graph(fn, warmup: int, iters: int, device):
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _ = fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters)


def pure_linear_from_prepacked_A(linear, A_pack, a_scale, M: int):
    """
    与 PolicyW4A4Linear._pure 等价，但跳过 pack_a_full_s4。
    只复用已经 pack 好的 A_pack 和 a_scale。
    """
    output = torch.empty(
        (M, linear.N),
        dtype=torch.float16,
        device=a_scale.device,
    )

    scratch = linear.scratch_pool.get(
        M,
        linear.K,
        linear.N,
    )

    linear.ext.cutlass_s4_gemm(
        A_pack,
        linear.B_col,
        scratch["C_i32"],
        M,
        linear.N,
        linear.K,
    )
    linear.ext.scale_i32_to_fp16(
        scratch["C_i32"],
        a_scale,
        linear.w_scale,
        output,
    )
    return output


def install_mlp_gate_up_shared_pack(mlp):
    """
    替换 Qwen3 MLP forward：
    原始：
        gate = gate_proj(x)  # pack once
        up   = up_proj(x)    # pack once again
        down = down_proj(act(gate) * up)

    v10：
        pack x once
        gate/up share A_pack and a_scale
        down_proj 保持原始 pure W4A4 路径
    """
    gate = mlp.gate_proj
    up = mlp.up_proj
    down = mlp.down_proj

    required = ["ext", "B_col", "w_scale", "scratch_pool", "K", "N", "eps"]
    for mod_name, mod in [("gate_proj", gate), ("up_proj", up)]:
        for attr in required:
            if not hasattr(mod, attr):
                raise RuntimeError(f"{mod_name} missing attr {attr}")

    if gate.K != up.K:
        raise RuntimeError(f"gate/up K mismatch: {gate.K} vs {up.K}")
    if gate.N != up.N:
        raise RuntimeError(f"gate/up N mismatch: {gate.N} vs {up.N}")

    act_fn = getattr(mlp, "act_fn", None)
    if act_fn is None:
        act_fn = getattr(mlp, "activation_fn", None)
    if act_fn is None:
        raise RuntimeError("Cannot find mlp.act_fn or mlp.activation_fn")

    def shared_forward(self, x):
        original_shape = x.shape[:-1]
        A = x.reshape(-1, gate.K).contiguous()
        if A.dtype != torch.float16:
            A = A.to(torch.float16)

        M = int(A.shape[0])

        # 只 pack 一次。使用 gate 的 scratch 保存 A_pack/a_scale。
        scratch = gate.scratch_pool.get(
            M,
            gate.K,
            gate.N,
        )

        gate.ext.pack_a_full_s4(
            A,
            scratch["A_pack"],
            scratch["a_scale"],
            gate.eps,
        )

        gate_out = pure_linear_from_prepacked_A(
            gate,
            scratch["A_pack"],
            scratch["a_scale"],
            M,
        )
        up_out = pure_linear_from_prepacked_A(
            up,
            scratch["A_pack"],
            scratch["a_scale"],
            M,
        )

        hidden = act_fn(gate_out) * up_out
        hidden = hidden.reshape(*original_shape, gate.N).contiguous()

        return down(hidden)

    mlp.forward = types.MethodType(shared_forward, mlp)
    return mlp


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
    log(f"[SEQ_LEN] {args.seq_len}")
    log(f"[BATCHES] {args.batches}")
    log("[NOTE] Pure W4A4 only. v10 shares MLP gate/up activation pack once.")

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

    pure_orig = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    pure_orig, pure_orig_records = V8.patch_layer_with_real_policy(
        layer=pure_orig,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="pure",
        ratio=0.0,
        eps=args.eps,
        device=device,
    )
    pure_orig.to(device=device).eval()

    pure_shared = copy.deepcopy(base_layer).to(device=device, dtype=torch.float16).eval()
    pure_shared, pure_shared_records = V8.patch_layer_with_real_policy(
        layer=pure_shared,
        B=B,
        main_ext=main_ext,
        layout_ext=layout_ext,
        policy_pack_ext=policy_pack_ext,
        mode="pure",
        ratio=0.0,
        eps=args.eps,
        device=device,
    )
    pure_shared.to(device=device).eval()

    if not hasattr(pure_shared, "mlp"):
        raise RuntimeError("layer has no mlp")
    install_mlp_gate_up_shared_pack(pure_shared.mlp)

    rows = []
    batches = [int(x) for x in args.batches.split(",") if x.strip()]

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
        log("[POSITION_EMBEDDINGS] " + str([(tuple(t.shape), str(t.dtype)) for t in pe]))

        with torch.no_grad():
            y_orig = run_layer_once(pure_orig, hidden, position_ids, pe)
            y_shared = run_layer_once(pure_shared, hidden, position_ids, pe)
            torch.cuda.synchronize(device)
            diff = (y_orig.float() - y_shared.float()).abs()
            max_abs = float(diff.max().item())
            mean_abs = float(diff.mean().item())
            rmse = float(torch.sqrt((diff * diff).mean()).item())
            log(f"[CHECK] max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} rmse={rmse:.6e}")

        orig_fn = lambda: run_layer_once(pure_orig, hidden, position_ids, pe)
        shared_fn = lambda: run_layer_once(pure_shared, hidden, position_ids, pe)

        torch.cuda.empty_cache()
        orig_ms = bench_graph(orig_fn, args.warmup, args.iters, device)
        shared_ms = bench_graph(shared_fn, args.warmup, args.iters, device)

        row = {
            "model": args.model,
            "layer_idx": args.layer_idx,
            "batch": batch,
            "seq_len": args.seq_len,
            "hidden_size": hidden_size,
            "pure_orig_ms": orig_ms,
            "pure_mlp_gate_up_shared_pack_ms": shared_ms,
            "speedup_orig_over_shared": orig_ms / shared_ms,
            "shared_over_orig": shared_ms / orig_ms,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "rmse_diff": rmse,
            "note": "Pure W4A4. MLP gate/up share one pack_a_full_s4. down_proj unchanged.",
        }
        rows.append(row)
        log("[RESULT] " + json.dumps(row, indent=2))

    csv_path = out_dir / "pure_mlp_gate_up_shared_pack_v10.csv"
    json_path = out_dir / "pure_mlp_gate_up_shared_pack_v10.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json.dump(rows, open(json_path, "w"), indent=2, ensure_ascii=False)
    json.dump(pure_orig_records, open(out_dir / "pure_orig_patch_records_v10.json", "w"), indent=2, ensure_ascii=False)
    json.dump(pure_shared_records, open(out_dir / "pure_shared_patch_records_v10.json", "w"), indent=2, ensure_ascii=False)

    log(f"[CSV] {csv_path}")
    log(f"[JSON] {json_path}")
    log(f"[DONE] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")


if __name__ == "__main__":
    main()
