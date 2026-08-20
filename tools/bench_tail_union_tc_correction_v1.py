#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

import bench_layer_bf16_pure_split_no_gptq_v8 as V8
import bench_layer_qfactory_raw_backend_v19 as QF19
from bench_multimodel_all_layers_policy_async_ablation_v30 import (
    TARGET_SUFFIXES,
    install_qfactory_fast_preset,
    load_policy,
    policy_key,
)


def log(msg: str):
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='Qwen/Qwen3-8B')
    p.add_argument('--layer_idx', type=int, default=19)
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--batches', default='16,64,256')
    p.add_argument('--modules', default='all')
    p.add_argument('--policy', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--local_files_only', action='store_true')
    p.add_argument('--qfactory_fast_preset', default='qwen3_sm120_v1', choices=['none', 'qwen3_sm120_v1'])
    p.add_argument('--cutlass_path', default=os.environ.get('CUTLASS_PATH', '/data/yzy/quarot-gpt-2/third_party/cutlass'))
    p.add_argument('--align_k', type=int, default=128)
    p.add_argument('--skip_fullk_tc', action='store_true')
    return p.parse_args()


def parse_csv_ints(text: str) -> List[int]:
    return [int(x) for x in text.split(',') if x.strip()]


def iter_target_linears(layer: nn.Module):
    found = []

    def rec(parent: nn.Module, prefix: str):
        for child_name, child in list(parent.named_children()):
            full = f'{prefix}.{child_name}' if prefix else child_name
            linear = None
            if isinstance(child, nn.Linear):
                linear = child
            else:
                inner = getattr(child, 'module', None)
                if isinstance(inner, nn.Linear):
                    linear = inner
            if linear is not None:
                if full in TARGET_SUFFIXES:
                    found.append((parent, child_name, full, linear))
            else:
                rec(child, full)

    rec(layer, '')
    return found


def select_modules(text: str) -> set:
    if text == 'all':
        return set(TARGET_SUFFIXES)
    requested = set()
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if item in TARGET_SUFFIXES:
            requested.add(item)
        else:
            matches = [x for x in TARGET_SUFFIXES if x.endswith(item)]
            if not matches:
                raise ValueError(f'unknown module selector: {item}')
            requested.update(matches)
    return requested


def ceil_ratio_count(K: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return max(1, int(math.ceil(float(K) * float(ratio))))


def align_up(x: int, a: int) -> int:
    if a <= 1:
        return x
    return ((x + a - 1) // a) * a


def cuda_time_ms(fn, warmup: int, iters: int, device: torch.device) -> float:
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)) / float(iters)


def pack_weight_col_from_existing_scale(main_ext, layout_ext, W_T: torch.Tensor, w_scale: torch.Tensor, K: int, N: int):
    B_row = torch.empty((K * N + 1) // 2, dtype=torch.uint8, device=W_T.device)
    B_col = torch.empty_like(B_row)
    main_ext.pack_weight_rowmajor_s4_from_scale(W_T, B_row, w_scale)
    layout_ext.row_to_col_s4_tiled(B_row, B_col, K, N)
    return B_row, B_col


def capture_layer_inputs(model, layer, layer_idx: int, batch: int, seq_len: int, hidden_size: int, device: torch.device, module_names: set):
    captures: Dict[str, torch.Tensor] = {}
    handles = []
    for _, _, local_name, module in iter_target_linears(layer):
        if local_name not in module_names:
            continue
        def make_hook(name):
            def hook(_mod, inp):
                x = inp[0].detach()
                captures[name] = x.reshape(-1, x.shape[-1]).contiguous()
            return hook
        handles.append(module.register_forward_pre_hook(make_hook(local_name)))

    hidden = torch.randn(batch, seq_len, hidden_size, device=device, dtype=torch.float16)
    position_ids = V8.make_position_ids(batch, seq_len, device)
    pe = V8.build_position_embeddings(model, hidden, position_ids, torch.float16)
    with torch.no_grad():
        out = layer(hidden, position_ids=position_ids, position_embeddings=pe)
        if isinstance(out, tuple):
            out = out[0]
    torch.cuda.synchronize(device)
    for h in handles:
        h.remove()
    del hidden, position_ids, pe, out
    return captures


def make_top_data(A: torch.Tensor, R: int, eps: float):
    _, idx = torch.topk(A.abs().float(), k=R, dim=1, largest=True, sorted=False)
    idx, _ = torch.sort(idx, dim=1)
    idx = idx.to(torch.int32).contiguous()
    vals = A.gather(1, idx.long()).contiguous()
    top_scale = vals.abs().float().amax(dim=1).clamp_min(eps) / 7.0
    top_q = torch.round(vals.float() / top_scale.view(-1, 1)).clamp(-8, 7).to(torch.int8).contiguous()
    return idx, vals, top_q, top_scale.contiguous()


def materialize_union(A_tail: torch.Tensor, vals: torch.Tensor, union_pos: torch.Tensor):
    A_tail.zero_()
    A_tail.scatter_(1, union_pos, vals)


def materialize_fullk(A_full: torch.Tensor, vals: torch.Tensor, idx: torch.Tensor):
    A_full.zero_()
    A_full.scatter_(1, idx.long(), vals)


def bench_module(args, main_ext, layout_ext, module_name: str, A: torch.Tensor, weight: torch.Tensor, cfg: dict, device: torch.device):
    M, K = map(int, A.shape)
    N = int(weight.shape[0])
    ratio = float(cfg['ratio'])
    R = ceil_ratio_count(K, ratio)
    if R <= 0:
        return None

    W_T = weight.detach().to(device=device, dtype=torch.float16).t().contiguous()
    w_scale = weight.detach().float().abs().amax(dim=1).clamp_min(args.eps).to(device=device) / 7.0
    w_scale = w_scale.contiguous()

    idx, vals, top_q, top_scale = make_top_data(A, R, args.eps)
    union_idx = torch.unique(idx.reshape(-1), sorted=True)
    U = int(union_idx.numel())
    U_pad = align_up(U, args.align_k)
    pad = U_pad - U
    if pad:
        # Padded rows are zero weights and zero activations. Use index 0 only as a placeholder before overwrite.
        W_tail = torch.empty((U_pad, N), dtype=torch.float16, device=device)
        W_tail[:U].copy_(W_T.index_select(0, union_idx.long()))
        W_tail[U:].zero_()
    else:
        W_tail = W_T.index_select(0, union_idx.long()).contiguous()
    W_tail = W_tail.contiguous()

    union_pos = torch.searchsorted(union_idx, idx.reshape(-1)).reshape(M, R).long().contiguous()
    A_tail = torch.empty((M, U_pad), dtype=torch.float16, device=device)
    A_tail_pack = torch.empty((M * U_pad + 1) // 2, dtype=torch.uint8, device=device)
    A_tail_scale = torch.empty((M,), dtype=torch.float32, device=device)
    C_tail = torch.empty((M, N), dtype=torch.int32, device=device)
    Y_tail = torch.empty((M, N), dtype=torch.float16, device=device)
    _, B_tail_col = pack_weight_col_from_existing_scale(main_ext, layout_ext, W_tail, w_scale, U_pad, N)

    B_full_row, B_full_col = pack_weight_col_from_existing_scale(main_ext, layout_ext, W_T, w_scale, K, N)
    Y_sparse = torch.empty((M, N), dtype=torch.float16, device=device)

    def current_sparse_add():
        Y_sparse.zero_()
        main_ext.sparse_top_add_rowmajor_quad_shared(top_q, idx, B_full_row, top_scale, w_scale, Y_sparse, K)

    def union_materialize():
        materialize_union(A_tail, vals, union_pos)

    def union_pack_gemm_scale():
        main_ext.pack_a_full_s4(A_tail, A_tail_pack, A_tail_scale, float(args.eps))
        C = QF19.qfactory_gemm_raw(A_tail_pack, B_tail_col, C_tail, M, N, U_pad)
        main_ext.scale_i32_to_fp16(C, A_tail_scale, w_scale, Y_tail)

    def union_tc_e2e():
        materialize_union(A_tail, vals, union_pos)
        main_ext.pack_a_full_s4(A_tail, A_tail_pack, A_tail_scale, float(args.eps))
        C = QF19.qfactory_gemm_raw(A_tail_pack, B_tail_col, C_tail, M, N, U_pad)
        main_ext.scale_i32_to_fp16(C, A_tail_scale, w_scale, Y_tail)

    # Warm qfactory JIT and buffers.
    union_tc_e2e()
    current_sparse_add()
    torch.cuda.synchronize(device)

    row = {
        'module': module_name,
        'M': M,
        'K': K,
        'N': N,
        'ratio': ratio,
        'R': R,
        'U': U,
        'U_pad': U_pad,
        'U_over_K': U / float(K),
        'U_over_R': U / float(R),
        'U_pad_over_K': U_pad / float(K),
        'tail_density_in_union': (M * R) / float(M * U_pad),
        'current_sparse_zero_plus_add_ms': cuda_time_ms(current_sparse_add, args.warmup, args.iters, device),
        'union_materialize_scatter_ms': cuda_time_ms(union_materialize, args.warmup, args.iters, device),
        'union_tc_pack_gemm_scale_ms': cuda_time_ms(union_pack_gemm_scale, args.warmup, args.iters, device),
        'union_tc_e2e_ms': cuda_time_ms(union_tc_e2e, args.warmup, args.iters, device),
    }
    row['union_tc_e2e_over_current_sparse'] = row['union_tc_e2e_ms'] / max(row['current_sparse_zero_plus_add_ms'], 1e-9)
    row['union_tc_compute_over_current_sparse'] = row['union_tc_pack_gemm_scale_ms'] / max(row['current_sparse_zero_plus_add_ms'], 1e-9)

    if not args.skip_fullk_tc:
        A_full = torch.empty((M, K), dtype=torch.float16, device=device)
        A_full_pack = torch.empty((M * K + 1) // 2, dtype=torch.uint8, device=device)
        A_full_scale = torch.empty((M,), dtype=torch.float32, device=device)
        C_full = torch.empty((M, N), dtype=torch.int32, device=device)
        Y_full = torch.empty((M, N), dtype=torch.float16, device=device)

        def fullk_materialize():
            materialize_fullk(A_full, vals, idx)

        def fullk_pack_gemm_scale():
            main_ext.pack_a_full_s4(A_full, A_full_pack, A_full_scale, float(args.eps))
            C = QF19.qfactory_gemm_raw(A_full_pack, B_full_col, C_full, M, N, K)
            main_ext.scale_i32_to_fp16(C, A_full_scale, w_scale, Y_full)

        def fullk_tc_e2e():
            materialize_fullk(A_full, vals, idx)
            main_ext.pack_a_full_s4(A_full, A_full_pack, A_full_scale, float(args.eps))
            C = QF19.qfactory_gemm_raw(A_full_pack, B_full_col, C_full, M, N, K)
            main_ext.scale_i32_to_fp16(C, A_full_scale, w_scale, Y_full)

        fullk_tc_e2e(); torch.cuda.synchronize(device)
        row['fullk_materialize_scatter_ms'] = cuda_time_ms(fullk_materialize, args.warmup, args.iters, device)
        row['fullk_tc_pack_gemm_scale_ms'] = cuda_time_ms(fullk_pack_gemm_scale, args.warmup, args.iters, device)
        row['fullk_tc_e2e_ms'] = cuda_time_ms(fullk_tc_e2e, args.warmup, args.iters, device)
        row['fullk_tc_e2e_over_current_sparse'] = row['fullk_tc_e2e_ms'] / max(row['current_sparse_zero_plus_add_ms'], 1e-9)

    # Avoid keeping giant module buffers across modules.
    del W_T, w_scale, idx, vals, top_q, top_scale, union_idx, union_pos
    del A_tail, A_tail_pack, A_tail_scale, C_tail, Y_tail, B_tail_col, B_full_row, B_full_col, Y_sparse
    torch.cuda.empty_cache()
    return row


def write_csv(path: Path, rows: List[dict]):
    fields = sorted({k for row in rows for k in row})
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    log(f'[START] {time.strftime("%Y-%m-%dT%H:%M:%S%z")}')
    log(f'[MODEL] {args.model}')
    log(f'[LAYER] {args.layer_idx}')
    log(f'[BATCHES] {args.batches}')
    log(f'[POLICY] {args.policy}')
    log(f'[QFACTORY_CACHE_DIR] {os.environ.get("QFACTORY_CACHE_DIR")}')
    install_qfactory_fast_preset(args.qfactory_fast_preset)

    import kernel_quant.scripts.bench_real_split_fullstack_v1 as B
    main_ext, layout_ext, _ = V8.resolve_extensions(B, args, out_dir)
    policy = load_policy(Path(args.policy), force_activation_percentile_100=False)
    selected = select_modules(args.modules)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()
    layers = V8.get_layers(model)
    layer = layers[args.layer_idx].to(device=device, dtype=torch.float16).eval()
    hidden_size = V8.infer_hidden_size(model)
    linears = {name: linear for _, _, name, linear in iter_target_linears(layer)}

    rows = []
    for batch in parse_csv_ints(args.batches):
        log(f'\n[BATCH_BEGIN] {batch}')
        captures = capture_layer_inputs(model, layer, args.layer_idx, batch, args.seq_len, hidden_size, device, selected)
        for module_name in TARGET_SUFFIXES:
            if module_name not in selected:
                continue
            key = policy_key(args.layer_idx, module_name)
            cfg = policy['modules'].get(key)
            if cfg is None:
                raise KeyError(key)
            ratio = float(cfg['ratio'])
            if ratio <= 0:
                log(f'[SKIP_ZERO_RATIO] batch={batch} module={module_name}')
                continue
            A = captures[module_name]
            weight = linears[module_name].weight.detach()
            log(f'[MODULE_BEGIN] batch={batch} module={module_name} shape=A{tuple(A.shape)} W{tuple(weight.shape)} ratio={ratio}')
            row = bench_module(args, main_ext, layout_ext, module_name, A, weight, cfg, device)
            if row is not None:
                row.update({
                    'model': args.model,
                    'layer_idx': args.layer_idx,
                    'batch': batch,
                    'seq_len': args.seq_len,
                    'policy_file': str(args.policy),
                    'backend': 'qfactory_int4_tc_for_union_and_fullk; current_sparse_quad_shared',
                    'note': 'module_correction_microbench_random_layer_inputs_no_gptq',
                })
                rows.append(row)
                log('[ROW] ' + json.dumps(row, ensure_ascii=False))
                write_csv(out_dir / 'tail_union_tc_correction_v1_partial.csv', rows)
                (out_dir / 'tail_union_tc_correction_v1_partial.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
            del A
            torch.cuda.empty_cache()
        del captures
        gc.collect(); torch.cuda.empty_cache()

    csv_path = out_dir / 'tail_union_tc_correction_v1.csv'
    json_path = out_dir / 'tail_union_tc_correction_v1.json'
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    log(f'[CSV] {csv_path}')
    log(f'[JSON] {json_path}')
    log(f'[END] {time.strftime("%Y-%m-%dT%H:%M:%S%z")}')


if __name__ == '__main__':
    main()
