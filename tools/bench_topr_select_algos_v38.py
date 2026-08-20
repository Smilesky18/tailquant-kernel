import argparse
import csv
import json
import math
from pathlib import Path

import torch


def bench_ms(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        out = fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end) / iters), out


def current_select(A, R, activation_percentile):
    M, K = A.shape
    percentile = min(max(float(activation_percentile), 0.0), 100.0)
    body_len = K - R
    body_kth = min(K, max(1, int(math.ceil(body_len * percentile / 100.0))))
    descending_rank = K - body_kth + 1
    select_k = max(R, descending_rank)
    abs_A = A.abs().float()
    top_values, top_indices = torch.topk(abs_A, k=select_k, dim=1, largest=True, sorted=True)
    body_threshold = top_values[:, descending_rank - 1].contiguous()
    tail_threshold = top_values[:, 0].contiguous()
    tail_indices = top_indices[:, :R]
    tail_indices, _ = torch.sort(tail_indices, dim=1)
    return tail_indices.to(torch.int32).contiguous(), body_threshold, tail_threshold


def unsorted_topk_kthvalue(A, R, activation_percentile):
    M, K = A.shape
    percentile = min(max(float(activation_percentile), 0.0), 100.0)
    body_len = K - R
    body_kth = min(K, max(1, int(math.ceil(body_len * percentile / 100.0))))
    abs_A = A.abs().float()
    tail_values, tail_indices = torch.topk(abs_A, k=R, dim=1, largest=True, sorted=False)
    kth = torch.kthvalue(abs_A, k=body_kth, dim=1).values.contiguous()
    tail_threshold = tail_values.max(dim=1).values.contiguous()
    tail_indices, _ = torch.sort(tail_indices, dim=1)
    return tail_indices.to(torch.int32).contiguous(), kth, tail_threshold


def unsorted_selectk_then_min(A, R, activation_percentile):
    M, K = A.shape
    percentile = min(max(float(activation_percentile), 0.0), 100.0)
    body_len = K - R
    body_kth = min(K, max(1, int(math.ceil(body_len * percentile / 100.0))))
    descending_rank = K - body_kth + 1
    select_k = max(R, descending_rank)
    abs_A = A.abs().float()
    top_values, top_indices = torch.topk(abs_A, k=select_k, dim=1, largest=True, sorted=False)
    tail_values2, tail_pos = torch.topk(top_values, k=R, dim=1, largest=True, sorted=False)
    tail_indices = torch.gather(top_indices, 1, tail_pos)
    body_threshold = top_values.min(dim=1).values.contiguous() if select_k == descending_rank else torch.topk(top_values, k=descending_rank, dim=1, largest=True, sorted=True).values[:, -1].contiguous()
    tail_threshold = tail_values2.max(dim=1).values.contiguous()
    tail_indices, _ = torch.sort(tail_indices, dim=1)
    return tail_indices.to(torch.int32).contiguous(), body_threshold, tail_threshold


def compare(a, b):
    return {
        "idx_equal": bool(torch.equal(a[0], b[0])),
        "body_max_abs": float((a[1] - b[1]).abs().max().item()),
        "tail_max_abs": float((a[2] - b[2]).abs().max().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, default=2048)
    ap.add_argument('--K', type=int, default=4096)
    ap.add_argument('--Rs', default='41,82,164')
    ap.add_argument('--percentiles', default='98.3,99.477')
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_dir', required=True)
    args=ap.parse_args()
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device=torch.device(args.device); torch.cuda.set_device(device)
    A=torch.randn(args.M,args.K,device=device,dtype=torch.float16)
    rows=[]
    algos=[('current_sorted', current_select), ('unsorted_topk_kthvalue', unsorted_topk_kthvalue), ('unsorted_selectk_then_min', unsorted_selectk_then_min)]
    for R in [int(x) for x in args.Rs.split(',') if x.strip()]:
        for pct in [float(x) for x in args.percentiles.split(',') if x.strip()]:
            ref_ms, ref_out = bench_ms(lambda: current_select(A,R,pct), args.warmup, args.iters, device)
            print('[REF]', R, pct, ref_ms, flush=True)
            for name, fn in algos:
                ms, out = bench_ms(lambda fn=fn: fn(A,R,pct), args.warmup, args.iters, device)
                diff = compare(ref_out, out)
                row={"M":args.M,"K":args.K,"R":R,"activation_percentile":pct,"algo":name,"ms":ms,"speedup_over_current":ref_ms/ms if ms>0 else 0.0, **diff}
                rows.append(row)
                print('[ROW]', json.dumps(row), flush=True)
    csv_path=out_dir/'topr_select_algos_v38.csv'
    fields=sorted({k for r in rows for k in r})
    with open(csv_path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(out_dir/'topr_select_algos_v38.json','w'), indent=2)
    print('[CSV]', csv_path, flush=True)

if __name__ == '__main__':
    main()
