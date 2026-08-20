#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QFactory A4W4 dense-scale GEMM with split sparse correction in the store path.

This module generates an experimental header from RoMeO/QFactory's
gemm_a4w4_perchannel_sm80.h without modifying RoMeO files. The prototype keeps
QFactory's dense accumulator store into the shared-memory C tile, then adds the
split sparse correction to that shared tile immediately before the single global
store.

Limitations:
  - a4w4 only
  - bf16 dense output/scales, matching the public QFactory per-channel kernel
  - R is compile-time for each JIT specialization
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
ROMEO_ROOT = Path(os.environ.get("ROMEO_ROOT", "/data/yzy/RoMeo")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for item in (TOOLS, ROMEO_ROOT, ROOT):
    sp = str(item)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from qfactory.jit import jit  # noqa: E402
from qfactory.kernels.utils import span_tuning_space  # noqa: E402
import qfactory.kernels.gemm_w4a4_mixed_precision as qg  # noqa: E402


ORIGINAL_HEADER = ROMEO_ROOT / "qfactory/include/cuda/gemm/mixed_precision/gemm_a4w4_perchannel_sm80.h"
GENERATED_HEADER = TOOLS / "qfactory_split_fused_epilogue_a4w4_perchannel_sm80_v1.h"


def _patch_header_text(text: str) -> str:
    text = text.replace(
        "int TileM, int TileN, int TileK, int NStage, int WarpM, int WarpN, int WarpK,\n    typename KTraits, typename SharedStorage",
        "int TileM, int TileN, int TileK, int NStage, int WarpM, int WarpN, int WarpK, int SparseR,\n    typename KTraits, typename SharedStorage",
    )
    text = text.replace(
        "int TileM, int TileN, int TileK, int NStage,\n    int WarpM, int WarpN, int WarpK\n>",
        "int TileM, int TileN, int TileK, int NStage,\n    int WarpM, int WarpN, int WarpK, int SparseR\n>",
    )
    text = text.replace(
        "using namespace cute;\n",
        """using namespace cute;

static __device__ __forceinline__ int decode_s4_split_v1(uint8_t nib) {
    int v = static_cast<int>(nib & 0x0f);
    return (v >= 8) ? (v - 16) : v;
}

""",
    )
    text = text.replace(
        "uint8_t *A, uint8_t *B, __nv_bfloat16 *C,\n    __nv_bfloat16 *A_scale, __nv_bfloat16 *B_scale\n) {",
        """uint8_t *A, uint8_t *B, __nv_bfloat16 *C,
    __nv_bfloat16 *A_scale, __nv_bfloat16 *B_scale,
    int8_t *Top_q, int32_t *Idx_sparse, uint8_t *B_row_pack,
    float *Top_scale
) {""",
    )
    insertion = r"""
            __syncthreads();
            for (int linear = threadIdx.x; linear < WarpM * WarpN; linear += blockDim.x) {
                int local_m = linear / WarpN;
                int local_n = linear - local_m * WarpN;
                int global_m = idx * TileM + i * WarpM + local_m;
                int global_n = idy * TileN + j * WarpN + local_n;
                if (global_m < M && global_n < N) {
                    int sparse_acc = 0;
                    int8_t* top_row = Top_q + static_cast<long long>(global_m) * SparseR;
                    int32_t* idx_row = Idx_sparse + static_cast<long long>(global_m) * SparseR;
                    for (int r = 0; r < SparseR; ++r) {
                        int k_sparse = idx_row[r];
                        int aq = static_cast<int>(top_row[r]);
                        long long packed_index = (static_cast<long long>(k_sparse) * N + global_n) >> 1;
                        uint8_t packed = B_row_pack[packed_index];
                        int bq = (global_n & 1) ? decode_s4_split_v1(packed >> 4) : decode_s4_split_v1(packed & 0x0f);
                        sparse_acc += aq * bq;
                    }
                    float dense_value = __bfloat162float(sC(local_m, local_n));
                    float sparse_value = static_cast<float>(sparse_acc)
                        * Top_scale[global_m]
                        * __bfloat162float(B_scale[global_n]);
                    sC(local_m, local_n) = __float2bfloat16(dense_value + sparse_value);
                }
            }
"""
    text = text.replace(
        "            __syncthreads();\n            copy(copyC, sC_s2g_thread, gC_tile_s2g_thread(_, _, _, i, j));",
        insertion + "            __syncthreads();\n            copy(copyC, sC_s2g_thread, gC_tile_s2g_thread(_, _, _, i, j));",
    )
    text = text.replace(
        "uint8_t *A, uint8_t *B, __nv_bfloat16 *C,\n    __nv_bfloat16 *A_scale, __nv_bfloat16 *B_scale,\n    cudaStream_t stream\n) {",
        """uint8_t *A, uint8_t *B, __nv_bfloat16 *C,
    __nv_bfloat16 *A_scale, __nv_bfloat16 *B_scale,
    int8_t *Top_q, int32_t *Idx_sparse, uint8_t *B_row_pack,
    float *Top_scale,
    cudaStream_t stream
) {""",
    )
    text = text.replace(
        "        A_scale,\n        B_scale\n    );",
        """        A_scale,
        B_scale,
        Top_q,
        Idx_sparse,
        B_row_pack,
        Top_scale
    );""",
    )
    text = text.replace(
        "N, K, LDC,\n            TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK,\n            KTraits, sharedStorage",
        "N, K, LDC,\n            TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK, SparseR,\n            KTraits, sharedStorage",
    )
    text = text.replace(
        "N, K, LDC,\n        TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK,\n        KTraits, sharedStorage",
        "N, K, LDC,\n        TileM, TileN, TileK, NStage, WarpM, WarpN, WarpK, SparseR,\n        KTraits, sharedStorage",
    )
    return text


def ensure_generated_header() -> Path:
    source = ORIGINAL_HEADER.read_text(encoding="utf-8")
    patched = _patch_header_text(source)
    GENERATED_HEADER.write_text(patched, encoding="utf-8")
    return GENERATED_HEADER


def install_one_config_qfactory():
    def one_config():
        return {
            "NStage": [2],
            "TileM": [128],
            "TileN": [128],
            "TileK": [128],
            "WarpM": [64],
            "WarpN": [64],
            "WarpK": [128],
        }

    qg.generate_tunable_keys = one_config


def gemm_a4w4_split_fused(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    top_q: torch.Tensor,
    idx_sparse: torch.Tensor,
    b_row_pack: torch.Tensor,
    top_scale: torch.Tensor,
) -> int:
    header = ensure_generated_header()
    m, n = activation.shape[0], weight.shape[0]
    k = activation.shape[1] * 2
    r = top_q.shape[1]

    assert activation.dtype == torch.uint8 and activation.is_contiguous()
    assert weight.dtype == torch.uint8 and weight.is_contiguous()
    assert activation_scale.dtype == torch.bfloat16 and activation_scale.shape == (m,)
    assert weight_scale.dtype == torch.bfloat16 and weight_scale.shape == (n,)
    assert output.dtype == torch.bfloat16 and output.shape == (m, n)
    assert top_q.dtype == torch.int8 and top_q.shape == (m, r) and top_q.is_contiguous()
    assert idx_sparse.dtype == torch.int32 and idx_sparse.shape == (m, r) and idx_sparse.is_contiguous()
    assert b_row_pack.dtype == torch.uint8 and b_row_pack.shape == (k, n // 2) and b_row_pack.is_contiguous()
    assert top_scale.dtype == torch.float32 and top_scale.shape == (m,) and top_scale.is_contiguous()

    output_stride = output.stride()
    assert len(output_stride) == 2 and output_stride[1] == 1

    perf_keys = {"M": m}
    keys = {"N": n, "K": k, "LDC": output_stride[0], "R": r}
    args = (
        (m, "m"),
        (activation, "activation"),
        (weight, "weight"),
        (output, "output"),
        (activation_scale, "activation_scale"),
        (weight_scale, "weight_scale"),
        (top_q, "top_q"),
        (idx_sparse, "idx_sparse"),
        (b_row_pack, "b_row_pack"),
        (top_scale, "top_scale"),
        (torch.cuda.current_stream(), "stream"),
    )
    includes = (str(header),)
    template = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int LDC = {LDC};
    constexpr int SparseR = {R};
    constexpr int TileM = {TileM};
    constexpr int TileN = {TileN};
    constexpr int TileK = {TileK};
    constexpr int NStage = {NStage};
    constexpr int WarpM = {WarpM};
    constexpr int WarpN = {WarpN};
    constexpr int WarpK = {WarpK};

    __return_code = call_gemm <
        N, K, LDC,
        TileM, TileN, TileK, NStage,
        WarpM, WarpN, WarpK, SparseR
    > (
        m,
        activation, weight, output,
        activation_scale, weight_scale,
        top_q, idx_sparse, b_row_pack,
        top_scale,
        stream
    );
    """
    runtime = jit.compile_and_tune(
        name="gemm_a4w4_split_fused_epilogue_v1",
        includes=includes,
        template=template,
        perf_keys=perf_keys,
        keys=keys,
        space=span_tuning_space(qg.generate_tunable_keys()),
        args=args,
    )
    return runtime.run(*(arg for arg, _ in args))


if __name__ == "__main__":
    print(ensure_generated_header())
