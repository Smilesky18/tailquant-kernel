#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experimental tuned sparse correction write kernels for split layer latency v21.

This file lives only in the experiment directory. It does not modify original
project sources. The main new variant is oct_b256: each block covers 2048 output
columns instead of 1024, reducing repeated shared-memory staging of top-R data
for large-N projections.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from torch.utils.cpp_extension import load_inline

ENV_BIN = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = ENV_BIN + os.pathsep + os.environ.get("PATH", "")

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif

static __device__ __forceinline__ int decode_s4(uint8_t nib) {
  int v = static_cast<int>(nib & 0x0f);
  return (v >= 8) ? (v - 16) : v;
}

template<int BLOCK_THREADS>
__global__ void sparse_top_write_rowmajor_oct_shared_kernel(
    const int8_t* __restrict__ top_q,
    const int32_t* __restrict__ idx,
    const uint8_t* __restrict__ B_row_pack,
    const float* __restrict__ top_scale,
    const float* __restrict__ w_scale,
    half* __restrict__ out,
    int M,
    int K,
    int N,
    int R) {
  int m = blockIdx.y;
  int ocol = blockIdx.x * BLOCK_THREADS + threadIdx.x;
  int n0 = ocol * 8;
  bool active = (m < M && n0 < N);

  extern __shared__ unsigned char smem_raw[];
  int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
  int8_t* s_top = reinterpret_cast<int8_t*>(s_idx + R);

  const int8_t* Trow = top_q + (long long)m * R;
  const int32_t* Irow = idx + (long long)m * R;
  for (int r = threadIdx.x; r < R; r += BLOCK_THREADS) {
    s_idx[r] = Irow[r];
    s_top[r] = Trow[r];
  }
  __syncthreads();

  if (!active) return;

  int acc0=0, acc1=0, acc2=0, acc3=0, acc4=0, acc5=0, acc6=0, acc7=0;
  #pragma unroll 1
  for (int r = 0; r < R; ++r) {
    int k = s_idx[r];
    int aq = static_cast<int>(s_top[r]);
    long long elem0 = (long long)k * N + n0;
    uint8_t b01 = B_row_pack[elem0 >> 1];
    acc0 += aq * decode_s4(b01 & 0x0f);
    acc1 += aq * decode_s4(b01 >> 4);
    if (n0 + 2 < N) {
      uint8_t b23 = B_row_pack[(elem0 + 2) >> 1];
      acc2 += aq * decode_s4(b23 & 0x0f);
      acc3 += aq * decode_s4(b23 >> 4);
    }
    if (n0 + 4 < N) {
      uint8_t b45 = B_row_pack[(elem0 + 4) >> 1];
      acc4 += aq * decode_s4(b45 & 0x0f);
      acc5 += aq * decode_s4(b45 >> 4);
    }
    if (n0 + 6 < N) {
      uint8_t b67 = B_row_pack[(elem0 + 6) >> 1];
      acc6 += aq * decode_s4(b67 & 0x0f);
      acc7 += aq * decode_s4(b67 >> 4);
    }
  }

  float ts = top_scale[m];
  long long off0 = (long long)m * N + n0;
#define WRITE_ACC(II, ACC) \
  if (n0 + (II) < N) { \
    out[off0 + (II)] = __float2half_rn(static_cast<float>(ACC) * ts * w_scale[n0 + (II)]); \
  }
  WRITE_ACC(0, acc0);
  WRITE_ACC(1, acc1);
  WRITE_ACC(2, acc2);
  WRITE_ACC(3, acc3);
  WRITE_ACC(4, acc4);
  WRITE_ACC(5, acc5);
  WRITE_ACC(6, acc6);
  WRITE_ACC(7, acc7);
#undef WRITE_ACC
}

template<int BLOCK_THREADS>
static void launch_sparse_top_write_oct(torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale, torch::Tensor out, int K) {
  CHECK_CUDA(top_q); CHECK_CUDA(idx); CHECK_CUDA(B_row_pack); CHECK_CUDA(top_scale); CHECK_CUDA(w_scale); CHECK_CUDA(out);
  CHECK_CONTIGUOUS(top_q); CHECK_CONTIGUOUS(idx); CHECK_CONTIGUOUS(B_row_pack); CHECK_CONTIGUOUS(top_scale); CHECK_CONTIGUOUS(w_scale); CHECK_CONTIGUOUS(out);
  int M = top_q.size(0), R = top_q.size(1), N = out.size(1);
  dim3 block(BLOCK_THREADS);
  dim3 grid(((N + 7) / 8 + BLOCK_THREADS - 1) / BLOCK_THREADS, M);
  size_t smem = (size_t)R * sizeof(int32_t) + (size_t)R * sizeof(int8_t);
  sparse_top_write_rowmajor_oct_shared_kernel<BLOCK_THREADS><<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
      top_q.data_ptr<int8_t>(), idx.data_ptr<int32_t>(), B_row_pack.data_ptr<uint8_t>(), top_scale.data_ptr<float>(), w_scale.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()), M, K, N, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_top_write_rowmajor_oct_b128_shared(torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale, torch::Tensor out, int K) {
  launch_sparse_top_write_oct<128>(top_q, idx, B_row_pack, top_scale, w_scale, out, K);
}

void sparse_top_write_rowmajor_oct_b256_shared(torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale, torch::Tensor out, int K) {
  launch_sparse_top_write_oct<256>(top_q, idx, B_row_pack, top_scale, w_scale, out, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_top_write_rowmajor_oct_b128_shared", &sparse_top_write_rowmajor_oct_b128_shared, "write-only split sparse top correction, oct columns, 128 threads");
  m.def("sparse_top_write_rowmajor_oct_b256_shared", &sparse_top_write_rowmajor_oct_b256_shared, "write-only split sparse top correction, oct columns, 256 threads");
}
"""


def load_sparse_tuned_ext(verbose: bool = False):
    return load_inline(
        name="split_sparse_tuned_v21_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )
