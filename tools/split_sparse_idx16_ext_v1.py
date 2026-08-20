"""Experimental idx16 sparse-correction kernels for split optimization smoke."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
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

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

static __device__ __forceinline__ int decode_s4(uint8_t nib) {
  int v = static_cast<int>(nib & 0x0f);
  return (v >= 8) ? (v - 16) : v;
}

__global__ void idx_i32_to_u16_kernel(const int32_t* __restrict__ src, uint16_t* __restrict__ dst, long long count) {
  long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) dst[i] = static_cast<uint16_t>(src[i]);
}

void idx_i32_to_u16(torch::Tensor src, torch::Tensor dst) {
  CHECK_CUDA(src); CHECK_CUDA(dst);
  CHECK_CONTIGUOUS(src); CHECK_CONTIGUOUS(dst);
  TORCH_CHECK(src.scalar_type() == at::ScalarType::Int, "src must be int32");
  TORCH_CHECK(dst.scalar_type() == at::ScalarType::Short, "dst must be int16/uint16 storage");
  TORCH_CHECK(src.numel() == dst.numel(), "src/dst numel mismatch");
  long long count = src.numel();
  int block = 256;
  int grid = static_cast<int>((count + block - 1) / block);
  idx_i32_to_u16_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      src.data_ptr<int32_t>(), reinterpret_cast<uint16_t*>(dst.data_ptr<int16_t>()), count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int COLS_PER_THREAD, int BLOCK_THREADS, typename IdxT>
__global__ void sparse_top_write_idx_kernel(
    const int8_t* __restrict__ top_q,
    const IdxT* __restrict__ idx,
    const uint8_t* __restrict__ B_row_pack,
    const float* __restrict__ top_scale,
    const float* __restrict__ w_scale,
    half* __restrict__ out,
    int M,
    int N,
    int R) {
  int m = blockIdx.y;
  int tile_col = blockIdx.x * BLOCK_THREADS + threadIdx.x;
  int n0 = tile_col * COLS_PER_THREAD;

  extern __shared__ unsigned char smem_raw[];
  IdxT* s_idx = reinterpret_cast<IdxT*>(smem_raw);
  int8_t* s_top = reinterpret_cast<int8_t*>(s_idx + R);

  const int8_t* Trow = top_q + static_cast<long long>(m) * R;
  const IdxT* Irow = idx + static_cast<long long>(m) * R;
  for (int r = threadIdx.x; r < R; r += BLOCK_THREADS) {
    s_idx[r] = Irow[r];
    s_top[r] = Trow[r];
  }
  __syncthreads();

  if (m >= M || n0 >= N) return;

  int acc[COLS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < COLS_PER_THREAD; ++i) acc[i] = 0;

  #pragma unroll 1
  for (int r = 0; r < R; ++r) {
    int k = static_cast<int>(s_idx[r]);
    int aq = static_cast<int>(s_top[r]);
    long long elem0 = static_cast<long long>(k) * N + n0;
    #pragma unroll
    for (int i = 0; i < COLS_PER_THREAD; i += 2) {
      if (n0 + i < N) {
        uint8_t packed = B_row_pack[(elem0 + i) >> 1];
        acc[i] += aq * decode_s4(packed & 0x0f);
        if (i + 1 < COLS_PER_THREAD && n0 + i + 1 < N) {
          acc[i + 1] += aq * decode_s4(packed >> 4);
        }
      }
    }
  }

  float ts = top_scale[m];
  long long off0 = static_cast<long long>(m) * N + n0;
  #pragma unroll
  for (int i = 0; i < COLS_PER_THREAD; ++i) {
    if (n0 + i < N) {
      float value = static_cast<float>(acc[i]) * ts * w_scale[n0 + i];
      out[off0 + i] = __float2half_rn(value);
    }
  }
}

template<int COLS_PER_THREAD, int BLOCK_THREADS, typename IdxT>
static void launch_sparse_top_write(torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale, torch::Tensor out) {
  CHECK_CUDA(top_q); CHECK_CUDA(idx); CHECK_CUDA(B_row_pack); CHECK_CUDA(top_scale); CHECK_CUDA(w_scale); CHECK_CUDA(out);
  CHECK_CONTIGUOUS(top_q); CHECK_CONTIGUOUS(idx); CHECK_CONTIGUOUS(B_row_pack); CHECK_CONTIGUOUS(top_scale); CHECK_CONTIGUOUS(w_scale); CHECK_CONTIGUOUS(out);
  TORCH_CHECK(top_q.scalar_type() == at::ScalarType::Char, "top_q must be int8");
  TORCH_CHECK(B_row_pack.scalar_type() == at::ScalarType::Byte, "B_row_pack must be uint8");
  TORCH_CHECK(top_scale.scalar_type() == at::ScalarType::Float, "top_scale must be fp32");
  TORCH_CHECK(w_scale.scalar_type() == at::ScalarType::Float, "w_scale must be fp32");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half, "out must be fp16");
  int M = static_cast<int>(top_q.size(0));
  int R = static_cast<int>(top_q.size(1));
  int N = static_cast<int>(out.size(1));
  dim3 block(BLOCK_THREADS);
  dim3 grid(((N + COLS_PER_THREAD - 1) / COLS_PER_THREAD + BLOCK_THREADS - 1) / BLOCK_THREADS, M);
  size_t smem = static_cast<size_t>(R) * sizeof(IdxT) + static_cast<size_t>(R) * sizeof(int8_t);
  sparse_top_write_idx_kernel<COLS_PER_THREAD, BLOCK_THREADS, IdxT><<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
      top_q.data_ptr<int8_t>(), reinterpret_cast<const IdxT*>(idx.data_ptr()), B_row_pack.data_ptr<uint8_t>(),
      top_scale.data_ptr<float>(), w_scale.data_ptr<float>(), reinterpret_cast<half*>(out.data_ptr<at::Half>()), M, N, R);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define DEF_I32(NAME, CPT, BT) \
void NAME(torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale, torch::Tensor out) { \
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Int, "idx must be int32"); \
  launch_sparse_top_write<CPT, BT, int32_t>(top_q, idx, B_row_pack, top_scale, w_scale, out); \
}

#define DEF_U16(NAME, CPT, BT) \
void NAME(torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale, torch::Tensor out) { \
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Short, "idx must be int16/uint16 storage"); \
  launch_sparse_top_write<CPT, BT, uint16_t>(top_q, idx, B_row_pack, top_scale, w_scale, out); \
}

DEF_I32(sparse_i32_c4_b128, 4, 128)
DEF_I32(sparse_i32_c8_b128, 8, 128)
DEF_I32(sparse_i32_c8_b256, 8, 256)
DEF_I32(sparse_i32_c16_b128, 16, 128)
DEF_U16(sparse_u16_c4_b128, 4, 128)
DEF_U16(sparse_u16_c8_b128, 8, 128)
DEF_U16(sparse_u16_c8_b256, 8, 256)
DEF_U16(sparse_u16_c16_b128, 16, 128)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("idx_i32_to_u16", &idx_i32_to_u16, "Compress idx int32 to uint16 storage");
  m.def("sparse_i32_c4_b128", &sparse_i32_c4_b128, "sparse write int32 idx, 4 cols/thread, 128 threads");
  m.def("sparse_i32_c8_b128", &sparse_i32_c8_b128, "sparse write int32 idx, 8 cols/thread, 128 threads");
  m.def("sparse_i32_c8_b256", &sparse_i32_c8_b256, "sparse write int32 idx, 8 cols/thread, 256 threads");
  m.def("sparse_i32_c16_b128", &sparse_i32_c16_b128, "sparse write int32 idx, 16 cols/thread, 128 threads");
  m.def("sparse_u16_c4_b128", &sparse_u16_c4_b128, "sparse write uint16 idx, 4 cols/thread, 128 threads");
  m.def("sparse_u16_c8_b128", &sparse_u16_c8_b128, "sparse write uint16 idx, 8 cols/thread, 128 threads");
  m.def("sparse_u16_c8_b256", &sparse_u16_c8_b256, "sparse write uint16 idx, 8 cols/thread, 256 threads");
  m.def("sparse_u16_c16_b128", &sparse_u16_c16_b128, "sparse write uint16 idx, 16 cols/thread, 128 threads");
}
"""


def load_split_sparse_idx16_ext_v1(verbose: bool = False):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return load_inline(
        name="split_sparse_idx16_ext_v1",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(load_split_sparse_idx16_ext_v1(verbose=True))
