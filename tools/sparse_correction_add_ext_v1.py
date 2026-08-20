"""Standalone sparse-correction add epilogue for QFactory dense output smoke tests."""
from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

static __device__ __forceinline__ int decode_s4(uint8_t nib) {
  int v = static_cast<int>(nib & 0x0f);
  return (v >= 8) ? (v - 16) : v;
}

template <typename OutT>
struct OutOps;

template <>
struct OutOps<half> {
  static __device__ __forceinline__ float load(const half* ptr) {
    return __half2float(*ptr);
  }
  static __device__ __forceinline__ void store(half* ptr, float value) {
    *ptr = __float2half_rn(value);
  }
};

template <>
struct OutOps<__nv_bfloat16> {
  static __device__ __forceinline__ float load(const __nv_bfloat16* ptr) {
    return __bfloat162float(*ptr);
  }
  static __device__ __forceinline__ void store(__nv_bfloat16* ptr, float value) {
    *ptr = __float2bfloat16_rn(value);
  }
};

template <typename WScaleT>
struct ScaleOps;

template <>
struct ScaleOps<float> {
  static __device__ __forceinline__ float load(const float* ptr) { return *ptr; }
};

template <>
struct ScaleOps<__nv_bfloat16> {
  static __device__ __forceinline__ float load(const __nv_bfloat16* ptr) { return __bfloat162float(*ptr); }
};

template<int COLS_PER_THREAD, int BLOCK_THREADS, typename OutT, typename WScaleT>
__global__ void sparse_correction_add_rowmajor_kernel(
    OutT* __restrict__ out,
    const int8_t* __restrict__ top_q,
    const int32_t* __restrict__ idx,
    const uint8_t* __restrict__ B_row_pack,
    const float* __restrict__ top_scale,
    const WScaleT* __restrict__ w_scale,
    int M,
    int N,
    int R) {
  int m = blockIdx.y;
  int tile_col = blockIdx.x * BLOCK_THREADS + threadIdx.x;
  int n0 = tile_col * COLS_PER_THREAD;
  if (m >= M || n0 >= N) return;

  extern __shared__ unsigned char smem_raw[];
  int32_t* s_idx = reinterpret_cast<int32_t*>(smem_raw);
  int8_t* s_top = reinterpret_cast<int8_t*>(s_idx + R);

  const int8_t* Trow = top_q + static_cast<long long>(m) * R;
  const int32_t* Irow = idx + static_cast<long long>(m) * R;
  for (int r = threadIdx.x; r < R; r += BLOCK_THREADS) {
    s_idx[r] = Irow[r];
    s_top[r] = Trow[r];
  }
  __syncthreads();

  int acc[COLS_PER_THREAD];
  #pragma unroll
  for (int i = 0; i < COLS_PER_THREAD; ++i) acc[i] = 0;

  #pragma unroll 1
  for (int r = 0; r < R; ++r) {
    int k = s_idx[r];
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
      float sparse = static_cast<float>(acc[i]) * ts * ScaleOps<WScaleT>::load(w_scale + n0 + i);
      OutT* ptr = out + off0 + i;
      OutOps<OutT>::store(ptr, OutOps<OutT>::load(ptr) + sparse);
    }
  }
}

template<int COLS_PER_THREAD, typename OutT, typename WScaleT>
void launch_sparse_correction_add(
    torch::Tensor out,
    torch::Tensor top_q,
    torch::Tensor idx,
    torch::Tensor B_row_pack,
    torch::Tensor top_scale,
    torch::Tensor w_scale) {
  constexpr int BLOCK_THREADS = 128;
  int M = static_cast<int>(top_q.size(0));
  int R = static_cast<int>(top_q.size(1));
  int N = static_cast<int>(out.size(1));
  dim3 block(BLOCK_THREADS);
  dim3 grid(((N + COLS_PER_THREAD - 1) / COLS_PER_THREAD + BLOCK_THREADS - 1) / BLOCK_THREADS, M);
  size_t smem = static_cast<size_t>(R) * sizeof(int32_t) + static_cast<size_t>(R) * sizeof(int8_t);
  sparse_correction_add_rowmajor_kernel<COLS_PER_THREAD, BLOCK_THREADS, OutT, WScaleT><<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<OutT*>(out.data_ptr()),
      top_q.data_ptr<int8_t>(),
      idx.data_ptr<int32_t>(),
      B_row_pack.data_ptr<uint8_t>(),
      top_scale.data_ptr<float>(),
      reinterpret_cast<WScaleT*>(w_scale.data_ptr()),
      M,
      N,
      R);
}

static void check_common(
    torch::Tensor out,
    torch::Tensor top_q,
    torch::Tensor idx,
    torch::Tensor B_row_pack,
    torch::Tensor top_scale,
    torch::Tensor w_scale) {
  CHECK_CUDA(out); CHECK_CUDA(top_q); CHECK_CUDA(idx); CHECK_CUDA(B_row_pack); CHECK_CUDA(top_scale); CHECK_CUDA(w_scale);
  CHECK_CONTIGUOUS(out); CHECK_CONTIGUOUS(top_q); CHECK_CONTIGUOUS(idx); CHECK_CONTIGUOUS(B_row_pack); CHECK_CONTIGUOUS(top_scale); CHECK_CONTIGUOUS(w_scale);
  TORCH_CHECK(out.dim() == 2, "out must be [M,N]");
  TORCH_CHECK(top_q.dim() == 2, "top_q must be [M,R]");
  TORCH_CHECK(idx.sizes() == top_q.sizes(), "idx shape must match top_q");
  TORCH_CHECK(B_row_pack.dim() == 2, "B_row_pack must be [K,N/2]");
  TORCH_CHECK(top_q.size(0) == out.size(0), "top_q/out M mismatch");
  TORCH_CHECK(B_row_pack.size(1) * 2 == out.size(1), "B_row_pack/out N mismatch");
  TORCH_CHECK(top_scale.numel() == out.size(0), "top_scale must have M elements");
  TORCH_CHECK(w_scale.numel() == out.size(1), "w_scale must have N elements");
  TORCH_CHECK(top_q.scalar_type() == at::ScalarType::Char, "top_q must be int8");
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Int, "idx must be int32");
  TORCH_CHECK(B_row_pack.scalar_type() == at::ScalarType::Byte, "B_row_pack must be uint8");
  TORCH_CHECK(top_scale.scalar_type() == at::ScalarType::Float, "top_scale must be fp32");
}

void sparse_correction_add_bf16_oct(
    torch::Tensor out,
    torch::Tensor top_q,
    torch::Tensor idx,
    torch::Tensor B_row_pack,
    torch::Tensor top_scale,
    torch::Tensor w_scale) {
  check_common(out, top_q, idx, B_row_pack, top_scale, w_scale);
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bf16");
  TORCH_CHECK(w_scale.scalar_type() == at::ScalarType::Float, "w_scale must be fp32");
  launch_sparse_correction_add<8, __nv_bfloat16, float>(out, top_q, idx, B_row_pack, top_scale, w_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_correction_add_bf16_wscale_bf16_oct(
    torch::Tensor out,
    torch::Tensor top_q,
    torch::Tensor idx,
    torch::Tensor B_row_pack,
    torch::Tensor top_scale,
    torch::Tensor w_scale) {
  check_common(out, top_q, idx, B_row_pack, top_scale, w_scale);
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "out must be bf16");
  TORCH_CHECK(w_scale.scalar_type() == at::ScalarType::BFloat16, "w_scale must be bf16");
  launch_sparse_correction_add<8, __nv_bfloat16, __nv_bfloat16>(out, top_q, idx, B_row_pack, top_scale, w_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_correction_add_fp16_oct(
    torch::Tensor out,
    torch::Tensor top_q,
    torch::Tensor idx,
    torch::Tensor B_row_pack,
    torch::Tensor top_scale,
    torch::Tensor w_scale) {
  check_common(out, top_q, idx, B_row_pack, top_scale, w_scale);
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half, "out must be fp16");
  TORCH_CHECK(w_scale.scalar_type() == at::ScalarType::Float, "w_scale must be fp32");
  launch_sparse_correction_add<8, half, float>(out, top_q, idx, B_row_pack, top_scale, w_scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_correction_add_bf16_oct", &sparse_correction_add_bf16_oct,
        "Add split sparse correction into bf16 dense output, 8 cols/thread");
  m.def("sparse_correction_add_bf16_wscale_bf16_oct", &sparse_correction_add_bf16_wscale_bf16_oct,
        "Add split sparse correction into bf16 dense output with bf16 w_scale, 8 cols/thread");
  m.def("sparse_correction_add_fp16_oct", &sparse_correction_add_fp16_oct,
        "Add split sparse correction into fp16 dense output, 8 cols/thread");
}
"""


def load_sparse_correction_add_ext_v1(verbose: bool = False):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return load_inline(
        name="sparse_correction_add_ext_v1",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(load_sparse_correction_add_ext_v1(verbose=True))
