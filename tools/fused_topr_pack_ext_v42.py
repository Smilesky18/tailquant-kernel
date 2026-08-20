
import os

import torch
from torch.utils.cpp_extension import load_inline


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <stdint.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Half, #x " must be fp16")

__device__ __forceinline__ int clamp_s4(float x) {
  int q = __float2int_rn(x);
  q = q < -8 ? -8 : q;
  q = q > 7 ? 7 : q;
  return q;
}

__device__ __forceinline__ uint8_t encode_s4(int q) {
  return static_cast<uint8_t>(q & 0x0f);
}

template<int K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void fused_topr_pack_cub_kernel(
    const half* __restrict__ A,
    int R,
    int descending_rank,
    uint8_t* __restrict__ A_pack,
    float* __restrict__ body_scale,
    float* __restrict__ top_scale,
    int8_t* __restrict__ top_q,
    int32_t* __restrict__ idx,
    int M,
    float eps) {
  int m = blockIdx.x;
  if (m >= M) return;

  using BlockSort = cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int>;
  __shared__ typename BlockSort::TempStorage sort_storage;
  __shared__ uint32_t bits[(K + 31) / 32];
  __shared__ float s_body_scale;
  __shared__ float s_top_scale;

  float keys[ITEMS_PER_THREAD];
  int vals[ITEMS_PER_THREAD];
  const half* Arow = A + static_cast<long long>(m) * K;

  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int k = threadIdx.x * ITEMS_PER_THREAD + item;
    keys[item] = fabsf(__half2float(Arow[k]));
    vals[item] = k;
  }

  BlockSort(sort_storage).SortDescending(keys, vals);
  __syncthreads();

  for (int w = threadIdx.x; w < (K + 31) / 32; w += blockDim.x) bits[w] = 0u;
  __syncthreads();

  int32_t* Irow = idx + static_cast<long long>(m) * R;
  int8_t* Trow = top_q + static_cast<long long>(m) * R;

  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int rank = threadIdx.x * ITEMS_PER_THREAD + item;
    if (rank == 0) {
      s_top_scale = fmaxf(keys[item], eps) / 7.0f;
      top_scale[m] = s_top_scale;
    }
    if (rank == descending_rank - 1) {
      s_body_scale = fmaxf(keys[item], eps) / 7.0f;
      body_scale[m] = s_body_scale;
    }
  }
  __syncthreads();

  #pragma unroll
  for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
    int rank = threadIdx.x * ITEMS_PER_THREAD + item;
    if (rank < R) {
      int k = vals[item];
      Irow[rank] = k;
      atomicOr(&bits[k >> 5], 1u << (k & 31));
      Trow[rank] = static_cast<int8_t>(clamp_s4(__half2float(Arow[k]) / s_top_scale));
    }
  }
  __syncthreads();

  for (int k2 = threadIdx.x * 2; k2 < K; k2 += blockDim.x * 2) {
    bool top0 = (bits[k2 >> 5] >> (k2 & 31)) & 1u;
    int q0 = top0 ? 0 : clamp_s4(__half2float(Arow[k2]) / s_body_scale);
    int q1 = 0;
    if (k2 + 1 < K) {
      bool top1 = (bits[(k2 + 1) >> 5] >> ((k2 + 1) & 31)) & 1u;
      q1 = top1 ? 0 : clamp_s4(__half2float(Arow[k2 + 1]) / s_body_scale);
    }
    long long elem0 = static_cast<long long>(m) * K + k2;
    A_pack[elem0 >> 1] = static_cast<uint8_t>(encode_s4(q0) | (encode_s4(q1) << 4));
  }
}

void fused_topr_pack(
    torch::Tensor A,
    int64_t R64,
    int64_t descending_rank64,
    torch::Tensor A_pack,
    torch::Tensor body_scale,
    torch::Tensor top_scale,
    torch::Tensor top_q,
    torch::Tensor idx,
    double eps64) {
  CHECK_CUDA(A); CHECK_CUDA(A_pack); CHECK_CUDA(body_scale); CHECK_CUDA(top_scale); CHECK_CUDA(top_q); CHECK_CUDA(idx);
  CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(A_pack); CHECK_CONTIGUOUS(body_scale); CHECK_CONTIGUOUS(top_scale); CHECK_CONTIGUOUS(top_q); CHECK_CONTIGUOUS(idx);
  CHECK_HALF(A);
  TORCH_CHECK(A_pack.scalar_type() == at::ScalarType::Byte, "A_pack must be uint8");
  TORCH_CHECK(body_scale.scalar_type() == at::ScalarType::Float, "body_scale must be fp32");
  TORCH_CHECK(top_scale.scalar_type() == at::ScalarType::Float, "top_scale must be fp32");
  TORCH_CHECK(top_q.scalar_type() == at::ScalarType::Char, "top_q must be int8");
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Int, "idx must be int32");
  int M = static_cast<int>(A.size(0));
  int K = static_cast<int>(A.size(1));
  int R = static_cast<int>(R64);
  int descending_rank = static_cast<int>(descending_rank64);
  TORCH_CHECK(K == 4096, "fused_topr_pack v42 currently supports K=4096 only");
  TORCH_CHECK(R > 0 && R <= 256 && R < K, "fused_topr_pack v42 supports 0<R<=256");
  TORCH_CHECK(descending_rank >= 1 && descending_rank <= K, "bad descending_rank");
  TORCH_CHECK(idx.size(0) == M && idx.size(1) == R, "idx shape mismatch");
  TORCH_CHECK(top_q.size(0) == M && top_q.size(1) == R, "top_q shape mismatch");
  fused_topr_pack_cub_kernel<4096, 256, 16><<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
      R,
      descending_rank,
      A_pack.data_ptr<uint8_t>(),
      body_scale.data_ptr<float>(),
      top_scale.data_ptr<float>(),
      top_q.data_ptr<int8_t>(),
      idx.data_ptr<int32_t>(),
      M,
      static_cast<float>(eps64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_topr_pack", &fused_topr_pack, "CUB fused top-r selection and split activation packing v42");
}
"""


def load_fused_topr_pack_ext(verbose: bool = False):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return load_inline(
        name="fused_topr_pack_v42_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(load_fused_topr_pack_ext(verbose=True))
