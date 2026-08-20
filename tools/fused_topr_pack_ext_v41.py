
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

__device__ __forceinline__ void sort_pair_asc(float &a, int &ai, float &b, int &bi) {
  if (a > b || (a == b && ai > bi)) {
    float tv = a; a = b; b = tv;
    int ti = ai; ai = bi; bi = ti;
  }
}

__global__ void fused_topr_pack_kernel(
    const half* __restrict__ A,
    int R,
    int descending_rank,
    uint8_t* __restrict__ A_pack,
    float* __restrict__ body_scale,
    float* __restrict__ top_scale,
    int8_t* __restrict__ top_q,
    int32_t* __restrict__ idx,
    int M,
    int K,
    float eps) {
  int m = blockIdx.x;
  if (m >= M) return;

  extern __shared__ unsigned char smem_raw[];
  float* vals = reinterpret_cast<float*>(smem_raw);
  int* inds = reinterpret_cast<int*>(vals + K);
  uint32_t* bits = reinterpret_cast<uint32_t*>(inds + K);
  int words = (K + 31) >> 5;

  const half* Arow = A + static_cast<long long>(m) * K;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    vals[k] = fabsf(__half2float(Arow[k]));
    inds[k] = k;
  }
  for (int w = threadIdx.x; w < words; w += blockDim.x) bits[w] = 0u;
  __syncthreads();

  for (int size = 2; size <= K; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = threadIdx.x; i < K; i += blockDim.x) {
        int ixj = i ^ stride;
        if (ixj > i) {
          float a = vals[i];
          float b = vals[ixj];
          int ai = inds[i];
          int bi = inds[ixj];
          bool ascending = ((i & size) == 0);
          if (ascending) {
            sort_pair_asc(a, ai, b, bi);
          } else {
            sort_pair_asc(b, bi, a, ai);
          }
          vals[i] = a; inds[i] = ai;
          vals[ixj] = b; inds[ixj] = bi;
        }
      }
      __syncthreads();
    }
  }

  float bs = fmaxf(vals[K - descending_rank], eps) / 7.0f;
  float ts = fmaxf(vals[K - 1], eps) / 7.0f;
  if (threadIdx.x == 0) {
    body_scale[m] = bs;
    top_scale[m] = ts;
  }

  int32_t* Irow = idx + static_cast<long long>(m) * R;
  int8_t* Trow = top_q + static_cast<long long>(m) * R;
  for (int r = threadIdx.x; r < R; r += blockDim.x) {
    int pos = K - 1 - r;
    int k = inds[pos];
    Irow[r] = k;
    atomicOr(&bits[k >> 5], 1u << (k & 31));
    Trow[r] = static_cast<int8_t>(clamp_s4(__half2float(Arow[k]) / ts));
  }
  __syncthreads();

  for (int k2 = threadIdx.x * 2; k2 < K; k2 += blockDim.x * 2) {
    bool top0 = (bits[k2 >> 5] >> (k2 & 31)) & 1u;
    int q0 = top0 ? 0 : clamp_s4(__half2float(Arow[k2]) / bs);
    int q1 = 0;
    if (k2 + 1 < K) {
      bool top1 = (bits[(k2 + 1) >> 5] >> ((k2 + 1) & 31)) & 1u;
      q1 = top1 ? 0 : clamp_s4(__half2float(Arow[k2 + 1]) / bs);
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
  TORCH_CHECK(K > 0 && (K & (K - 1)) == 0, "K must be a power of two");
  TORCH_CHECK(K <= 4096, "fused_topr_pack v41 supports K<=4096");
  TORCH_CHECK(R > 0 && R <= 256 && R < K, "fused_topr_pack v41 supports 0<R<=256");
  TORCH_CHECK(descending_rank >= 1 && descending_rank <= K, "bad descending_rank");
  TORCH_CHECK(idx.size(0) == M && idx.size(1) == R, "idx shape mismatch");
  TORCH_CHECK(top_q.size(0) == M && top_q.size(1) == R, "top_q shape mismatch");
  int threads = 256;
  int words = (K + 31) >> 5;
  size_t smem = static_cast<size_t>(K) * sizeof(float) + static_cast<size_t>(K) * sizeof(int) + static_cast<size_t>(words) * sizeof(uint32_t);
  fused_topr_pack_kernel<<<M, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
      R,
      descending_rank,
      A_pack.data_ptr<uint8_t>(),
      body_scale.data_ptr<float>(),
      top_scale.data_ptr<float>(),
      top_q.data_ptr<int8_t>(),
      idx.data_ptr<int32_t>(),
      M,
      K,
      static_cast<float>(eps64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_topr_pack", &fused_topr_pack, "Fused top-r selection and split activation packing v41");
}
"""


def load_fused_topr_pack_ext(verbose: bool = False):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return load_inline(
        name="fused_topr_pack_v41_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(load_fused_topr_pack_ext(verbose=True))
