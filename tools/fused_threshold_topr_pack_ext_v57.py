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

__device__ __forceinline__ uint16_t abs_half_key(half x) {
  uint16_t raw = *reinterpret_cast<uint16_t*>(&x);
  return static_cast<uint16_t>(raw & 0x7fffu);
}

__device__ __forceinline__ float half_key_to_float(uint16_t key) {
  half h = *reinterpret_cast<half*>(&key);
  return __half2float(h);
}

__device__ __forceinline__ unsigned char* align_ptr(unsigned char* p, uintptr_t align) {
  uintptr_t x = reinterpret_cast<uintptr_t>(p);
  x = (x + align - 1) & ~(align - 1);
  return reinterpret_cast<unsigned char*>(x);
}

template<int K, int BLOCK_THREADS>
__device__ uint16_t select_desc_rank_key(
    const half* __restrict__ Arow,
    int rank,
    uint32_t* __restrict__ hist) {
  __shared__ uint16_t s_prefix;
  __shared__ int s_remaining;
  if (threadIdx.x == 0) {
    s_prefix = 0;
    s_remaining = rank;
  }
  __syncthreads();

  #pragma unroll
  for (int shift = 12; shift >= 0; shift -= 4) {
    if (threadIdx.x < 16) hist[threadIdx.x] = 0u;
    __syncthreads();

    for (int k = threadIdx.x; k < K; k += BLOCK_THREADS) {
      uint16_t key = abs_half_key(Arow[k]);
      bool match = true;
      if (shift < 12) {
        uint16_t mask = static_cast<uint16_t>(0xffffu << (shift + 4));
        match = ((key & mask) == (s_prefix & mask));
      }
      if (match) {
        int digit = (key >> shift) & 0x0f;
        atomicAdd(hist + digit, 1u);
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      for (int digit = 15; digit >= 0; --digit) {
        int c = static_cast<int>(hist[digit]);
        if (s_remaining > c) {
          s_remaining -= c;
        } else {
          s_prefix = static_cast<uint16_t>(s_prefix | (digit << shift));
          break;
        }
      }
    }
    __syncthreads();
  }
  return s_prefix;
}

template<int K, int BLOCK_THREADS>
__global__ void threshold_topr_pack_kernel(
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

  using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
  extern __shared__ unsigned char smem_raw[];
  auto* scan_storage = reinterpret_cast<typename BlockScan::TempStorage*>(smem_raw);
  unsigned char* p = smem_raw + sizeof(typename BlockScan::TempStorage);
  uint32_t* hist = reinterpret_cast<uint32_t*>(align_ptr(p, alignof(uint32_t)));
  p = reinterpret_cast<unsigned char*>(hist + 16);
  uint32_t* bits = reinterpret_cast<uint32_t*>(align_ptr(p, alignof(uint32_t)));
  p = reinterpret_cast<unsigned char*>(bits + ((K + 31) / 32));
  int* counter = reinterpret_cast<int*>(align_ptr(p, alignof(int)));

  const half* Arow = A + static_cast<long long>(m) * K;
  int32_t* Irow = idx + static_cast<long long>(m) * R;
  int8_t* Trow = top_q + static_cast<long long>(m) * R;

  uint16_t top_key = select_desc_rank_key<K, BLOCK_THREADS>(Arow, 1, hist);
  uint16_t tail_key = select_desc_rank_key<K, BLOCK_THREADS>(Arow, R, hist);
  uint16_t body_key = select_desc_rank_key<K, BLOCK_THREADS>(Arow, descending_rank, hist);

  __shared__ float s_top_scale;
  __shared__ float s_body_scale;
  if (threadIdx.x == 0) {
    s_top_scale = fmaxf(half_key_to_float(top_key), eps) / 7.0f;
    s_body_scale = fmaxf(half_key_to_float(body_key), eps) / 7.0f;
    top_scale[m] = s_top_scale;
    body_scale[m] = s_body_scale;
    *counter = 0;
  }
  for (int w = threadIdx.x; w < (K + 31) / 32; w += BLOCK_THREADS) bits[w] = 0u;
  __syncthreads();

  for (int base = 0; base < K; base += BLOCK_THREADS) {
    int k = base + threadIdx.x;
    int flag = (k < K && abs_half_key(Arow[k]) > tail_key) ? 1 : 0;
    int prefix = 0;
    int tile_count = 0;
    BlockScan(*scan_storage).ExclusiveSum(flag, prefix, tile_count);
    __syncthreads();
    int base_pos = *counter;
    int pos = base_pos + prefix;
    if (flag && pos < R) {
      Irow[pos] = k;
      Trow[pos] = static_cast<int8_t>(clamp_s4(__half2float(Arow[k]) / s_top_scale));
      atomicOr(bits + (k >> 5), 1u << (k & 31));
    }
    __syncthreads();
    if (threadIdx.x == 0) *counter = base_pos + tile_count;
    __syncthreads();
  }

  for (int base = 0; base < K; base += BLOCK_THREADS) {
    int k = base + threadIdx.x;
    int current = *counter;
    int flag = (k < K && current < R && abs_half_key(Arow[k]) == tail_key) ? 1 : 0;
    int prefix = 0;
    int tile_count = 0;
    BlockScan(*scan_storage).ExclusiveSum(flag, prefix, tile_count);
    __syncthreads();
    int base_pos = *counter;
    int pos = base_pos + prefix;
    if (flag && pos < R) {
      Irow[pos] = k;
      Trow[pos] = static_cast<int8_t>(clamp_s4(__half2float(Arow[k]) / s_top_scale));
      atomicOr(bits + (k >> 5), 1u << (k & 31));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int next = base_pos + tile_count;
      *counter = next > R ? R : next;
    }
    __syncthreads();
  }

  for (int k2 = threadIdx.x * 2; k2 < K; k2 += BLOCK_THREADS * 2) {
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

template<int K>
void launch_threshold_topr_pack(
    torch::Tensor A,
    int R,
    int descending_rank,
    torch::Tensor A_pack,
    torch::Tensor body_scale,
    torch::Tensor top_scale,
    torch::Tensor top_q,
    torch::Tensor idx,
    int M,
    float eps) {
  constexpr int BLOCK_THREADS = 256;
  using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
  size_t smem = sizeof(typename BlockScan::TempStorage)
      + 16 * sizeof(uint32_t)
      + ((K + 31) / 32) * sizeof(uint32_t)
      + sizeof(int)
      + 64;
  auto kernel = threshold_topr_pack_kernel<K, BLOCK_THREADS>;
  cudaError_t attr_status = cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem));
  if (attr_status != cudaSuccess && smem > 48 * 1024) {
    C10_CUDA_CHECK(attr_status);
  }
  kernel<<<M, BLOCK_THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
      R,
      descending_rank,
      A_pack.data_ptr<uint8_t>(),
      body_scale.data_ptr<float>(),
      top_scale.data_ptr<float>(),
      top_q.data_ptr<int8_t>(),
      idx.data_ptr<int32_t>(),
      M,
      eps);
}

void threshold_topr_pack(
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
  TORCH_CHECK((K == 4096 || K == 12288), "threshold_topr_pack v57 supports K=4096 or K=12288");
  TORCH_CHECK(R > 0 && R <= 512 && R < K, "threshold_topr_pack v57 supports 0<R<=512");
  TORCH_CHECK(descending_rank >= 1 && descending_rank <= K, "bad descending_rank");
  TORCH_CHECK(idx.size(0) == M && idx.size(1) == R, "idx shape mismatch");
  TORCH_CHECK(top_q.size(0) == M && top_q.size(1) == R, "top_q shape mismatch");
  if (K == 4096) {
    launch_threshold_topr_pack<4096>(A, R, descending_rank, A_pack, body_scale, top_scale, top_q, idx, M, static_cast<float>(eps64));
  } else {
    launch_threshold_topr_pack<12288>(A, R, descending_rank, A_pack, body_scale, top_scale, top_q, idx, M, static_cast<float>(eps64));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("threshold_topr_pack", &threshold_topr_pack, "v57 exact radix-threshold top-r selection with ordered compact tail and activation packing");
}
"""


def load_threshold_topr_pack_ext(verbose: bool = False):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return load_inline(
        name="threshold_topr_pack_v57_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(load_threshold_topr_pack_ext(verbose=True))
