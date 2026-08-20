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

static __device__ __forceinline__ int decode_s4(uint8_t nib) {
  int v = static_cast<int>(nib & 0x0f);
  return (v >= 8) ? (v - 16) : v;
}

template<int COLS_PER_THREAD, int BLOCK_THREADS>
__global__ void scale_sparse_epilogue_rowmajor_kernel(
    const int32_t* __restrict__ C,
    const float* __restrict__ body_scale,
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

  float bs = body_scale[m];
  float ts = top_scale[m];
  long long off0 = static_cast<long long>(m) * N + n0;
  #pragma unroll
  for (int i = 0; i < COLS_PER_THREAD; ++i) {
    if (n0 + i < N) {
      float ws = w_scale[n0 + i];
      float dense = static_cast<float>(C[off0 + i]) * bs * ws;
      float sparse = static_cast<float>(acc[i]) * ts * ws;
      out[off0 + i] = __float2half_rn(dense + sparse);
    }
  }
}

template<int COLS_PER_THREAD>
void launch_scale_sparse_epilogue(
    torch::Tensor C,
    torch::Tensor body_scale,
    torch::Tensor top_q,
    torch::Tensor idx,
    torch::Tensor B_row_pack,
    torch::Tensor top_scale,
    torch::Tensor w_scale,
    torch::Tensor out,
    int K) {
  constexpr int BLOCK_THREADS = 128;
  int M = static_cast<int>(top_q.size(0));
  int R = static_cast<int>(top_q.size(1));
  int N = static_cast<int>(out.size(1));
  dim3 block(BLOCK_THREADS);
  dim3 grid(((N + COLS_PER_THREAD - 1) / COLS_PER_THREAD + BLOCK_THREADS - 1) / BLOCK_THREADS, M);
  size_t smem = static_cast<size_t>(R) * sizeof(int32_t) + static_cast<size_t>(R) * sizeof(int8_t);
  scale_sparse_epilogue_rowmajor_kernel<COLS_PER_THREAD, BLOCK_THREADS><<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
      C.data_ptr<int32_t>(),
      body_scale.data_ptr<float>(),
      top_q.data_ptr<int8_t>(),
      idx.data_ptr<int32_t>(),
      B_row_pack.data_ptr<uint8_t>(),
      top_scale.data_ptr<float>(),
      w_scale.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      M,
      K,
      N,
      R);
}

void scale_sparse_epilogue_quad(
    torch::Tensor C,
    torch::Tensor body_scale,
    torch::Tensor top_q,
    torch::Tensor idx,
    torch::Tensor B_row_pack,
    torch::Tensor top_scale,
    torch::Tensor w_scale,
    torch::Tensor out,
    int64_t K64) {
  CHECK_CUDA(C); CHECK_CUDA(body_scale); CHECK_CUDA(top_q); CHECK_CUDA(idx); CHECK_CUDA(B_row_pack); CHECK_CUDA(top_scale); CHECK_CUDA(w_scale); CHECK_CUDA(out);
  CHECK_CONTIGUOUS(C); CHECK_CONTIGUOUS(body_scale); CHECK_CONTIGUOUS(top_q); CHECK_CONTIGUOUS(idx); CHECK_CONTIGUOUS(B_row_pack); CHECK_CONTIGUOUS(top_scale); CHECK_CONTIGUOUS(w_scale); CHECK_CONTIGUOUS(out);
  TORCH_CHECK(C.scalar_type() == at::ScalarType::Int, "C must be int32");
  TORCH_CHECK(body_scale.scalar_type() == at::ScalarType::Float, "body_scale must be fp32");
  TORCH_CHECK(top_q.scalar_type() == at::ScalarType::Char, "top_q must be int8");
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Int, "idx must be int32");
  TORCH_CHECK(B_row_pack.scalar_type() == at::ScalarType::Byte, "B_row_pack must be uint8");
  TORCH_CHECK(top_scale.scalar_type() == at::ScalarType::Float, "top_scale must be fp32");
  TORCH_CHECK(w_scale.scalar_type() == at::ScalarType::Float, "w_scale must be fp32");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half, "out must be fp16");
  launch_scale_sparse_epilogue<4>(C, body_scale, top_q, idx, B_row_pack, top_scale, w_scale, out, static_cast<int>(K64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void scale_sparse_epilogue_oct(
    torch::Tensor C,
    torch::Tensor body_scale,
    torch::Tensor top_q,
    torch::Tensor idx,
    torch::Tensor B_row_pack,
    torch::Tensor top_scale,
    torch::Tensor w_scale,
    torch::Tensor out,
    int64_t K64) {
  CHECK_CUDA(C); CHECK_CUDA(body_scale); CHECK_CUDA(top_q); CHECK_CUDA(idx); CHECK_CUDA(B_row_pack); CHECK_CUDA(top_scale); CHECK_CUDA(w_scale); CHECK_CUDA(out);
  CHECK_CONTIGUOUS(C); CHECK_CONTIGUOUS(body_scale); CHECK_CONTIGUOUS(top_q); CHECK_CONTIGUOUS(idx); CHECK_CONTIGUOUS(B_row_pack); CHECK_CONTIGUOUS(top_scale); CHECK_CONTIGUOUS(w_scale); CHECK_CONTIGUOUS(out);
  TORCH_CHECK(C.scalar_type() == at::ScalarType::Int, "C must be int32");
  TORCH_CHECK(body_scale.scalar_type() == at::ScalarType::Float, "body_scale must be fp32");
  TORCH_CHECK(top_q.scalar_type() == at::ScalarType::Char, "top_q must be int8");
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Int, "idx must be int32");
  TORCH_CHECK(B_row_pack.scalar_type() == at::ScalarType::Byte, "B_row_pack must be uint8");
  TORCH_CHECK(top_scale.scalar_type() == at::ScalarType::Float, "top_scale must be fp32");
  TORCH_CHECK(w_scale.scalar_type() == at::ScalarType::Float, "w_scale must be fp32");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Half, "out must be fp16");
  launch_scale_sparse_epilogue<8>(C, body_scale, top_q, idx, B_row_pack, top_scale, w_scale, out, static_cast<int>(K64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scale_sparse_epilogue_quad", &scale_sparse_epilogue_quad, "v58 fused dense scale and sparse top correction epilogue, 4 cols/thread");
  m.def("scale_sparse_epilogue_oct", &scale_sparse_epilogue_oct, "v58 fused dense scale and sparse top correction epilogue, 8 cols/thread");
}
"""


def load_fused_sparse_epilogue_ext(verbose: bool = False):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    return load_inline(
        name="fused_sparse_epilogue_v58_ext",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(load_fused_sparse_epilogue_ext(verbose=True))
