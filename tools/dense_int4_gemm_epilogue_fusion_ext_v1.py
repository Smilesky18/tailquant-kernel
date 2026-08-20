"""Experimental dense INT4 GEMM epilogue-fusion extension.

This file is intentionally standalone. It does not patch the installed
``quarot._CUDA`` extension or change the production GEMM path.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/scale_type.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/numeric_types.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_U8(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Byte, #x " must be uint8")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Half, #x " must be fp16")

static void check_int4_gemm_inputs(torch::Tensor A, torch::Tensor B) {
  CHECK_CUDA(A);
  CHECK_CUDA(B);
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);
  CHECK_U8(A);
  CHECK_U8(B);
  TORCH_CHECK(A.dim() == 2, "A must be 2D packed [M,K/2]");
  TORCH_CHECK(B.dim() == 2, "B must be 2D packed [N,K/2]");
  TORCH_CHECK(A.size(1) == B.size(1), "A/B packed-K mismatch");
  TORCH_CHECK((A.size(1) % 32) == 0, "packed K must be a multiple of 32, matching quarot.matmul");
}

template <typename Gemm>
static void run_gemm(torch::Tensor A, torch::Tensor B, torch::Tensor D, typename Gemm::EpilogueOutputOp::Params epilogue) {
  using GemmCoord = cutlass::gemm::GemmCoord;

  int64_t M64 = A.size(0);
  int64_t N64 = B.size(0);
  int64_t K64 = A.size(1) * 2;
  TORCH_CHECK(M64 <= std::numeric_limits<int>::max(), "M too large");
  TORCH_CHECK(N64 <= std::numeric_limits<int>::max(), "N too large");
  TORCH_CHECK(K64 <= std::numeric_limits<int>::max(), "K too large");

  auto* Aptr = reinterpret_cast<cutlass::int4b_t const*>(A.data_ptr<uint8_t>());
  auto* Bptr = reinterpret_cast<cutlass::int4b_t const*>(B.data_ptr<uint8_t>());
  auto* Dptr = reinterpret_cast<cutlass::half_t*>(D.data_ptr<at::Half>());

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M64), static_cast<GemmCoord::Index>(N64), static_cast<GemmCoord::Index>(K64)},
      {Aptr, static_cast<int>(K64)},
      {Bptr, static_cast<int>(K64)},
      {Dptr, static_cast<int>(N64)},
      {Dptr, static_cast<int>(N64)},
      epilogue};

  Gemm gemm_op;
  auto status = gemm_op(arguments, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed: ", cutlassGetStatusString(status));
}

torch::Tensor dense_i4_gemm_fp16_unscaled(torch::Tensor A, torch::Tensor B) {
  check_int4_gemm_inputs(A, B);
  auto D = torch::empty({A.size(0), B.size(0)}, torch::dtype(torch::kHalf).device(A.device()));

  using OutputOp = cutlass::epilogue::thread::LinearCombination<
      cutlass::half_t,
      8,
      int32_t,
      float,
      cutlass::epilogue::thread::ScaleType::Default>;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t,
      cutlass::layout::RowMajor,
      cutlass::int4b_t,
      cutlass::layout::ColumnMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      int32_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          cutlass::arch::OpClassTensorOp,
          cutlass::arch::Sm80,
          cutlass::int4b_t,
          cutlass::int4b_t,
          cutlass::half_t,
          int32_t>::ThreadblockShape,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          cutlass::arch::OpClassTensorOp,
          cutlass::arch::Sm80,
          cutlass::int4b_t,
          cutlass::int4b_t,
          cutlass::half_t,
          int32_t>::WarpShape,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          cutlass::arch::OpClassTensorOp,
          cutlass::arch::Sm80,
          cutlass::int4b_t,
          cutlass::int4b_t,
          cutlass::half_t,
          int32_t>::InstructionShape,
      OutputOp>;

  run_gemm<Gemm>(A, B, D, typename OutputOp::Params(1.0f, 0.0f));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_i4_gemm_fp16_unscaled", &dense_i4_gemm_fp16_unscaled,
        "Experimental dense INT4 GEMM with fp16 output epilogue, unscaled");
}
"""


def load_dense_int4_gemm_epilogue_fusion_ext_v1(verbose: bool = False):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    include_dirs = [
        str(ROOT / "third_party/cutlass/include"),
        str(ROOT / "third_party/cutlass/tools/util/include"),
    ]
    return load_inline(
        name="dense_int4_gemm_epilogue_fusion_ext_v1",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_include_paths=include_dirs,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(load_dense_int4_gemm_epilogue_fusion_ext_v1(verbose=True))
