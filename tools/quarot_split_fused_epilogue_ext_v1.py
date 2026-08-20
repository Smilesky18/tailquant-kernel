"""Experimental QuaRot-style INT4 GEMM with split sparse correction inside CUTLASS output-op.

This is intentionally standalone. It does not modify the production QuaRot extension.
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
#include <stdint.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/gemm/kernel/gemm.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/numeric_types.h>
#include <cutlass/array.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_U8(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Byte, #x " must be uint8")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Half, #x " must be fp16")

static __device__ __forceinline__ int decode_s4(uint8_t nib) {
  int v = static_cast<int>(nib & 0x0f);
  return (v >= 8) ? (v - 16) : v;
}

template <typename ThreadMap_, typename ThreadblockShape_, typename ThreadblockSwizzle_>
struct SplitSparseOutputOp {
  using ThreadMap = ThreadMap_;
  using ThreadblockShape = ThreadblockShape_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;
  using ElementSource = cutlass::half_t;
  using ElementC = ElementSource;
  using ElementD = ElementOutput;
  static int const kCount = 8;

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentSource = cutlass::Array<ElementSource, kCount>;

  struct Params {
    float const* body_scale;
    int8_t const* top_q;
    int32_t const* idx;
    uint8_t const* B_row_pack;
    float const* top_scale;
    float const* w_scale;
    int M;
    int N;
    int R;
    int swizzle_log_tile;

    CUTLASS_HOST_DEVICE
    Params(): body_scale(nullptr), top_q(nullptr), idx(nullptr), B_row_pack(nullptr), top_scale(nullptr), w_scale(nullptr), M(0), N(0), R(0), swizzle_log_tile(0) {}

    CUTLASS_HOST_DEVICE
    Params(float const* body_scale_, int8_t const* top_q_, int32_t const* idx_, uint8_t const* B_row_pack_, float const* top_scale_, float const* w_scale_, int M_, int N_, int R_, int swizzle_log_tile_):
      body_scale(body_scale_), top_q(top_q_), idx(idx_), B_row_pack(B_row_pack_), top_scale(top_scale_), w_scale(w_scale_), M(M_), N(N_), R(R_), swizzle_log_tile(swizzle_log_tile_) {}
  };

  Params params;
  int32_t const* shared_idx;
  int8_t const* shared_top_q;
  int tile_m_start;
  mutable int call_idx;

  CUTLASS_HOST_DEVICE
  SplitSparseOutputOp(Params const& p): params(p), shared_idx(nullptr), shared_top_q(nullptr), tile_m_start(0), call_idx(0) {}

  CUTLASS_DEVICE
  void set_shared(int32_t const* s_idx, int8_t const* s_top, int tile_m_start_) {
    shared_idx = s_idx;
    shared_top_q = s_top;
    tile_m_start = tile_m_start_;
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const { return false; }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int, int) {}

  CUTLASS_DEVICE
  int advanced_row_start(int iter, int start_row) const {
    int row = start_row;
    int s0 = 0, s1 = 0, s2 = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int t = 0; t < ThreadMap::Count::kTile; ++t) {
      if (t >= iter) break;
      ++s0;
      row += ThreadMap::Shape::kRow;
      if (s0 == ThreadMap::Count::kRow) {
        s0 = 0;
        ++s1;
        row += (ThreadMap::Shape::kGroup - 1) * ThreadMap::Shape::kRow * ThreadMap::Count::kRow;
        if (s1 == ThreadMap::Count::kGroup) {
          s1 = 0;
          ++s2;
          row += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;
          if (s2 == ThreadMap::Count::kCluster) {
            s2 = 0;
            row += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
          }
        }
      }
    }
    return row;
  }

  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
    FragmentOutput result;

    int local_call = call_idx++;
    int calls_per_fragment = ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow * ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster;
    int iter = local_call / calls_per_fragment;
    int frag_vec = local_call - iter * calls_per_fragment;
    int column_iter = frag_vec % ThreadMap::Iterations::kColumn;
    int frag_row_idx = frag_vec / ThreadMap::Iterations::kColumn;
    int row_iter = frag_row_idx % ThreadMap::Iterations::kRow;
    int tmp = frag_row_idx / ThreadMap::Iterations::kRow;
    int group_iter = tmp % ThreadMap::Iterations::kGroup;
    int cluster_iter = tmp / ThreadMap::Iterations::kGroup;

    cutlass::gemm::GemmCoord tile_offset = ThreadblockSwizzle::get_tile_offset(params.swizzle_log_tile);
    cutlass::MatrixCoord tb_offset(tile_offset.m() * ThreadblockShape::kM, tile_offset.n() * ThreadblockShape::kN);
    cutlass::MatrixCoord thread_offset = ThreadMap::initial_offset(threadIdx.x) + tb_offset;

    int row_start = advanced_row_start(iter, thread_offset.row());
    int row_offset = row_iter * ThreadMap::Delta::kRow + group_iter * ThreadMap::Delta::kGroup + cluster_iter * ThreadMap::Delta::kCluster;
    int m = row_start + row_offset;
    int n_base = thread_offset.column() + column_iter * ThreadMap::Delta::kColumn;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      int n = n_base + i;
      float value = 0.0f;
      if (m < params.M && n < params.N) {
        float ws = params.w_scale[n];
        float dense = static_cast<float>(accumulator[i]) * params.body_scale[m] * ws;
        int sparse_acc = 0;
        int row_base = m * params.R;
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < 30; ++r) {
          int k;
          int aq;
          int local_m = m - tile_m_start;
          if (shared_idx && local_m >= 0 && local_m < ThreadblockShape::kM) {
            int shared_off = local_m * params.R + r;
            k = shared_idx[shared_off];
            aq = static_cast<int>(shared_top_q[shared_off]);
          } else {
            k = params.idx[row_base + r];
            aq = static_cast<int>(params.top_q[row_base + r]);
          }
          uint8_t packed = params.B_row_pack[(static_cast<long long>(k) * params.N + n) >> 1];
          int bq = decode_s4((n & 1) ? (packed >> 4) : (packed & 0x0f));
          sparse_acc += aq * bq;
        }
        float sparse = static_cast<float>(sparse_acc) * params.top_scale[m] * ws;
        value = dense + sparse;
      }
      result[i] = cutlass::half_t(__float2half_rn(value));
    }
    return result;
  }

  CUTLASS_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator, FragmentSource const&) const {
    return (*this)(accumulator);
  }
};


template <typename BaseEpilogue_, typename ThreadblockShape_, typename ThreadblockSwizzle_, int MaxR>
class SplitStagingEpilogue {
public:
  using BaseEpilogue = BaseEpilogue_;
  using ThreadblockShape = ThreadblockShape_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using OutputOp = typename BaseEpilogue::OutputOp;
  using OutputTileIterator = typename BaseEpilogue::OutputTileIterator;
  using AccumulatorTile = typename BaseEpilogue::AccumulatorTile;
  using ElementOutput = typename BaseEpilogue::ElementOutput;
  using WarpCount = typename BaseEpilogue::WarpCount;
  static int const kBlockThreads = BaseEpilogue::kBlockThreads;

  struct SharedStorage {
    typename BaseEpilogue::Base::SharedStorage base;
    int32_t staged_idx[ThreadblockShape::kM * MaxR];
    int8_t staged_top_q[ThreadblockShape::kM * MaxR];
  };

private:
  SharedStorage &storage_;
  BaseEpilogue base_;
  int thread_idx_;

public:
  CUTLASS_DEVICE
  SplitStagingEpilogue(SharedStorage &shared_storage, int thread_idx, int warp_idx, int lane_idx)
      : storage_(shared_storage), base_(shared_storage.base, thread_idx, warp_idx, lane_idx), thread_idx_(thread_idx) {}

  CUTLASS_DEVICE
  int stage_sparse_metadata(OutputOp const &output_op) {
    cutlass::gemm::GemmCoord tile_offset = ThreadblockSwizzle::get_tile_offset(output_op.params.swizzle_log_tile);
    int tile_m_start = tile_offset.m() * ThreadblockShape::kM;
    int total = ThreadblockShape::kM * output_op.params.R;
    for (int linear = thread_idx_; linear < total; linear += kBlockThreads) {
      int row = linear / output_op.params.R;
      int r = linear - row * output_op.params.R;
      int gm = tile_m_start + row;
      int dst = row * output_op.params.R + r;
      if (gm < output_op.params.M) {
        int src = gm * output_op.params.R + r;
        storage_.staged_idx[dst] = output_op.params.idx[src];
        storage_.staged_top_q[dst] = output_op.params.top_q[src];
      } else {
        storage_.staged_idx[dst] = 0;
        storage_.staged_top_q[dst] = 0;
      }
    }
    return tile_m_start;
  }

  CUTLASS_DEVICE
  void operator()(OutputOp const &output_op, OutputTileIterator destination_iterator, AccumulatorTile const &accumulators) {
    int tile_m_start = stage_sparse_metadata(output_op);
    __syncthreads();
    OutputOp local_op = output_op;
    local_op.set_shared(storage_.staged_idx, storage_.staged_top_q, tile_m_start);
    base_(local_op, destination_iterator, accumulators);
  }

  CUTLASS_DEVICE
  void operator()(OutputOp const &output_op, OutputTileIterator destination_iterator, AccumulatorTile const &accumulators, OutputTileIterator source_iterator) {
    int tile_m_start = stage_sparse_metadata(output_op);
    __syncthreads();
    OutputOp local_op = output_op;
    local_op.set_shared(storage_.staged_idx, storage_.staged_top_q, tile_m_start);
    base_(local_op, destination_iterator, accumulators, source_iterator);
  }
};

static void check_inputs(torch::Tensor A, torch::Tensor B, torch::Tensor body_scale, torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale) {
  CHECK_CUDA(A); CHECK_CUDA(B); CHECK_CUDA(body_scale); CHECK_CUDA(top_q); CHECK_CUDA(idx); CHECK_CUDA(B_row_pack); CHECK_CUDA(top_scale); CHECK_CUDA(w_scale);
  CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B); CHECK_CONTIGUOUS(body_scale); CHECK_CONTIGUOUS(top_q); CHECK_CONTIGUOUS(idx); CHECK_CONTIGUOUS(B_row_pack); CHECK_CONTIGUOUS(top_scale); CHECK_CONTIGUOUS(w_scale);
  CHECK_U8(A); CHECK_U8(B); CHECK_U8(B_row_pack);
  TORCH_CHECK(body_scale.scalar_type() == at::ScalarType::Float, "body_scale must be fp32");
  TORCH_CHECK(top_scale.scalar_type() == at::ScalarType::Float, "top_scale must be fp32");
  TORCH_CHECK(w_scale.scalar_type() == at::ScalarType::Float, "w_scale must be fp32");
  TORCH_CHECK(top_q.scalar_type() == at::ScalarType::Char, "top_q must be int8");
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Int, "idx must be int32");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A/B must be packed 2D");
  TORCH_CHECK(A.size(1) == B.size(1), "packed K mismatch");
  TORCH_CHECK(top_q.dim() == 2 && idx.sizes() == top_q.sizes(), "top_q/idx shape mismatch");
  TORCH_CHECK(top_q.size(0) == A.size(0), "top_q M mismatch");
  TORCH_CHECK(B_row_pack.dim() == 2 && B_row_pack.size(1) * 2 == B.size(0), "B_row_pack must be [K,N/2]");
}

torch::Tensor quarot_split_gemm_fused_epilogue_v1(torch::Tensor A, torch::Tensor B, torch::Tensor body_scale, torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale) {
  check_inputs(A, B, body_scale, top_q, idx, B_row_pack, top_scale, w_scale);
  int M = static_cast<int>(A.size(0));
  int N = static_cast<int>(B.size(0));
  int K = static_cast<int>(A.size(1) * 2);
  int R = static_cast<int>(top_q.size(1));
  TORCH_CHECK(R == 30, "v1c specialized output-op currently supports R=30 only");
  auto D = torch::empty({M, N}, torch::dtype(torch::kHalf).device(A.device()));

  using ElementA = cutlass::int4b_t;
  using ElementB = cutlass::int4b_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using Config = cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass, ArchTag, ElementA, ElementB, ElementD, ElementAccumulator>;
  using ThreadblockShape = typename Config::ThreadblockShape;
  using WarpShape = typename Config::WarpShape;
  using InstructionShape = typename Config::InstructionShape;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  static int const PartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<ThreadblockShape, WarpShape, PartitionsK, ElementD, 8>::Type;
  using OutputOp = SplitSparseOutputOp<ThreadMap, ThreadblockShape, ThreadblockSwizzle>;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementA, cutlass::layout::RowMajor,
      ElementB, cutlass::layout::ColumnMajor,
      ElementD, cutlass::layout::RowMajor,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      OutputOp,
      ThreadblockSwizzle>;

  cutlass::gemm::GemmCoord problem(M, N, K);
  ThreadblockSwizzle swizzle;
  auto tiled = swizzle.get_tiled_shape(problem, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, 1);
  int log_tile = swizzle.get_log_tile(tiled);

  auto* Aptr = reinterpret_cast<ElementA const*>(A.data_ptr<uint8_t>());
  auto* Bptr = reinterpret_cast<ElementB const*>(B.data_ptr<uint8_t>());
  auto* Dptr = reinterpret_cast<ElementD*>(D.data_ptr<at::Half>());
  typename OutputOp::Params epilogue(
      body_scale.data_ptr<float>(), top_q.data_ptr<int8_t>(), idx.data_ptr<int32_t>(), B_row_pack.data_ptr<uint8_t>(), top_scale.data_ptr<float>(), w_scale.data_ptr<float>(), M, N, R, log_tile);

  typename Gemm::Arguments args{
      problem,
      {Aptr, K},
      {Bptr, K},
      {Dptr, N},
      {Dptr, N},
      epilogue};

  Gemm gemm;
  auto status = gemm(args, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS fused split GEMM failed: ", cutlassGetStatusString(status));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}


torch::Tensor quarot_split_gemm_fused_epilogue_staged_v2(torch::Tensor A, torch::Tensor B, torch::Tensor body_scale, torch::Tensor top_q, torch::Tensor idx, torch::Tensor B_row_pack, torch::Tensor top_scale, torch::Tensor w_scale) {
  check_inputs(A, B, body_scale, top_q, idx, B_row_pack, top_scale, w_scale);
  int M = static_cast<int>(A.size(0));
  int N = static_cast<int>(B.size(0));
  int K = static_cast<int>(A.size(1) * 2);
  int R = static_cast<int>(top_q.size(1));
  TORCH_CHECK(R == 30, "v2 staged epilogue currently supports R=30 only");
  auto D = torch::empty({M, N}, torch::dtype(torch::kHalf).device(A.device()));

  using ElementA = cutlass::int4b_t;
  using ElementB = cutlass::int4b_t;
  using ElementD = cutlass::half_t;
  using ElementAccumulator = int32_t;
  using ArchTag = cutlass::arch::Sm80;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using Config = cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass, ArchTag, ElementA, ElementB, ElementD, ElementAccumulator>;
  using ThreadblockShape = typename Config::ThreadblockShape;
  using WarpShape = typename Config::WarpShape;
  using InstructionShape = typename Config::InstructionShape;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  static int const PartitionsK = ThreadblockShape::kK / WarpShape::kK;
  using ThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<ThreadblockShape, WarpShape, PartitionsK, ElementD, 8>::Type;
  using OutputOp = SplitSparseOutputOp<ThreadMap, ThreadblockShape, ThreadblockSwizzle>;

  using DefaultGemmKernel = typename cutlass::gemm::kernel::DefaultGemm<
      ElementA, cutlass::layout::RowMajor, Config::kAlignmentA,
      ElementB, cutlass::layout::ColumnMajor, Config::kAlignmentB,
      ElementD, cutlass::layout::RowMajor,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      OutputOp,
      ThreadblockSwizzle,
      Config::kStages,
      false,
      typename Config::Operator>::GemmKernel;

  using BaseEpilogue = typename DefaultGemmKernel::Epilogue;
  using StagedEpilogue = SplitStagingEpilogue<BaseEpilogue, ThreadblockShape, ThreadblockSwizzle, 30>;
  using GemmKernel = cutlass::gemm::kernel::Gemm<typename DefaultGemmKernel::Mma, StagedEpilogue, ThreadblockSwizzle, false>;

  cutlass::gemm::GemmCoord problem(M, N, K);
  ThreadblockSwizzle swizzle;
  auto grid_shape = swizzle.get_tiled_shape(problem, {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK}, 1);
  int log_tile = swizzle.get_log_tile(grid_shape);

  auto* Aptr = reinterpret_cast<ElementA const*>(A.data_ptr<uint8_t>());
  auto* Bptr = reinterpret_cast<ElementB const*>(B.data_ptr<uint8_t>());
  auto* Dptr = reinterpret_cast<ElementD*>(D.data_ptr<at::Half>());
  typename OutputOp::Params epilogue(
      body_scale.data_ptr<float>(), top_q.data_ptr<int8_t>(), idx.data_ptr<int32_t>(), B_row_pack.data_ptr<uint8_t>(), top_scale.data_ptr<float>(), w_scale.data_ptr<float>(), M, N, R, log_tile);

  typename DefaultGemmKernel::Mma::IteratorA::TensorRef ref_A(const_cast<ElementA*>(Aptr), cutlass::layout::RowMajor(K));
  typename DefaultGemmKernel::Mma::IteratorB::TensorRef ref_B(const_cast<ElementB*>(Bptr), cutlass::layout::ColumnMajor(K));
  typename StagedEpilogue::OutputTileIterator::TensorRef ref_C(Dptr, cutlass::layout::RowMajor(N));
  typename StagedEpilogue::OutputTileIterator::TensorRef ref_D(Dptr, cutlass::layout::RowMajor(N));
  auto can = GemmKernel::can_implement(problem, ref_A, ref_B, ref_C, ref_D);
  TORCH_CHECK(can == cutlass::Status::kSuccess, "CUTLASS staged fused split GEMM cannot implement problem: ", cutlassGetStatusString(can));

  typename GemmKernel::Params params(
      problem,
      grid_shape,
      ref_A,
      ref_B,
      ref_C,
      ref_D,
      epilogue,
      nullptr,
      nullptr,
      nullptr,
      nullptr);

  dim3 grid = swizzle.get_grid_shape(grid_shape);
  dim3 block(GemmKernel::kThreadCount, 1, 1);
  int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    cudaError_t attr = cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    TORCH_CHECK(attr == cudaSuccess, "cudaFuncSetAttribute failed for staged fused GEMM: ", cudaGetErrorString(attr), " smem=", smem_size);
  }
  cutlass::Kernel<GemmKernel><<<grid, block, smem_size, at::cuda::getCurrentCUDAStream()>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quarot_split_gemm_fused_epilogue_v1", &quarot_split_gemm_fused_epilogue_v1, "Experimental QuaRot INT4 GEMM with split sparse correction in CUTLASS output-op");
  m.def("quarot_split_gemm_fused_epilogue_staged_v2", &quarot_split_gemm_fused_epilogue_staged_v2, "Experimental QuaRot INT4 GEMM with split sparse correction staged in custom threadblock epilogue");
}
"""


def load_quarot_split_fused_epilogue_ext_v1(verbose: bool = False):
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    include_dirs = [
        str(ROOT / "third_party/cutlass/include"),
        str(ROOT / "third_party/cutlass/tools/util/include"),
    ]
    return load_inline(
        name="quarot_split_fused_epilogue_ext_v2",
        cpp_sources="",
        cuda_sources=CUDA_SRC,
        functions=None,
        extra_include_paths=include_dirs,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
        verbose=verbose,
    )


if __name__ == "__main__":
    print(load_quarot_split_fused_epilogue_ext_v1(verbose=True))
