import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
DEFAULT_BUILD_DIR = ROOT / "experiments/kernel_quant/layer_latency_split_v1/reports/quarot_sm120_jit_v1"


def _remove_unwanted_pytorch_nvcc_flags():
    import torch.utils.cpp_extension as torch_cpp_ext

    for flag in [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


def load_quarot_sm120_extension(build_dir: str | os.PathLike | None = None, verbose: bool = False):
    build_path = Path(build_dir) if build_dir is not None else DEFAULT_BUILD_DIR
    build_path.mkdir(parents=True, exist_ok=True)
    _remove_unwanted_pytorch_nvcc_flags()

    include_dirs = [
        str(ROOT / "quarot/kernels/include"),
        str(ROOT / "third_party/cutlass/include"),
        str(ROOT / "third_party/cutlass/tools/util/include"),
    ]
    sources = [
        str(ROOT / "quarot/kernels/bindings.cpp"),
        str(ROOT / "quarot/kernels/gemm.cu"),
        str(ROOT / "quarot/kernels/quant.cu"),
        str(ROOT / "quarot/kernels/flashinfer.cu"),
    ]
    extra_cuda_cflags = [
        "-O3",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_120,code=sm_120",
    ]
    ext = load(
        name="quarot_sm120_cuda_v1",
        sources=sources,
        extra_include_paths=include_dirs,
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=str(build_path),
        verbose=verbose,
        with_cuda=True,
    )

    import quarot

    quarot._CUDA = ext
    return ext
