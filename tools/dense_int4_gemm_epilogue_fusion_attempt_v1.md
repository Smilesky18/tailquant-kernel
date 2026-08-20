# Dense INT4 GEMM Sparse-Epilogue Fusion Attempt v1

## Goal

Try to fuse the v58 Split sparse correction into the dense INT4 GEMM epilogue:

```text
int4 tensor-core GEMM accumulator
  -> dense dequant: C_frag * body_scale[m] * w_scale[n]
  -> sparse correction: sum_r top_q[m, r] * B_row_pack[n, idx[m, r]]
  -> fp16 output
```

This would remove the current int32 `C` global-memory write/read between
`quarot_dense_gemm()` and `scale_sparse_epilogue_oct()`.

## Current Code Path

The current dense GEMM entry is:

```text
quarot/kernels/bindings.cpp::matmul()
  -> quarot/kernels/gemm.cu::matmul_host()
  -> cutlass::gemm::device::Gemm<..., int32_t output, int32_t accumulator>
```

The current Split fused epilogue is separate:

```text
tools/fused_sparse_epilogue_ext_v58.py
  -> scale_sparse_epilogue_quad/oct()
```

## Finding

The current `cutlass::gemm::device::Gemm` instantiation only exposes the
standard output-op path. That epilogue can scale/store accumulator fragments,
but it does not have access to the extra per-row/per-column Split state:

```text
top_q[m, r]
idx[m, r]
top_scale[m]
body_scale[m]
w_scale[n]
B_row_pack[n, k/2]
```

Because the sparse correction is indexed by `idx[m, r]` and weight row `n`, it
needs a custom epilogue visitor or a custom GEMM kernel whose epilogue can load
these tensors while the accumulator fragment is still in registers.

## Decision

I did not modify or replace the existing `quarot/kernels/*.cu` files, and I did
not wire a fake "fusion" that still materializes int32 `C`. That would not test
the requested optimization.

The safe next implementation path is:

1. Add new files, e.g. `quarot/kernels/gemm_split_epilogue.cu` and
   `quarot/kernels/include/gemm_split_epilogue.h`.
2. Instantiate a CUTLASS kernel with a custom epilogue/visitor that accepts
   `body_scale`, `top_scale`, `w_scale`, `top_q`, `idx`, and `B_row_pack`.
3. Add a separate binding name such as `matmul_split_sparse_epilogue()` so the
   old `matmul()` remains untouched.
4. Patch `RealPolicyLinear._split_compute` only for supported shapes after
   correctness tests against the current v58 epilogue.

This is a backend rewrite, not a Python wrapper optimization. The current
v58 path remains the correct benchmark path for this round.
