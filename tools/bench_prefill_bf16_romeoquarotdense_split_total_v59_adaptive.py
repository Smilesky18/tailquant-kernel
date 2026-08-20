"""Total layer latency benchmark with runtime adaptive split/dense backend.

Base path:
  * v57 threshold top-R prepare
  * no up-type weight pre-rotation
  * v58 oct fused dense-scale + sparse-correction epilogue

Adaptive policy:
  Tiny split modules with R <= ADAPTIVE_MAX_R are run through the existing pure
  W4A4 dense backend. This avoids fixed sparse-correction overhead for very
  small tails and keeps the original policy file unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("KQ_PROJECT_ROOT", "/data/yzy/quarot-gpt-2")).resolve()
TOOLS = ROOT / "experiments/kernel_quant/layer_latency_split_v1/tools"
for p in (TOOLS, ROOT, ROOT / "fake_quant", ROOT / "kernel_quant"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import bench_prefill_bf16_romeoquarotdense_split_total_v53 as BASE  # noqa: E402
import bench_qwen3_l18_split_opts_v57 as V57  # noqa: E402
import bench_qwen3_l18_split_opts_v58 as V58  # noqa: E402
import bench_hadamard_three_schemes_quarot_dense_sharedqkv_inplace_fusedtopr_v43 as V43  # noqa: E402
from fused_sparse_epilogue_ext_v58 import load_fused_sparse_epilogue_ext  # noqa: E402
from fused_threshold_topr_pack_ext_v57 import load_threshold_topr_pack_ext  # noqa: E402
import types  # noqa: E402


_ORIGINAL_BUILD_SPLIT_LAYER = BASE.build_split_layer
_EPILOGUE_EXT = None
_ADAPTIVE_MAX_R = int(os.environ.get("ADAPTIVE_MAX_R", "32"))


def load_fused_topr_pack_ext(verbose: bool = False):
    V57.install_threshold_topr_prepare_v57()
    ext = load_threshold_topr_pack_ext(verbose=verbose)
    BASE.log("[V59_ADAPTIVE_THRESHOLD_TOPR] v57 K in {4096,12288}, R<=512")
    return ext


def patch_runtime_adaptive_dense(layer, qext, max_r: int):
    patched = []
    for name, mod in layer.named_modules():
        if not V43.is_real_policy_linear(mod):
            continue
        if not bool(getattr(mod, "is_split", False)):
            continue
        r = int(getattr(mod, "R", 0))
        if r <= 0 or r > max_r:
            continue
        if not hasattr(mod, "B_col"):
            continue

        def make_dense_only(module_name):
            def dense_only_split_compute(self, A, scratch, B_col, dense_ready_event, dense_stream, sparse_stream):
                M = int(A.shape[0])
                ext = getattr(self, "ext", None)
                if ext is None:
                    ext = getattr(self, "main_ext", None)
                if ext is None:
                    raise RuntimeError(f"{module_name}: cannot find ext/main_ext")

                # Reuse split scratch buffers: body_scale plays the role of pure a_scale.
                ext.pack_a_full_s4(A, scratch["A_pack"], scratch["body_scale"], self.eps)
                C = V43.quarot_dense_gemm(qext, scratch["A_pack"], B_col, M, self.N, self.K)
                ext.scale_i32_to_fp16(C, scratch["body_scale"], self.w_scale, scratch["Y_body"])
                return scratch["Y_body"]

            return dense_only_split_compute

        mod._split_compute = types.MethodType(make_dense_only(name), mod)
        mod._runtime_adaptive_dense_v59 = True
        patched.append({
            "name": name,
            "K": int(getattr(mod, "K", 0)),
            "N": int(getattr(mod, "N", 0)),
            "R": r,
            "ratio": float(getattr(mod, "ratio", 0.0)),
        })
    BASE.log("[V59_RUNTIME_ADAPTIVE_DENSE] " + __import__("json").dumps({
        "max_r": max_r,
        "patched_modules": patched,
    }, indent=2))
    return patched


def build_split_layer_adaptive(
    base_layer,
    layer_idx,
    policy,
    rot_flags,
    B,
    main_ext,
    layout_ext,
    policy_pack_ext,
    eps,
    device,
    qext,
    fused_topr_ext,
    qwen_shared: bool,
):
    layer, rec = _ORIGINAL_BUILD_SPLIT_LAYER(
        base_layer,
        layer_idx,
        policy,
        rot_flags,
        B,
        main_ext,
        layout_ext,
        policy_pack_ext,
        eps,
        device,
        qext,
        fused_topr_ext,
        qwen_shared,
    )
    if _EPILOGUE_EXT is not None:
        patched = V58.patch_fused_sparse_epilogue(layer, qext, _EPILOGUE_EXT, "oct")
        BASE.log(f"[V59_ADAPTIVE_EPILOGUE_OCT] layer={layer_idx} modules={len(patched)}")
    patch_runtime_adaptive_dense(layer, qext, _ADAPTIVE_MAX_R)
    return layer, rec


def main():
    global _EPILOGUE_EXT
    _EPILOGUE_EXT = load_fused_sparse_epilogue_ext(verbose=bool(int(os.environ.get("FUSED_EPILOGUE_VERBOSE", "0"))))
    BASE.load_fused_topr_pack_ext = load_fused_topr_pack_ext
    BASE.build_split_layer = build_split_layer_adaptive
    BASE.log(f"[BENCH_TOTAL_V59_ADAPTIVE] threshold_v57_epilogue_oct + runtime dense for R<={_ADAPTIVE_MAX_R}.")
    BASE.main()


if __name__ == "__main__":
    main()
