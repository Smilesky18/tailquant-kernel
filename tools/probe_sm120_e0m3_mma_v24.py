import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path


CUDA_SRC = r"""
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef FORMAT_A
#define FORMAT_A e2m1
#endif

#ifndef FORMAT_B
#define FORMAT_B e2m1
#endif

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

#define CUDA_CHECK(expr)                                                     \
  do {                                                                       \
    cudaError_t status_ = (expr);                                             \
    if (status_ != cudaSuccess) {                                             \
      std::fprintf(stderr, "%s failed: %s\n", #expr,                        \
                   cudaGetErrorString(status_));                              \
      return EXIT_FAILURE;                                                    \
    }                                                                        \
  } while (0)

__device__ __forceinline__ void sm120_mma_f4(
    float &d0, float &d1, float &d2, float &d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb) {
  constexpr uint16_t bid_a = 0;
  constexpr uint16_t tid_a = 0;
  constexpr uint16_t bid_b = 0;
  constexpr uint16_t tid_b = 0;

  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
      "m16n8k64.row.col.f32." STR(FORMAT_A) "." STR(FORMAT_B)
      ".f32.ue4m3 "
      "{%0, %1, %2, %3},"
      "{%4, %5, %6, %7},"
      "{%8, %9},"
      "{%10, %11, %12, %13},"
      "{%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
        "r"(b0), "r"(b1),
        "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
        "r"(sfa), "h"(bid_a), "h"(tid_a),
        "r"(sfb), "h"(bid_b), "h"(tid_b));
}

__global__ void one_mma_kernel(uint32_t packed_a, uint32_t packed_b,
                               uint32_t scale_a, uint32_t scale_b,
                               float *out) {
  float d0 = 0.0f;
  float d1 = 0.0f;
  float d2 = 0.0f;
  float d3 = 0.0f;
  sm120_mma_f4(d0, d1, d2, d3,
               packed_a, packed_a, packed_a, packed_a,
               packed_b, packed_b,
               scale_a, scale_b);
  int offset = int(threadIdx.x) * 4;
  out[offset + 0] = d0;
  out[offset + 1] = d1;
  out[offset + 2] = d2;
  out[offset + 3] = d3;
}

uint32_t repeat_nibble(unsigned value) {
  value &= 0xfu;
  uint32_t packed = 0;
  for (int shift = 0; shift < 32; shift += 4) {
    packed |= value << shift;
  }
  return packed;
}

uint32_t repeat_byte(unsigned value) {
  value &= 0xffu;
  return value | (value << 8) | (value << 16) | (value << 24);
}

int main(int argc, char **argv) {
  unsigned a = argc > 1 ? std::strtoul(argv[1], nullptr, 0) : 7u;
  unsigned b = argc > 2 ? std::strtoul(argv[2], nullptr, 0) : 7u;
  unsigned sa = argc > 3 ? std::strtoul(argv[3], nullptr, 0) : 0x38u;
  unsigned sb = argc > 4 ? std::strtoul(argv[4], nullptr, 0) : 0x38u;

  int device = 0;
  cudaDeviceProp props{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));

  float *dev = nullptr;
  constexpr int kOut = 32 * 4;
  CUDA_CHECK(cudaMalloc(&dev, kOut * sizeof(float)));
  one_mma_kernel<<<1, 32>>>(repeat_nibble(a), repeat_nibble(b),
                            repeat_byte(sa), repeat_byte(sb), dev);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> host(kOut);
  CUDA_CHECK(cudaMemcpy(host.data(), dev, kOut * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(dev));

  auto mm = std::minmax_element(host.begin(), host.end());
  bool all_equal = std::all_of(host.begin(), host.end(),
                               [&](float x) { return x == host.front(); });
  std::printf(
      "gpu=%s sm=%d%d declared=%s_x_%s a=0x%x b=0x%x sa=0x%x sb=0x%x "
      "first=%.9g min=%.9g max=%.9g all_equal=%d\n",
      props.name, props.major, props.minor, STR(FORMAT_A), STR(FORMAT_B),
      a, b, sa, sb, host[0], *mm.first, *mm.second, int(all_equal));
  return all_equal ? EXIT_SUCCESS : EXIT_FAILURE;
}
"""


FORMAT_BITS = {
    "e0m3_e2m1": 0b01,
    "e2m1_e0m3": 0b10,
    "e0m3_e0m3": 0b11,
}


def run(cmd, cwd=None):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
    }


def fail_if_bad(result):
    if result["returncode"] != 0:
        raise RuntimeError(
            "command failed rc={}\ncmd={}\n{}".format(
                result["returncode"],
                " ".join(result["cmd"]),
                result["stdout"],
            )
        )


def parse_first_value(text: str) -> float:
    m = re.search(r"\bfirst=([-+0-9.eE]+)", text)
    if not m:
        raise RuntimeError(f"cannot parse first= from output:\n{text}")
    return float(m.group(1))


def patch_binary(baseline: Path, cuobjdump: Path, out_path: Path, format_name: str):
    sass = run([str(cuobjdump), "--dump-sass", str(baseline)])
    fail_if_bad(sass)
    pattern = re.compile(
        r"OMMA\.SF\.16864\.F32\.E2M1\.E2M1\.UE4M3\.4X[^\n]*"
        r"/\* 0x([0-9a-fA-F]{16}) \*/\s*\n\s*"
        r"/\* 0x([0-9a-fA-F]{16}) \*/"
    )
    encodings = sorted({(int(a, 16), int(b, 16)) for a, b in pattern.findall(sass["stdout"])})
    if not encodings:
        raise RuntimeError("cuobjdump did not find an E2M1 x E2M1 OMMA encoding")

    data = baseline.read_bytes()
    patched = data
    replacements = 0
    patched_words = []
    bits = FORMAT_BITS[format_name]
    for first_word, second_word in encodings:
        if ((second_word >> 14) & 0b11) != 0:
            raise RuntimeError(f"unexpected nonzero format bits in {second_word:#x}")
        new_second = second_word | (bits << 14)
        needle = struct.pack("<QQ", first_word, second_word)
        repl = struct.pack("<QQ", first_word, new_second)
        count = patched.count(needle)
        if count == 0:
            raise RuntimeError(f"cannot find instruction bytes for {first_word:#x} {second_word:#x}")
        patched = patched.replace(needle, repl)
        replacements += count
        patched_words.append(f"0x{new_second:016x}")

    out_path.write_bytes(patched)
    os.chmod(out_path, baseline.stat().st_mode)
    patched_sass = run([str(cuobjdump), "--dump-sass", str(out_path)])
    fail_if_bad(patched_sass)
    return {
        "format": format_name,
        "replacements": replacements,
        "patched_words": patched_words,
        "baseline_encodings": [[f"0x{a:016x}", f"0x{b:016x}"] for a, b in encodings],
        "sass_head": "\n".join(
            line
            for line in patched_sass["stdout"].splitlines()
            if "OMMA.SF.16864" in line
        )[:2000],
    }


def expected_e2m1(nibble: int) -> float:
    table = {
        0x0: 0.0,
        0x1: 0.5,
        0x2: 1.0,
        0x3: 1.5,
        0x4: 2.0,
        0x5: 3.0,
        0x6: 4.0,
        0x7: 6.0,
        0x8: -0.0,
        0x9: -0.5,
        0xA: -1.0,
        0xB: -1.5,
        0xC: -2.0,
        0xD: -3.0,
        0xE: -4.0,
        0xF: -6.0,
    }
    return table[nibble & 0xF]


def expected_e0m3(nibble: int) -> float:
    nibble &= 0xF
    sign = -1.0 if (nibble & 0x8) else 1.0
    return sign * float(nibble & 0x7)


def expected_dot(format_name: str, a: int, b: int) -> float:
    if format_name == "e2m1_e2m1":
        av = expected_e2m1(a)
        bv = expected_e2m1(b)
    elif format_name == "e0m3_e2m1":
        av = expected_e0m3(a)
        bv = expected_e2m1(b)
    elif format_name == "e2m1_e0m3":
        av = expected_e2m1(a)
        bv = expected_e0m3(b)
    elif format_name == "e0m3_e0m3":
        av = expected_e0m3(a)
        bv = expected_e0m3(b)
    else:
        raise ValueError(format_name)
    return 64.0 * av * bv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cuda_home", default=os.environ.get("CUDA_HOME", "/usr/local/cuda"))
    parser.add_argument("--arch", default="120a")
    parser.add_argument("--a", type=lambda x: int(x, 0), default=0x7)
    parser.add_argument("--b", type=lambda x: int(x, 0), default=0x7)
    parser.add_argument("--formats", default="e0m3_e0m3,e0m3_e2m1,e2m1_e0m3")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    build_dir = out_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    nvcc = Path(args.cuda_home) / "bin" / "nvcc"
    cuobjdump = Path(args.cuda_home) / "bin" / "cuobjdump"
    if not nvcc.exists():
        raise FileNotFoundError(nvcc)
    if not cuobjdump.exists():
        raise FileNotFoundError(cuobjdump)

    src = build_dir / "sm120_e2m1_probe.cu"
    baseline = build_dir / "sm120_e2m1_probe"
    src.write_text(CUDA_SRC, encoding="utf-8")

    print(f"[START] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print(f"[NVCC] {nvcc}", flush=True)
    print(f"[CUOBJDUMP] {cuobjdump}", flush=True)
    print(f"[ARCH] {args.arch}", flush=True)

    compile_result = run(
        [
            str(nvcc),
            "-O3",
            f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
            str(src),
            "-o",
            str(baseline),
        ]
    )
    fail_if_bad(compile_result)

    records = []
    baseline_run = run([str(baseline), hex(args.a), hex(args.b)])
    fail_if_bad(baseline_run)
    baseline_value = parse_first_value(baseline_run["stdout"])
    records.append(
        {
            "format": "e2m1_e2m1",
            "binary": str(baseline),
            "stdout": baseline_run["stdout"].strip(),
            "value": baseline_value,
            "expected": expected_dot("e2m1_e2m1", args.a, args.b),
            "abs_error": abs(baseline_value - expected_dot("e2m1_e2m1", args.a, args.b)),
        }
    )

    patch_records = []
    for fmt in [x.strip() for x in args.formats.split(",") if x.strip()]:
        patched = build_dir / f"sm120_{fmt}_probe"
        patch_records.append(patch_binary(baseline, cuobjdump, patched, fmt))
        run_result = run([str(patched), hex(args.a), hex(args.b)])
        fail_if_bad(run_result)
        value = parse_first_value(run_result["stdout"])
        exp = expected_dot(fmt, args.a, args.b)
        records.append(
            {
                "format": fmt,
                "binary": str(patched),
                "stdout": run_result["stdout"].strip(),
                "value": value,
                "expected": exp,
                "abs_error": abs(value - exp),
            }
        )

    ok = all(r["abs_error"] == 0.0 for r in records)
    report = {
        "ok": ok,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "arch": args.arch,
        "a": args.a,
        "b": args.b,
        "compile": compile_result,
        "patch_records": patch_records,
        "records": records,
        "note": "This validates SM120 E0M3 MMA format decoding only; it is not a GEMM kernel.",
    }
    report_path = out_dir / "sm120_e0m3_mma_probe_v24.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": ok, "records": records}, indent=2), flush=True)
    print(f"[REPORT] {report_path}", flush=True)
    print(f"[END] {time.strftime('%Y-%m-%dT%H:%M:%S%z')} rc={0 if ok else 2}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
