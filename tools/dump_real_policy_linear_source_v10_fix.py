import inspect
import json
from pathlib import Path

import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

out = {}

def add_obj(name, obj):
    item = {"name": name, "type": str(type(obj))}
    try:
        item["signature"] = str(inspect.signature(obj))
    except Exception as e:
        item["signature"] = f"<signature error: {e!r}>"
    try:
        item["source"] = inspect.getsource(obj)
    except Exception as e:
        item["source"] = f"<source error: {e!r}>"
    out[name] = item

# bench_real_split_fullstack_v1.py 里的类和函数
for name in [
    "RealPolicyLinear",
    "patch_real_linears",
    "load_policy_pack_ext",
    "benchmark_first_layer_profiles",
    "run_first_layer",
    "graph_time_ms",
    "eager_time_ms",
]:
    obj = getattr(B, name, None)
    if obj is not None:
        add_obj(name, obj)

BASE = getattr(B, "BASE", None)
add_obj("B.BASE", BASE)

# BASE 模块里最关键的类/函数
if BASE is not None:
    for name in [
        "PolicyW4A4Linear",
        "SharedScratchPool",
        "PrefetchWorkspace",
        "ceil_ratio_count",
        "load_ext",
        "load_layout_ext",
        "find_cutlass_path",
    ]:
        obj = getattr(BASE, name, None)
        if obj is not None:
            add_obj("BASE." + name, obj)

# 递归把 BASE 里名字像 forward/pack/gemm/policy/scratch 的符号也抓出来
if BASE is not None:
    for name in dir(BASE):
        low = name.lower()
        if any(k in low for k in [
            "forward", "pack", "gemm", "policy", "scratch", "workspace",
            "quant", "top", "linear", "layout", "profile"
        ]):
            obj = getattr(BASE, name, None)
            if obj is not None and callable(obj):
                key = "BASE." + name
                if key not in out:
                    add_obj(key, obj)

report_dir = Path("/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/reports")
report_dir.mkdir(parents=True, exist_ok=True)

json_path = report_dir / "real_policy_linear_source_v10_fix.json"
txt_path = report_dir / "real_policy_linear_source_v10_fix.txt"

json.dump(out, open(json_path, "w"), indent=2, ensure_ascii=False)

with open(txt_path, "w", encoding="utf-8") as f:
    for k, v in out.items():
        f.write("\n" + "=" * 120 + "\n")
        f.write(k + "\n")
        f.write("=" * 120 + "\n")
        f.write(v.get("signature", "") + "\n\n")
        f.write(v.get("source", "") + "\n")

print("[JSON]", json_path)
print("[TXT]", txt_path)
print("[OBJECTS]", len(out))
for k in out:
    print("[OBJ]", k)
