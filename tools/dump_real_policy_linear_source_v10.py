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

add_obj("RealPolicyLinear", B.RealPolicyLinear)

BASE = B.BASE
for name in [
    "PolicyW4A4Linear",
    "SharedScratchPool",
    "PrefetchWorkspace",
    "ceil_ratio_count",
    "load_ext",
    "load_layout_ext",
]:
    obj = getattr(BASE, name, None)
    if obj is not None:
        add_obj("BASE." + name, obj)

for name in [
    "load_policy_pack_ext",
    "patch_real_linears",
    "benchmark_first_layer_profiles",
    "graph_time_ms",
    "eager_time_ms",
]:
    obj = getattr(B, name, None)
    if obj is not None:
        add_obj(name, obj)

path = Path("/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/reports/real_policy_linear_source_v10.json")
path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)

txt = Path("/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/reports/real_policy_linear_source_v10.txt")
with open(txt, "w", encoding="utf-8") as f:
    for k, v in out.items():
        f.write("\n" + "=" * 120 + "\n")
        f.write(k + "\n")
        f.write("=" * 120 + "\n")
        f.write(v.get("signature", "") + "\n\n")
        f.write(v.get("source", "") + "\n")

print("[JSON]", path)
print("[TXT]", txt)
