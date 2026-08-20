import inspect
import json
import sys
from pathlib import Path

ROOT = Path("/data/yzy/quarot-gpt-2").resolve()
for item in (
    ROOT,
    ROOT / "fake_quant",
    ROOT / "kernel_quant",
    ROOT / "kernel_quant/scripts",
):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import kernel_quant.scripts.bench_real_split_fullstack_v1 as B

out = []

for name in dir(B):
    low = name.lower()
    if any(k in low for k in [
        "ext", "compile", "load", "build", "import",
        "prepare", "gptq", "quant", "hadamard",
        "profile", "timing", "policy", "base"
    ]):
        obj = getattr(B, name)
        item = {
            "name": name,
            "type": str(type(obj)),
        }
        if callable(obj):
            try:
                item["signature"] = str(inspect.signature(obj))
            except Exception as e:
                item["signature"] = f"<signature error: {e!r}>"
            try:
                src = inspect.getsource(obj)
                item["source_head"] = "\n".join(src.splitlines()[:80])
            except Exception as e:
                item["source_head"] = f"<source error: {e!r}>"
        else:
            item["repr"] = repr(obj)[:500]
        out.append(item)

path = Path("/data/yzy/quarot-gpt-2/experiments/kernel_quant/layer_latency_split_v1/reports/real_split_symbols_v1.json")
path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(path, "w"), indent=2, ensure_ascii=False)

print("[OUT]", path)
for x in out:
    print("\n==", x["name"], "==")
    print("type:", x["type"])
    if "signature" in x:
        print("signature:", x["signature"])
