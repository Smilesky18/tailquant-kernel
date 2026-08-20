import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ratio", type=float, required=True)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    obj = json.load(open(src))

    if "modules" not in obj or not isinstance(obj["modules"], dict):
        raise RuntimeError(f"Invalid policy: no modules dict in {src}")

    ratio = float(args.ratio)
    for name, cfg in obj["modules"].items():
        cfg["ratio"] = ratio
        cfg["ratio_continuous"] = ratio

        # 保守补齐字段，避免后续代码读取时报 KeyError。
        cfg.setdefault("activation_percentile", 100.0)
        cfg.setdefault("weight_percentile", 100.0)

    meta = obj.setdefault("meta", {})
    meta["fixed_ratio_for_layer_latency_benchmark"] = ratio
    meta["source_policy"] = str(src)
    meta["note"] = "Generated only for BF16-vs-Split layer-level latency benchmark; not an accuracy-search policy."

    dst.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(dst, "w"), indent=2, ensure_ascii=False)

    hist = {}
    for cfg in obj["modules"].values():
        r = float(cfg.get("ratio", 0.0))
        hist[str(r)] = hist.get(str(r), 0) + 1

    print("[SRC]", src)
    print("[DST]", dst)
    print("[MODULES]", len(obj["modules"]))
    print("[RATIO_HIST]", hist)

if __name__ == "__main__":
    main()
