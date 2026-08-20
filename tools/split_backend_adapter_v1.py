import inspect
import importlib
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


CANDIDATE_MODULES = [
    "kernel_quant.runtime.split_real_fullstack_v1",
    "kernel_quant.runtime",
    "kernel_quant.split_real_fullstack_v1",
    "kernel_quant.ops.split_real_fullstack_v1",
    "kernel_quant.scripts.bench_real_split_fullstack_v1",
    "split_real_fullstack_v1",
]


def _callable_score(name: str) -> int:
    s = name.lower()
    score = 0
    if "split" in s:
        score += 10
    if "linear" in s:
        score += 6
    if "gemm" in s or "matmul" in s:
        score += 5
    if "forward" in s:
        score += 4
    if "run" in s:
        score += 1
    if s.startswith("_"):
        score -= 10
    return score


def discover_split_backends() -> List[Dict[str, str]]:
    found: List[Dict[str, str]] = []
    for mod_name in CANDIDATE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            found.append({
                "module": mod_name,
                "status": "import_failed",
                "error": repr(e),
                "callable": "",
                "signature": "",
            })
            continue

        entries = []
        for name in dir(mod):
            obj = getattr(mod, name)
            if callable(obj):
                score = _callable_score(name)
                if score > 0:
                    try:
                        sig = str(inspect.signature(obj))
                    except Exception:
                        sig = "<signature unavailable>"
                    entries.append((score, name, sig))

        if not entries:
            found.append({
                "module": mod_name,
                "status": "import_ok_no_candidate_callable",
                "error": "",
                "callable": "",
                "signature": "",
            })
            continue

        entries.sort(reverse=True)
        for score, name, sig in entries[:20]:
            found.append({
                "module": mod_name,
                "status": "candidate",
                "error": "",
                "callable": name,
                "signature": sig,
            })

    return found


def _try_call(fn: Callable[..., Any], x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], ratio: float):
    # 这里做多种保守尝试，避免强绑定现有 kernel API。
    # 如果你的 real split kernel API 和下面都不匹配，脚本会在 report 里列出候选函数签名。
    attempts = [
        lambda: fn(x, weight, bias=bias, ratio=ratio),
        lambda: fn(x, weight, bias=bias, split_ratio=ratio),
        lambda: fn(x, weight, bias, ratio),
        lambda: fn(x, weight, ratio),
        lambda: fn(x, weight),
        lambda: fn(x, weight.t().contiguous(), bias=bias, ratio=ratio),
        lambda: fn(x, weight.t().contiguous(), bias=bias, split_ratio=ratio),
        lambda: fn(x, weight.t().contiguous(), bias, ratio),
        lambda: fn(x, weight.t().contiguous(), ratio),
        lambda: fn(x, weight.t().contiguous()),
    ]

    last_err = None
    for attempt in attempts:
        try:
            y = attempt()
            if isinstance(y, tuple):
                y = y[0]
            if torch.is_tensor(y):
                return y
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Unable to call discovered split backend. Last error: {repr(last_err)}")


class SplitLinearAdapter(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        ratio: float = 0.05,
        backend_module: Optional[str] = None,
        backend_callable: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.ratio = float(ratio)

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)

        self.backend_module = backend_module
        self.backend_callable = backend_callable
        self._fn: Optional[Callable[..., Any]] = None
        self._backend_error: Optional[str] = None

        self._load_backend()

    def _load_backend(self):
        if self.backend_module and self.backend_callable:
            try:
                mod = importlib.import_module(self.backend_module)
                fn = getattr(mod, self.backend_callable)
                if not callable(fn):
                    raise TypeError(f"{self.backend_module}.{self.backend_callable} is not callable")
                self._fn = fn
                return
            except Exception as e:
                self._backend_error = repr(e)
                return

        # 自动找一个最像 split linear/gemm 的函数。
        candidates = discover_split_backends()
        for c in candidates:
            if c.get("status") != "candidate":
                continue
            name = c["callable"].lower()
            if "split" not in name:
                continue
            try:
                mod = importlib.import_module(c["module"])
                fn = getattr(mod, c["callable"])
                if callable(fn):
                    self.backend_module = c["module"]
                    self.backend_callable = c["callable"]
                    self._fn = fn
                    return
            except Exception as e:
                self._backend_error = repr(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()

        if self._fn is None:
            raise RuntimeError(
                "No usable real split backend found. "
                "Run with --mode discover to print candidates, then bind --split_backend_module and --split_backend_callable. "
                f"backend_error={self._backend_error}"
            )

        y = _try_call(self._fn, x2d, self.weight, self.bias, self.ratio)
        return y.reshape(*orig_shape[:-1], self.out_features)


def replace_linear_with_split(module: nn.Module, ratio: float, backend_module: Optional[str], backend_callable: Optional[str]):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, SplitLinearAdapter(child, ratio=ratio, backend_module=backend_module, backend_callable=backend_callable))
        else:
            replace_linear_with_split(child, ratio, backend_module, backend_callable)
    return module
