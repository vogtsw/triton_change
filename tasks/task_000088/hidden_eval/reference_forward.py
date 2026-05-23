"""PyTorch reference forward for task_000088."""
from __future__ import annotations

import torch
import torch.nn.functional as F

HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 3584
LN_EPS = 1e-05
DTYPE_NAME = "float16"

def _compute_dtype() -> torch.dtype:
    return getattr(torch, DTYPE_NAME)

def reference_forward(
    x: torch.Tensor,
    ln_w: torch.Tensor,
    ln_b: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    dtype = _compute_dtype()
    x = x.to(dtype)
    ln_w = ln_w.to(dtype)
    ln_b = ln_b.to(dtype)
    w1 = w1.to(dtype)
    b1 = b1.to(dtype)
    w2 = w2.to(dtype)
    b2 = b2.to(dtype)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + LN_EPS)
    h = x * rms * ln_w.to(dtype)
    h = F.linear(h, w1, b1)
    h = F.gelu(h, approximate="tanh")
    h = F.linear(h, w2, b2)
    return h.to(dtype)
