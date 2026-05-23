"""Oracle Triton model for task_000071 (float16)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

HIDDEN_SIZE = 896
INTERMEDIATE_SIZE = 3584
LN_EPS = 1e-05
GELU_BLOCK_SIZE = 1024
DTYPE_NAME = "float16"

def _compute_dtype() -> torch.dtype:
    return getattr(torch, DTYPE_NAME)

@triton.jit
def layernorm_fwd_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    stride_xm,
    N: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X_ptr + row * stride_xm + cols
    y_ptrs = Y_ptr + row * stride_xm + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    y = xc * rstd * w + b
    tl.store(y_ptrs, y, mask=mask)

@triton.jit
def gelu_act_kernel(
    X_ptr, Y_ptr, N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X_ptr + offsets, mask=mask).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    e2x = tl.exp(2.0 * inner)
    tanh_inner = (e2x - 1.0) / (e2x + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    tl.store(Y_ptr + offsets, y, mask=mask)

def _layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] == HIDDEN_SIZE
    M, N = x.shape
    y = torch.empty_like(x)
    block_size = triton.next_power_of_2(N)
    layernorm_fwd_kernel[(M,)](
        x, y, weight, bias,
        x.stride(0), N, LN_EPS,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return y

def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, GELU_BLOCK_SIZE),)
    gelu_act_kernel[grid](
        x, y, n,
        BLOCK_SIZE=GELU_BLOCK_SIZE,
        num_warps=4,
    )
    return y

def _gelu_bias(x: torch.Tensor) -> torch.Tensor:
    return _gelu_tanh(x)

def _gelu_relu(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, GELU_BLOCK_SIZE),)
    gelu_act_kernel[grid](
        x, y, n,
        BLOCK_SIZE=GELU_BLOCK_SIZE,
        num_warps=4,
    )
    return y

def model_forward(
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
    B, S, H = x.shape
    assert H == HIDDEN_SIZE, f"input hidden mismatch: {H} vs {HIDDEN_SIZE}"
    x_flat = x.reshape(B * S, H).contiguous()
    h = _layernorm(x_flat, ln_w, ln_b)
    h = torch.nn.functional.linear(h, w1, b1)
    assert h.shape[-1] == INTERMEDIATE_SIZE
    h = _gelu_tanh(h)
    h = torch.nn.functional.linear(h, w2, b2)
    assert h.shape[-1] == HIDDEN_SIZE
    return h.reshape(B, S, H).to(dtype)
