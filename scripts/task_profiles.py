"""Task profile templates for generate_tasks.py.

Each profile renders:
  old_model_triton.py, oracle/new_model_triton.py, cpu_demo, reference_forward,
  and the surgical oracle patch_ops list.

Profiles map to Phase 3a categories in the spec.
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RenderedTask:
    old_triton: str
    new_triton: str
    cpu_demo: str
    reference: str
    patch_ops: list[dict[str, Any]]


def _p(cfg: Any, side: str, key: str, default: Any = None) -> Any:
    params = cfg.from_params if side == "from" else cfg.to_params
    if key in params:
        return params[key]
    if default is not None:
        return default
    other = cfg.to_params if side == "from" else cfg.from_params
    return other.get(key, default)


def _hidden(cfg, side) -> int:
    return int(_p(cfg, side, "hidden_size", 768))


def _inter(cfg, side) -> int:
    return int(_p(cfg, side, "intermediate_size", 3072))


def _eps(cfg, side) -> float:
    return float(_p(cfg, side, "ln_eps", 1e-5))


def _seq(cfg, side) -> int:
    return int(_p(cfg, side, "seq_len", 128))


def _batch(cfg, side) -> int:
    return int(_p(cfg, side, "batch_size", 1))


def _dtype_name(cfg, side) -> str:
    return str(_p(cfg, side, "dtype_name", "float32"))


def _patch_const(name: str, old, new, reason: str) -> dict[str, Any]:
    op: dict[str, Any] = {
        "operation": "update_constant",
        "path": "candidate_model_triton.py",
        "constant_name": name,
        "new_value": new,
        "reason": reason,
    }
    if old is not None:
        op["old_value"] = old
    return op


def _extract_kernel_body(source: str, kernel_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == kernel_name:
            lines = source.splitlines(keepends=True)
            body_start = node.body[0].lineno
            body_end = node.body[-1].end_lineno or node.body[-1].lineno
            body_lines = lines[body_start - 1 : body_end]
            return textwrap.dedent("".join(body_lines))
    raise ValueError(f"kernel {kernel_name} not found")


def _extract_function(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = node.lineno
            if node.decorator_list:
                start = node.decorator_list[0].lineno
            end = node.end_lineno or node.lineno
            lines = source.splitlines(keepends=True)
            return "".join(lines[start - 1 : end])
    raise ValueError(f"function {name} not found")


# ---------- Shared Triton building blocks ----------


_GELU_TANH_BODY = textwrap.dedent("""\
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X_ptr + offsets, mask=mask).to(tl.float32)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    e2x = tl.exp(2.0 * inner)
    tanh_inner = (e2x - 1.0) / (e2x + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    tl.store(Y_ptr + offsets, y, mask=mask)
""")


_GELU_BIAS_BODY = textwrap.dedent("""\
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X_ptr + offsets, mask=mask).to(tl.float32)
    x = x + GELU_IN_BIAS
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    e2x = tl.exp(2.0 * inner)
    tanh_inner = (e2x - 1.0) / (e2x + 1.0)
    y = 0.5 * x * (1.0 + tanh_inner)
    tl.store(Y_ptr + offsets, y, mask=mask)
""")


_RELU_BODY = textwrap.dedent("""\
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    x = tl.load(X_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offsets, y, mask=mask)
""")


_LN_BODY = textwrap.dedent("""\
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
""")


_RMS_BODY = textwrap.dedent("""\
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_ptrs = X_ptr + row * stride_xm + cols
    y_ptrs = Y_ptr + row * stride_xm + cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = x * rstd * w
    tl.store(y_ptrs, y, mask=mask)
""")


def _finalize_source(text: str) -> str:
    out = textwrap.dedent(text).strip() + "\n"
    ast.parse(out)
    return out


def _triton_file(
    *,
    doc: str,
    hidden: int,
    inter: int,
    eps: float,
    seq_len: int | None = None,
    fixed_batch: int | None = None,
    dtype_name: str = "float32",
    gelu_body: str = _GELU_TANH_BODY,
    gelu_in_bias: float | None = None,
    ln_body: str = _LN_BODY,
    activation: str = "gelu_tanh",
) -> str:
    lines: list[str] = [
        f'"""{doc}"""',
        "from __future__ import annotations",
        "",
        "import torch",
        "import triton",
        "import triton.language as tl",
        "",
        f"HIDDEN_SIZE = {hidden}",
        f"INTERMEDIATE_SIZE = {inter}",
        f"LN_EPS = {eps!r}",
        "GELU_BLOCK_SIZE = 1024",
    ]
    if seq_len is not None:
        lines.append(f"SEQ_LEN = {seq_len}")
    if fixed_batch is not None:
        lines.append(f"FIXED_BATCH = {fixed_batch}")
    if gelu_in_bias is not None:
        lines.append(f"GELU_IN_BIAS = {gelu_in_bias}")
    lines.append(f'DTYPE_NAME = "{dtype_name}"')
    lines += [
        "",
        "def _compute_dtype() -> torch.dtype:",
        "    return getattr(torch, DTYPE_NAME)",
        "",
        "@triton.jit",
        "def layernorm_fwd_kernel(",
        "    X_ptr, Y_ptr, W_ptr, B_ptr,",
        "    stride_xm,",
        "    N: tl.constexpr,",
        "    eps: tl.constexpr,",
        "    BLOCK_SIZE: tl.constexpr,",
        "):",
    ]
    lines.extend(textwrap.indent(ln_body.strip(), "    ").splitlines())
    lines += [
        "",
        "@triton.jit",
        "def gelu_act_kernel(",
        "    X_ptr, Y_ptr, N_ELEMENTS,",
        "    BLOCK_SIZE: tl.constexpr,",
        "):",
    ]
    lines.extend(textwrap.indent(gelu_body.strip(), "    ").splitlines())
    lines += [
        "",
        "def _layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:",
        "    assert x.shape[-1] == HIDDEN_SIZE",
        "    M, N = x.shape",
        "    y = torch.empty_like(x)",
        "    block_size = triton.next_power_of_2(N)",
        "    layernorm_fwd_kernel[(M,)](",
        "        x, y, weight, bias,",
        "        x.stride(0), N, LN_EPS,",
        "        BLOCK_SIZE=block_size,",
        "        num_warps=4,",
        "    )",
        "    return y",
        "",
        "def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:",
        "    x = x.contiguous()",
        "    y = torch.empty_like(x)",
        "    n = x.numel()",
        "    grid = (triton.cdiv(n, GELU_BLOCK_SIZE),)",
        "    gelu_act_kernel[grid](",
        "        x, y, n,",
        "        BLOCK_SIZE=GELU_BLOCK_SIZE,",
        "        num_warps=4,",
        "    )",
        "    return y",
        "",
        "def _gelu_bias(x: torch.Tensor) -> torch.Tensor:",
        "    return _gelu_tanh(x)",
        "",
        "def _gelu_relu(x: torch.Tensor) -> torch.Tensor:",
        "    x = x.contiguous()",
        "    y = torch.empty_like(x)",
        "    n = x.numel()",
        "    grid = (triton.cdiv(n, GELU_BLOCK_SIZE),)",
        "    gelu_act_kernel[grid](",
        "        x, y, n,",
        "        BLOCK_SIZE=GELU_BLOCK_SIZE,",
        "        num_warps=4,",
        "    )",
        "    return y",
        "",
        "def model_forward(",
        "    x: torch.Tensor,",
        "    ln_w: torch.Tensor,",
        "    ln_b: torch.Tensor,",
        "    w1: torch.Tensor,",
        "    b1: torch.Tensor,",
        "    w2: torch.Tensor,",
        "    b2: torch.Tensor,",
        ") -> torch.Tensor:",
        "    dtype = _compute_dtype()",
        "    x = x.to(dtype)",
        "    ln_w = ln_w.to(dtype)",
        "    ln_b = ln_b.to(dtype)",
        "    w1 = w1.to(dtype)",
        "    b1 = b1.to(dtype)",
        "    w2 = w2.to(dtype)",
        "    b2 = b2.to(dtype)",
        "    B, S, H = x.shape",
        '    assert H == HIDDEN_SIZE, f"input hidden mismatch: {H} vs {HIDDEN_SIZE}"',
    ]
    if seq_len is not None:
        lines.append('    assert S == SEQ_LEN, f"seq len mismatch: {S} vs {SEQ_LEN}"')
    if fixed_batch is not None:
        lines.append('    assert B == FIXED_BATCH, f"batch mismatch: {B} vs {FIXED_BATCH}"')
    gelu_helper = "_gelu_tanh" if activation == "gelu_tanh" else ("_gelu_bias" if activation == "gelu_bias" else "_gelu_relu")
    lines += [
        "    x_flat = x.reshape(B * S, H).contiguous()",
        "    h = _layernorm(x_flat, ln_w, ln_b)",
        "    h = torch.nn.functional.linear(h, w1, b1)",
        "    assert h.shape[-1] == INTERMEDIATE_SIZE",
        f"    h = {gelu_helper}(h)",
        "    h = torch.nn.functional.linear(h, w2, b2)",
        "    assert h.shape[-1] == HIDDEN_SIZE",
        "    return h.reshape(B, S, H).to(dtype)",
    ]
    out = "\n".join(lines) + "\n"
    ast.parse(out)
    return out


def _cpu_demo(
    *,
    task_id: str,
    hidden: int,
    inter: int,
    eps: float,
    seq_len: int | None = None,
    fixed_batch: int | None = None,
    dtype_name: str = "float32",
    activation: str = "gelu_tanh",
    norm: str = "layernorm",
    gelu_in_bias: float = 0.125,
) -> str:
    lines: list[str] = [
        f'"""CPU demo equivalent for {task_id}."""',
        "from __future__ import annotations",
        "",
        "import torch",
        "import torch.nn.functional as F",
        "",
        f"HIDDEN_SIZE = {hidden}",
        f"INTERMEDIATE_SIZE = {inter}",
        f"LN_EPS = {eps!r}",
    ]
    if seq_len is not None:
        lines.append(f"SEQ_LEN = {seq_len}")
    if fixed_batch is not None:
        lines.append(f"FIXED_BATCH = {fixed_batch}")
    if activation == "gelu_bias":
        lines.append(f"GELU_IN_BIAS = {gelu_in_bias}")
    lines.append(f'DTYPE_NAME = "{dtype_name}"')
    lines += [
        "",
        "def _compute_dtype() -> torch.dtype:",
        "    return getattr(torch, DTYPE_NAME)",
        "",
        "def model_forward(",
        "    x: torch.Tensor,",
        "    ln_w: torch.Tensor,",
        "    ln_b: torch.Tensor,",
        "    w1: torch.Tensor,",
        "    b1: torch.Tensor,",
        "    w2: torch.Tensor,",
        "    b2: torch.Tensor,",
        ") -> torch.Tensor:",
        "    dtype = _compute_dtype()",
        "    x = x.to(dtype)",
        "    ln_w = ln_w.to(dtype)",
        "    ln_b = ln_b.to(dtype)",
        "    w1 = w1.to(dtype)",
        "    b1 = b1.to(dtype)",
        "    w2 = w2.to(dtype)",
        "    b2 = b2.to(dtype)",
        "    B, S, H = x.shape",
        "    assert H == HIDDEN_SIZE",
    ]
    if seq_len is not None:
        lines.append("    assert S == SEQ_LEN")
    if fixed_batch is not None:
        lines.append("    assert B == FIXED_BATCH")
    if norm == "rmsnorm":
        lines += [
            "    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + LN_EPS)",
            "    h = x * rms * ln_w.to(dtype)",
        ]
    else:
        lines.append("    h = F.layer_norm(x, (HIDDEN_SIZE,), weight=ln_w, bias=ln_b, eps=LN_EPS)")
    lines += [
        "    h = F.linear(h, w1, b1)",
        "    assert h.shape[-1] == INTERMEDIATE_SIZE",
    ]
    if activation == "relu":
        lines.append("    h = torch.relu(h)")
    elif activation == "gelu_bias":
        lines.append('    h = F.gelu(h + GELU_IN_BIAS, approximate="tanh")')
    else:
        lines.append('    h = F.gelu(h, approximate="tanh")')
    lines += [
        "    h = F.linear(h, w2, b2)",
        "    return h.to(dtype)",
    ]
    out = "\n".join(lines) + "\n"
    ast.parse(out)
    return out


def _reference(
    *,
    task_id: str,
    hidden: int,
    inter: int,
    eps: float,
    dtype_name: str = "float32",
    activation: str = "gelu_tanh",
    norm: str = "layernorm",
    gelu_in_bias: float = 0.0,
) -> str:
    lines: list[str] = [
        f'"""PyTorch reference forward for {task_id}."""',
        "from __future__ import annotations",
        "",
        "import torch",
        "import torch.nn.functional as F",
        "",
        f"HIDDEN_SIZE = {hidden}",
        f"INTERMEDIATE_SIZE = {inter}",
        f"LN_EPS = {eps!r}",
        f'DTYPE_NAME = "{dtype_name}"',
    ]
    if activation == "gelu_bias":
        lines.append(f"GELU_IN_BIAS = {gelu_in_bias}")
    lines += [
        "",
        "def _compute_dtype() -> torch.dtype:",
        "    return getattr(torch, DTYPE_NAME)",
        "",
        "def reference_forward(",
        "    x: torch.Tensor,",
        "    ln_w: torch.Tensor,",
        "    ln_b: torch.Tensor,",
        "    w1: torch.Tensor,",
        "    b1: torch.Tensor,",
        "    w2: torch.Tensor,",
        "    b2: torch.Tensor,",
        ") -> torch.Tensor:",
        "    dtype = _compute_dtype()",
        "    x = x.to(dtype)",
        "    ln_w = ln_w.to(dtype)",
        "    ln_b = ln_b.to(dtype)",
        "    w1 = w1.to(dtype)",
        "    b1 = b1.to(dtype)",
        "    w2 = w2.to(dtype)",
        "    b2 = b2.to(dtype)",
    ]
    if norm == "rmsnorm":
        lines += [
            "    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + LN_EPS)",
            "    h = x * rms * ln_w.to(dtype)",
        ]
    else:
        lines.append("    h = F.layer_norm(x, (HIDDEN_SIZE,), weight=ln_w, bias=ln_b, eps=LN_EPS)")
    lines.append("    h = F.linear(h, w1, b1)")
    if activation == "relu":
        lines.append("    h = torch.relu(h)")
    elif activation == "gelu_bias":
        lines.append('    h = F.gelu(h + GELU_IN_BIAS, approximate="tanh")')
    else:
        lines.append('    h = F.gelu(h, approximate="tanh")')
    lines += [
        "    h = F.linear(h, w2, b2)",
        "    return h.to(dtype)",
    ]
    out = "\n".join(lines) + "\n"
    ast.parse(out)
    return out


# ---------- Profile renderers ----------


def render_shape(cfg: Any) -> RenderedTask:
    old_triton = _triton_file(
        doc=f"Old Triton model for {cfg.task_id}.",
        hidden=_hidden(cfg, "from"), inter=_inter(cfg, "from"), eps=_eps(cfg, "from"),
    )
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id}.",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
    )
    ops = []
    if _hidden(cfg, "from") != _hidden(cfg, "to"):
        ops.append(_patch_const("HIDDEN_SIZE", _hidden(cfg, "from"), _hidden(cfg, "to"),
                                f"hidden { _hidden(cfg,'from')} -> {_hidden(cfg,'to')}"))
    if _inter(cfg, "from") != _inter(cfg, "to"):
        ops.append(_patch_const("INTERMEDIATE_SIZE", _inter(cfg, "from"), _inter(cfg, "to"),
                                f"intermediate {_inter(cfg,'from')} -> {_inter(cfg,'to')}"))
    if _eps(cfg, "from") != _eps(cfg, "to"):
        ops.append(_patch_const("LN_EPS", _eps(cfg, "from"), _eps(cfg, "to"),
                                f"LayerNorm eps {_eps(cfg,'from')} -> {_eps(cfg,'to')}"))
    return RenderedTask(old_triton, new_triton, cpu, ref, ops)


def render_seq_len(cfg: Any) -> RenderedTask:
    old_triton = _triton_file(
        doc=f"Old Triton model for {cfg.task_id} (fixed SEQ_LEN).",
        hidden=_hidden(cfg, "from"), inter=_inter(cfg, "from"), eps=_eps(cfg, "from"),
        seq_len=_seq(cfg, "from"),
    )
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} (updated SEQ_LEN).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        seq_len=_seq(cfg, "to"),
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        seq_len=_seq(cfg, "to"),
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
    )
    ops = [_patch_const("SEQ_LEN", _seq(cfg, "from"), _seq(cfg, "to"),
                        f"seq_len {_seq(cfg,'from')} -> {_seq(cfg,'to')}")]
    return RenderedTask(old_triton, new_triton, cpu, ref, ops)


def render_batch(cfg: Any) -> RenderedTask:
    old_triton = _triton_file(
        doc=f"Old Triton model for {cfg.task_id} (fixed batch).",
        hidden=_hidden(cfg, "from"), inter=_inter(cfg, "from"), eps=_eps(cfg, "from"),
        fixed_batch=_batch(cfg, "from"),
    )
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} (dynamic batch).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        fixed_batch=None,
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        fixed_batch=None,
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
    )
    # Remove FIXED_BATCH constant and batch assert from model_forward.
    new_fn = _extract_function(new_triton, "model_forward")
    ops = [{
        "operation": "replace_function",
        "path": "candidate_model_triton.py",
        "function_name": "model_forward",
        "new_code": new_fn,
        "reason": f"remove FIXED_BATCH={_batch(cfg,'from')} assert; target ONNX uses dynamic batch",
    }, {
        "operation": "regex_replace",
        "path": "candidate_model_triton.py",
        "pattern": r"^FIXED_BATCH = \d+\n",
        "replacement": "",
        "reason": "drop unused FIXED_BATCH constant",
    }]
    return RenderedTask(old_triton, new_triton, cpu, ref, ops)


def render_dtype(cfg: Any) -> RenderedTask:
    old_triton = _triton_file(
        doc=f"Old Triton model for {cfg.task_id} ({_dtype_name(cfg,'from')}).",
        hidden=_hidden(cfg, "from"), inter=_inter(cfg, "from"), eps=_eps(cfg, "from"),
        dtype_name=_dtype_name(cfg, "from"),
    )
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} ({_dtype_name(cfg,'to')}).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        dtype_name=_dtype_name(cfg, "to"),
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        dtype_name=_dtype_name(cfg, "to"),
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        dtype_name=_dtype_name(cfg, "to"),
    )
    ops = [_patch_const("DTYPE_NAME", _dtype_name(cfg, "from"), _dtype_name(cfg, "to"),
                        f"dtype {_dtype_name(cfg,'from')} -> {_dtype_name(cfg,'to')}")]
    return RenderedTask(old_triton, new_triton, cpu, ref, ops)


def render_gelu_bias(cfg: Any) -> RenderedTask:
    old_triton = _triton_file(
        doc=f"Old Triton model for {cfg.task_id} (GELU tanh).",
        hidden=_hidden(cfg, "from"), inter=_inter(cfg, "from"), eps=_eps(cfg, "from"),
        gelu_body=_GELU_TANH_BODY, activation="gelu_tanh",
    )
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} (BiasGELU).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        gelu_body=_GELU_BIAS_BODY, gelu_in_bias=float(_p(cfg, "to", "gelu_in_bias", 0.125)),
        activation="gelu_bias",
    )
    bias_val = float(_p(cfg, "to", "gelu_in_bias", 0.125))
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        activation="gelu_bias",
        gelu_in_bias=bias_val,
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        activation="gelu_bias",
        gelu_in_bias=bias_val,
    )
    ops = [
        {
            "operation": "regex_replace",
            "path": "candidate_model_triton.py",
            "pattern": r"(GELU_BLOCK_SIZE = \d+\n)",
            "replacement": rf"\1GELU_IN_BIAS = {bias_val}\n",
            "reason": "add GELU_IN_BIAS constant for BiasGELU",
        },
        {
            "operation": "replace_kernel_body",
            "path": "candidate_model_triton.py",
            "kernel_name": "gelu_act_kernel",
            "new_body": _GELU_BIAS_BODY,
            "reason": "switch GELU kernel to BiasGELU (add GELU_IN_BIAS inside kernel)",
        },
        {
            "operation": "regex_replace",
            "path": "candidate_model_triton.py",
            "pattern": r"_gelu_tanh\(h\)",
            "replacement": "_gelu_bias(h)",
            "reason": "call BiasGELU helper in model_forward",
        },
    ]
    return RenderedTask(old_triton, new_triton, cpu, ref, ops)


def render_gelu_relu(cfg: Any) -> RenderedTask:
    old_triton = _triton_file(
        doc=f"Old Triton model for {cfg.task_id} (GELU).",
        hidden=_hidden(cfg, "from"), inter=_inter(cfg, "from"), eps=_eps(cfg, "from"),
        gelu_body=_GELU_TANH_BODY, activation="gelu_tanh",
    )
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} (ReLU).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        gelu_body=_RELU_BODY, activation="relu",
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        activation="relu",
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        activation="relu",
    )
    ops = [{
        "operation": "replace_kernel_body",
        "path": "candidate_model_triton.py",
        "kernel_name": "gelu_act_kernel",
        "new_body": _RELU_BODY,
        "reason": "replace GELU activation kernel with ReLU",
    }, {
        "operation": "regex_replace",
        "path": "candidate_model_triton.py",
        "pattern": r"_gelu_tanh\(h\)",
        "replacement": "_gelu_relu(h)",
        "reason": "call ReLU helper in model_forward",
    }]
    return RenderedTask(old_triton, new_triton, cpu, ref, ops)


def render_rmsnorm(cfg: Any) -> RenderedTask:
    old_triton = _triton_file(
        doc=f"Old Triton model for {cfg.task_id} (LayerNorm).",
        hidden=_hidden(cfg, "from"), inter=_inter(cfg, "from"), eps=_eps(cfg, "from"),
        ln_body=_LN_BODY,
    )
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} (RMSNorm).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        ln_body=_RMS_BODY,
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        norm="rmsnorm",
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        norm="rmsnorm",
    )
    ops = [{
        "operation": "replace_kernel_body",
        "path": "candidate_model_triton.py",
        "kernel_name": "layernorm_fwd_kernel",
        "new_body": _RMS_BODY,
        "reason": "replace LayerNorm kernel body with RMSNorm (no mean centering)",
    }]
    return RenderedTask(old_triton, new_triton, cpu, ref, ops)


def render_combo_hidden_dtype(cfg: Any) -> RenderedTask:
    rendered = render_shape(cfg)
    dtype_ops = [_patch_const("DTYPE_NAME", _dtype_name(cfg, "from"), _dtype_name(cfg, "to"),
                             f"dtype {_dtype_name(cfg,'from')} -> {_dtype_name(cfg,'to')}")]
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} (shape + dtype).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        dtype_name=_dtype_name(cfg, "to"),
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        dtype_name=_dtype_name(cfg, "to"),
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        dtype_name=_dtype_name(cfg, "to"),
    )
    return RenderedTask(rendered.old_triton, new_triton, cpu, ref, rendered.patch_ops + dtype_ops)


def render_combo_shape_eps(cfg: Any) -> RenderedTask:
    """Hard: shape constants + LN_EPS (render_shape handles both)."""
    return render_shape(cfg)


def render_combo_shape_gelu_relu(cfg: Any) -> RenderedTask:
    """Hard: update shape constants + swap GELU kernel to ReLU."""
    shape = render_shape(cfg)
    relu = render_gelu_relu(cfg)
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} (shape + ReLU).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        gelu_body=_RELU_BODY, activation="relu",
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        activation="relu",
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        activation="relu",
    )
    relu_only = [op for op in relu.patch_ops if op.get("operation") != "update_constant"]
    return RenderedTask(shape.old_triton, new_triton, cpu, ref, shape.patch_ops + relu_only)


def render_combo_rmsnorm_dtype(cfg: Any) -> RenderedTask:
    """Hard: RMSNorm kernel swap + dtype constant."""
    rms = render_rmsnorm(cfg)
    dtype_ops = [_patch_const(
        "DTYPE_NAME", _dtype_name(cfg, "from"), _dtype_name(cfg, "to"),
        f"dtype {_dtype_name(cfg,'from')} -> {_dtype_name(cfg,'to')}",
    )]
    new_triton = _triton_file(
        doc=f"Oracle Triton model for {cfg.task_id} (RMSNorm + dtype).",
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        ln_body=_RMS_BODY, dtype_name=_dtype_name(cfg, "to"),
    )
    cpu = _cpu_demo(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        norm="rmsnorm", dtype_name=_dtype_name(cfg, "to"),
    )
    ref = _reference(
        task_id=cfg.task_id,
        hidden=_hidden(cfg, "to"), inter=_inter(cfg, "to"), eps=_eps(cfg, "to"),
        norm="rmsnorm", dtype_name=_dtype_name(cfg, "to"),
    )
    return RenderedTask(rms.old_triton, new_triton, cpu, ref, rms.patch_ops + dtype_ops)


PROFILE_RENDERERS: dict[str, Callable[[Any], RenderedTask]] = {
    "shape": render_shape,
    "seq_len": render_seq_len,
    "batch": render_batch,
    "dtype": render_dtype,
    "gelu_bias": render_gelu_bias,
    "gelu_relu": render_gelu_relu,
    "rmsnorm": render_rmsnorm,
    "combo_hidden_dtype": render_combo_hidden_dtype,
    "combo_shape_eps": render_combo_shape_eps,
    "combo_shape_gelu_relu": render_combo_shape_gelu_relu,
    "combo_rmsnorm_dtype": render_combo_rmsnorm_dtype,
}


def render_task(cfg: Any) -> RenderedTask:
    fn = PROFILE_RENDERERS.get(cfg.profile)
    if fn is None:
        raise ValueError(f"unknown profile: {cfg.profile}")
    return fn(cfg)
