"""Build the full 100-task catalog (Phase 3b: easy:medium:hard ≈ 5:3:2).

Tasks task_000001–task_000020 are fixed Phase-3a handcrafted entries.
Tasks task_000021–task_000100 are generated programmatically with deterministic
parameter variations (reproducible via seed_offset = task_index * 7).
"""
from __future__ import annotations

from typing import Any

from task_config import TaskConfig, _fp  # noqa: E402  (same scripts/ dir)


# ---------- Phase 3a: fixed 20 tasks (unchanged) ----------

PHASE3A_CONFIGS: list[TaskConfig] = [
    TaskConfig("task_000001", "MVP-easy", "shape", ["shape_param_change"],
               "Hidden 768->1024, intermediate 3072->4096.",
               _fp(hidden_size=768, intermediate_size=3072), _fp(hidden_size=1024, intermediate_size=4096), 0, 1),
    TaskConfig("task_000002", "MVP-easy", "shape", ["shape_param_change"],
               "Hidden 1024->768, intermediate 4096->3072.",
               _fp(hidden_size=1024, intermediate_size=4096), _fp(hidden_size=768, intermediate_size=3072), 10, 1),
    TaskConfig("task_000003", "MVP-easy", "shape", ["shape_param_change"],
               "Hidden 512->768, intermediate 2048->3072.",
               _fp(hidden_size=512, intermediate_size=2048), _fp(hidden_size=768, intermediate_size=3072), 20, 1),
    TaskConfig("task_000004", "MVP-easy", "shape", ["shape_param_change"],
               "Intermediate-only 3072->4096.",
               _fp(intermediate_size=3072), _fp(intermediate_size=4096), 30, 1),
    TaskConfig("task_000005", "MVP-easy", "shape", ["shape_param_change"],
               "Intermediate-only 2048->3072.",
               _fp(intermediate_size=2048), _fp(intermediate_size=3072), 40, 1),
    TaskConfig("task_000006", "MVP-easy", "seq_len", ["shape_param_change"],
               "Sequence length 128->256.",
               _fp(seq_len=128), _fp(seq_len=256), 50, 1),
    TaskConfig("task_000007", "MVP-easy", "seq_len", ["shape_param_change"],
               "Sequence length 256->512.",
               _fp(seq_len=256), _fp(seq_len=512), 60, 1),
    TaskConfig("task_000008", "MVP-easy", "seq_len", ["shape_param_change"],
               "Sequence length 512->1024.",
               _fp(seq_len=512), _fp(seq_len=1024), 70, 1),
    TaskConfig("task_000009", "MVP-easy", "shape", ["norm_change"],
               "LayerNorm epsilon 1e-5 -> 1e-6.",
               _fp(ln_eps=1e-5), _fp(ln_eps=1e-6), 80, 1),
    TaskConfig("task_000010", "MVP-easy", "shape", ["norm_change"],
               "LayerNorm epsilon 1e-12 -> 1e-5.",
               _fp(ln_eps=1e-12), _fp(ln_eps=1e-5), 90, 1),
    TaskConfig("task_000011", "MVP-easy", "batch", ["shape_param_change"],
               "Static batch B=1 -> dynamic.",
               _fp(batch_size=1), _fp(batch_size=2), 100, 2, dynamic_batch_target=True),
    TaskConfig("task_000012", "MVP-easy", "batch", ["shape_param_change"],
               "Static batch B=4 -> dynamic.",
               _fp(batch_size=4), _fp(batch_size=3), 110, 2, dynamic_batch_target=True),
    TaskConfig("task_000013", "MVP-medium", "dtype", ["dtype_change"],
               "fp32 -> fp16.",
               _fp(dtype_name="float32"), _fp(dtype_name="float16"), 120, 2),
    TaskConfig("task_000014", "MVP-medium", "dtype", ["dtype_change"],
               "fp32 -> fp16 (hidden=512).",
               _fp(hidden_size=512, intermediate_size=2048, dtype_name="float32"),
               _fp(hidden_size=512, intermediate_size=2048, dtype_name="float16"), 130, 2),
    TaskConfig("task_000015", "MVP-medium", "dtype", ["dtype_change"],
               "fp32 -> bf16.",
               _fp(dtype_name="float32"), _fp(dtype_name="bfloat16"), 140, 2),
    TaskConfig("task_000016", "MVP-medium", "gelu_bias", ["activation_change"],
               "GELU -> BiasGELU (bias=0.125).",
               _fp(), _fp(gelu_in_bias=0.125), 150, 3),
    TaskConfig("task_000017", "MVP-medium", "gelu_bias", ["activation_change"],
               "GELU -> BiasGELU (bias=0.25).",
               _fp(), _fp(gelu_in_bias=0.25), 160, 3),
    TaskConfig("task_000018", "MVP-medium", "gelu_relu", ["activation_change"],
               "GELU -> ReLU.",
               _fp(), _fp(), 170, 3),
    TaskConfig("task_000019", "MVP-medium", "rmsnorm", ["norm_change"],
               "LayerNorm -> RMSNorm.",
               _fp(), _fp(), 180, 3),
    TaskConfig("task_000020", "MVP-medium", "combo_hidden_dtype", ["shape_param_change", "dtype_change"],
               "hidden 768->1024 + fp32->fp16.",
               _fp(hidden_size=768, intermediate_size=3072, dtype_name="float32"),
               _fp(hidden_size=1024, intermediate_size=4096, dtype_name="float16"), 190, 4),
]


_HIDDEN_POOL = [512, 768, 896, 1024, 1280]
_INTER_POOL = [2048, 3072, 3584, 4096, 5120]
_SEQ_POOL = [64, 128, 256, 512, 768, 1024]
_EPS_PAIRS = [(1e-5, 1e-6), (1e-12, 1e-5), (1e-6, 1e-5), (1e-5, 1e-4)]
_BATCH_PAIRS = [(1, 2), (2, 4), (4, 2), (8, 4)]
_DTYPE_PAIRS = [("float32", "float16"), ("float32", "bfloat16")]
_GELU_BIAS = [0.0625, 0.125, 0.25, 0.5]


def _tid(n: int) -> str:
    return f"task_{n:06d}"


def _seed(n: int) -> int:
    return n * 7


def _easy_config(n: int) -> TaskConfig:
    """Deterministic easy-task factory for n in 21..58."""
    k = n - 21
    kind = k % 5
    seed = _seed(n)
    if kind == 0:  # shape hidden+inter
        h0, h1 = _HIDDEN_POOL[k % len(_HIDDEN_POOL)], _HIDDEN_POOL[(k + 2) % len(_HIDDEN_POOL)]
        i0, i1 = h0 * 4, h1 * 4
        return TaskConfig(
            _tid(n), "MVP-easy", "shape", ["shape_param_change"],
            f"Shape hidden {h0}->{h1}, inter {i0}->{i1}.",
            _fp(hidden_size=h0, intermediate_size=i0),
            _fp(hidden_size=h1, intermediate_size=i1),
            seed, 1,
        )
    if kind == 1:  # intermediate only
        h = _HIDDEN_POOL[k % len(_HIDDEN_POOL)]
        i0, i1 = _INTER_POOL[k % len(_INTER_POOL)], _INTER_POOL[(k + 1) % len(_INTER_POOL)]
        return TaskConfig(
            _tid(n), "MVP-easy", "shape", ["shape_param_change"],
            f"Intermediate-only {i0}->{i1}, hidden={h}.",
            _fp(hidden_size=h, intermediate_size=i0),
            _fp(hidden_size=h, intermediate_size=i1),
            seed, 1,
        )
    if kind == 2:  # seq_len
        s0, s1 = _SEQ_POOL[k % len(_SEQ_POOL)], _SEQ_POOL[(k + 1) % len(_SEQ_POOL)]
        if s0 >= s1:
            s0, s1 = s1, s0
        return TaskConfig(
            _tid(n), "MVP-easy", "seq_len", ["shape_param_change"],
            f"SEQ_LEN {s0}->{s1}.",
            _fp(seq_len=s0), _fp(seq_len=s1),
            seed, 1,
        )
    if kind == 3:  # ln_eps
        e0, e1 = _EPS_PAIRS[k % len(_EPS_PAIRS)]
        return TaskConfig(
            _tid(n), "MVP-easy", "shape", ["norm_change"],
            f"LN_EPS {e0}->{e1}.",
            _fp(ln_eps=e0), _fp(ln_eps=e1),
            seed, 1,
        )
    # batch
    b0, b1 = _BATCH_PAIRS[k % len(_BATCH_PAIRS)]
    return TaskConfig(
        _tid(n), "MVP-easy", "batch", ["shape_param_change"],
        f"Fixed batch {b0} -> dynamic (test B={b1}).",
        _fp(batch_size=b0), _fp(batch_size=b1),
        seed, 2, dynamic_batch_target=True,
    )


def _medium_config(n: int) -> TaskConfig:
    k = n - 59
    seed = _seed(n)
    kind = k % 4
    if kind == 0:
        d0, d1 = _DTYPE_PAIRS[k % len(_DTYPE_PAIRS)]
        h = _HIDDEN_POOL[k % len(_HIDDEN_POOL)]
        inter = h * 4
        return TaskConfig(
            _tid(n), "MVP-medium", "dtype", ["dtype_change"],
            f"dtype {d0}->{d1} (hidden={h}).",
            _fp(hidden_size=h, intermediate_size=inter, dtype_name=d0),
            _fp(hidden_size=h, intermediate_size=inter, dtype_name=d1),
            seed, 2,
        )
    if kind == 1:
        bias = _GELU_BIAS[k % len(_GELU_BIAS)]
        h = _HIDDEN_POOL[(k + 1) % len(_HIDDEN_POOL)]
        return TaskConfig(
            _tid(n), "MVP-medium", "gelu_bias", ["activation_change"],
            f"GELU->BiasGELU bias={bias} (hidden={h}).",
            _fp(hidden_size=h, intermediate_size=h * 4),
            _fp(hidden_size=h, intermediate_size=h * 4, gelu_in_bias=bias),
            seed, 3,
        )
    if kind == 2:
        return TaskConfig(
            _tid(n), "MVP-medium", "gelu_relu", ["activation_change"],
            f"GELU->ReLU (variant {k}).",
            _fp(hidden_size=_HIDDEN_POOL[k % len(_HIDDEN_POOL)]),
            _fp(hidden_size=_HIDDEN_POOL[k % len(_HIDDEN_POOL)]),
            seed, 3,
        )
    return TaskConfig(
        _tid(n), "MVP-medium", "rmsnorm", ["norm_change"],
        f"LayerNorm->RMSNorm (variant {k}).",
        _fp(hidden_size=_HIDDEN_POOL[(k + 2) % len(_HIDDEN_POOL)]),
        _fp(hidden_size=_HIDDEN_POOL[(k + 2) % len(_HIDDEN_POOL)]),
        seed, 3,
    )


def _hard_config(n: int) -> TaskConfig:
    k = n - 81
    seed = _seed(n)
    kind = k % 4
    h0 = _HIDDEN_POOL[k % len(_HIDDEN_POOL)]
    h1 = _HIDDEN_POOL[(k + 1) % len(_HIDDEN_POOL)]
    i0, i1 = h0 * 4, h1 * 4
    if kind == 0:
        return TaskConfig(
            _tid(n), "MVP-hard", "combo_shape_eps", ["shape_param_change", "norm_change"],
            f"Combo shape+eps: H {h0}->{h1}, eps 1e-5->1e-6.",
            _fp(hidden_size=h0, intermediate_size=i0, ln_eps=1e-5),
            _fp(hidden_size=h1, intermediate_size=i1, ln_eps=1e-6),
            seed, 4,
        )
    if kind == 1:
        d0, d1 = _DTYPE_PAIRS[k % len(_DTYPE_PAIRS)]
        return TaskConfig(
            _tid(n), "MVP-hard", "combo_hidden_dtype", ["shape_param_change", "dtype_change"],
            f"Combo shape+dtype: H {h0}->{h1}, {d0}->{d1}.",
            _fp(hidden_size=h0, intermediate_size=i0, dtype_name=d0),
            _fp(hidden_size=h1, intermediate_size=i1, dtype_name=d1),
            seed, 5,
        )
    if kind == 2:
        return TaskConfig(
            _tid(n), "MVP-hard", "combo_shape_gelu_relu", ["shape_param_change", "activation_change"],
            f"Combo shape+ReLU: H {h0}->{h1}.",
            _fp(hidden_size=h0, intermediate_size=i0),
            _fp(hidden_size=h1, intermediate_size=i1),
            seed, 5,
        )
    return TaskConfig(
        _tid(n), "MVP-hard", "combo_rmsnorm_dtype", ["norm_change", "dtype_change"],
        f"Combo RMSNorm+dtype fp32->fp16 (H={h0}).",
        _fp(hidden_size=h0, intermediate_size=i0, dtype_name="float32"),
        _fp(hidden_size=h0, intermediate_size=i0, dtype_name="float16"),
        seed, 5,
    )


def build_phase3b_configs() -> list[TaskConfig]:
    out: list[TaskConfig] = []
    for n in range(21, 59):
        out.append(_easy_config(n))
    for n in range(59, 81):
        out.append(_medium_config(n))
    for n in range(81, 101):
        out.append(_hard_config(n))
    return out


def build_all_configs() -> list[TaskConfig]:
    return PHASE3A_CONFIGS + build_phase3b_configs()


def tier_counts(configs: list[TaskConfig] | None = None) -> dict[str, int]:
    configs = configs or build_all_configs()
    counts: dict[str, int] = {}
    for c in configs:
        counts[c.tier] = counts.get(c.tier, 0) + 1
    return counts
