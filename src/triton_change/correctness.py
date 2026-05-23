"""Sandboxed correctness check for Triton candidate files.

Per spec section 5.6. Runs the candidate's `model_forward` in a Python
subprocess, captures outputs, and compares to `target_outputs.pt`.

Sandbox guarantees (by platform):

- Wall-clock timeout (cross-platform).
- subprocess.DEVNULL stdin (no interactive input).
- stdout/stderr capped (kept tail only) to bound memory.
- Working dir locked to `<task_dir>/sandbox/` (candidate cannot reach the
  oracle/ directory which holds the gold answer).
- On Linux/Unix: best-effort `RLIMIT_AS` 4GB via `resource.setrlimit`.
- Network is NOT explicitly firewalled at this level (TODO: container layer);
  the import whitelist in static_check is the first line of defense.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


__all__ = ["CorrectnessResult", "correctness_check"]


# Runner template. Uses sentinel placeholders so the f-strings inside the
# template stay literal until runner.py is parsed.
_RUNNER_TEMPLATE = textwrap.dedent('''\
    """Auto-generated correctness runner. Do not edit."""
    import importlib.util
    import json
    import sys
    import traceback
    from pathlib import Path

    DEVICE = "__DEVICE__"
    WEIGHT_KEYS = __WEIGHT_KEYS__

    HERE = Path(__file__).resolve().parent
    CANDIDATE = HERE / "candidate_model_triton.py"
    WEIGHTS = HERE / "weights.pt"
    INPUTS = HERE / "test_inputs.pt"
    OUTPUTS = HERE / "candidate_outputs.pt"
    ERR = HERE / "runner_error.json"


    def _err(failure_class, message):
        ERR.write_text(
            json.dumps({"failure_class": failure_class, "error": message}, ensure_ascii=False)
        )
        return 2


    try:
        import torch
    except Exception as e:
        sys.exit(_err("import", f"failed to import torch: {e!r}"))

    try:
        spec = importlib.util.spec_from_file_location("candidate_model_triton", CANDIDATE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        tb = traceback.format_exc(limit=20)
        sys.exit(_err("import", f"failed to load candidate: {e!r}\\n{tb}"))

    if not hasattr(mod, "model_forward"):
        sys.exit(_err("import", "candidate has no model_forward"))

    try:
        weights = torch.load(WEIGHTS, weights_only=False)
        inputs = torch.load(INPUTS, weights_only=False)
    except Exception as e:
        sys.exit(_err("runtime", f"failed to load weights/inputs: {e!r}"))

    outputs = []
    try:
        for x in inputs:
            x_d = x.to(DEVICE) if DEVICE != "cpu" else x
            wargs = {k: (v.to(DEVICE) if DEVICE != "cpu" else v) for k, v in weights.items()}
            y = mod.model_forward(x_d, *(wargs[k] for k in WEIGHT_KEYS))
            outputs.append(y.detach().to("cpu"))
    except Exception as e:
        tb = traceback.format_exc(limit=20)
        sys.exit(_err("runtime", f"model_forward failed: {e!r}\\n{tb}"))

    torch.save(outputs, OUTPUTS)
    sys.exit(0)
''')


def _make_runner_source(device: str, weight_keys: list[str]) -> str:
    return (
        _RUNNER_TEMPLATE
        .replace("__DEVICE__", device)
        .replace("__WEIGHT_KEYS__", repr(list(weight_keys)))
    )


@dataclass
class CorrectnessResult:
    executed: bool
    passed: bool
    failure_class: str | None
    error: str | None
    shape_match: bool
    dtype_match: bool
    max_abs_error: float | None
    max_rel_error: float | None
    elapsed_s: float
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def correctness_check(
    candidate_path: Path,
    task_dir: Path,
    timeout: float = 30.0,
    device: str = "cpu",
    python_exe: str | None = None,
) -> CorrectnessResult:
    """Run the candidate in a subprocess and compare outputs to target_outputs.pt.

    Args:
        candidate_path: path to a candidate .py file. Will be COPIED into a
            sandbox dir; the original is not touched.
        task_dir: tasks/<task_id>/ — used to locate hidden_eval/.
        timeout: wall-clock seconds.
        device: "cpu" or "cuda" (or "cuda:0" etc). For Triton runs choose cuda.
        python_exe: override Python interpreter (defaults to sys.executable).
    """
    candidate_path = Path(candidate_path)
    task_dir = Path(task_dir)

    eval_dir = task_dir / "hidden_eval"
    specs = json.loads((eval_dir / "input_specs.json").read_text(encoding="utf-8"))
    weight_keys = list(specs["weight_keys"])
    atol = float(specs.get("tolerance", {}).get("atol", 1e-4))
    rtol = float(specs.get("tolerance", {}).get("rtol", 1e-4))

    sandbox = task_dir / "sandbox"
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate_path, sandbox / "candidate_model_triton.py")
    shutil.copy2(eval_dir / "weights.pt", sandbox / "weights.pt")
    shutil.copy2(eval_dir / "test_inputs.pt", sandbox / "test_inputs.pt")

    runner_src = _make_runner_source(device=device, weight_keys=weight_keys)
    runner_path = sandbox / "runner.py"
    runner_path.write_text(runner_src, encoding="utf-8")

    python_exe = python_exe or sys.executable
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    preexec_fn = None
    if not sys.platform.startswith("win"):
        try:
            import resource

            mem_bytes = 4 * 1024 * 1024 * 1024  # 4GB

            def _limit() -> None:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

            preexec_fn = _limit
        except ImportError:
            pass

    t0 = time.time()
    try:
        proc = subprocess.run(
            [python_exe, str(runner_path)],
            cwd=str(sandbox),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            preexec_fn=preexec_fn,  # type: ignore[arg-type]
        )
    except subprocess.TimeoutExpired as e:
        return CorrectnessResult(
            executed=True,
            passed=False,
            failure_class="timeout",
            error=f"subprocess exceeded {timeout}s",
            shape_match=False,
            dtype_match=False,
            max_abs_error=None,
            max_rel_error=None,
            elapsed_s=time.time() - t0,
            stdout_tail=_tail((e.stdout or b"").decode(errors="replace")),
            stderr_tail=_tail((e.stderr or b"").decode(errors="replace")),
        )

    elapsed = time.time() - t0
    stdout_tail = _tail(proc.stdout.decode(errors="replace"))
    stderr_tail = _tail(proc.stderr.decode(errors="replace"))

    if proc.returncode != 0:
        err_path = sandbox / "runner_error.json"
        if err_path.exists():
            err = json.loads(err_path.read_text(encoding="utf-8"))
            return CorrectnessResult(
                executed=True,
                passed=False,
                failure_class=err["failure_class"],
                error=err["error"],
                shape_match=False,
                dtype_match=False,
                max_abs_error=None,
                max_rel_error=None,
                elapsed_s=elapsed,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )
        return CorrectnessResult(
            executed=True,
            passed=False,
            failure_class="runtime",
            error=f"subprocess exited {proc.returncode}",
            shape_match=False,
            dtype_match=False,
            max_abs_error=None,
            max_rel_error=None,
            elapsed_s=elapsed,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    return _compare_outputs(
        sandbox / "candidate_outputs.pt",
        eval_dir / "target_outputs.pt",
        atol=atol,
        rtol=rtol,
        elapsed=elapsed,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


# ---------- Helpers ----------


def _tail(s: str, n_bytes: int = 4096) -> str:
    if len(s) <= n_bytes:
        return s
    return "...<truncated>...\n" + s[-n_bytes:]


def _compare_outputs(
    candidate_outs_path: Path,
    target_outs_path: Path,
    atol: float,
    rtol: float,
    elapsed: float,
    stdout_tail: str,
    stderr_tail: str,
) -> CorrectnessResult:
    import torch  # local import — keeps module importable without torch present

    cand_outs = torch.load(candidate_outs_path, weights_only=False)
    tgt_outs = torch.load(target_outs_path, weights_only=False)

    if len(cand_outs) != len(tgt_outs):
        return CorrectnessResult(
            executed=True,
            passed=False,
            failure_class="shape_mismatch",
            error=f"output count mismatch: {len(cand_outs)} vs {len(tgt_outs)}",
            shape_match=False,
            dtype_match=False,
            max_abs_error=None,
            max_rel_error=None,
            elapsed_s=elapsed,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
        )

    max_abs = 0.0
    max_rel = 0.0
    dtype_match = True

    for cand, tgt in zip(cand_outs, tgt_outs):
        if cand.shape != tgt.shape:
            return CorrectnessResult(
                executed=True,
                passed=False,
                failure_class="shape_mismatch",
                error=f"shape: {tuple(cand.shape)} vs {tuple(tgt.shape)}",
                shape_match=False,
                dtype_match=False,
                max_abs_error=None,
                max_rel_error=None,
                elapsed_s=elapsed,
                stdout_tail=stdout_tail,
                stderr_tail=stderr_tail,
            )
        if cand.dtype != tgt.dtype:
            dtype_match = False
        diff = (cand.float() - tgt.float()).abs()
        max_abs = max(max_abs, diff.max().item())
        denom = tgt.float().abs().clamp(min=1e-8)
        max_rel = max(max_rel, (diff / denom).max().item())

    numerical_pass = (max_abs <= atol) and (max_rel <= rtol)
    passed = numerical_pass and dtype_match

    failure_class: str | None = None
    if not numerical_pass:
        failure_class = "numerical_diverge"
    elif not dtype_match:
        failure_class = "dtype_mismatch"

    err_msg = (
        None
        if passed
        else f"max_abs={max_abs:.3e} max_rel={max_rel:.3e} (atol={atol} rtol={rtol})"
    )

    return CorrectnessResult(
        executed=True,
        passed=passed,
        failure_class=failure_class,
        error=err_msg,
        shape_match=True,
        dtype_match=dtype_match,
        max_abs_error=max_abs,
        max_rel_error=max_rel,
        elapsed_s=elapsed,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )
