"""Unit tests for triton_change.static_check."""
from __future__ import annotations

from pathlib import Path

import pytest

from triton_change.static_check import static_check


REPO_ROOT = Path(__file__).resolve().parent.parent
TASK_DIR = REPO_ROOT / "tasks" / "task_000001"


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "f.py"
    p.write_text(content, encoding="utf-8")
    return p


def test_old_triton_passes(tmp_path):
    r = static_check(TASK_DIR / "old_model_triton.py")
    assert r.passed, r.to_dict()
    assert r.model_forward_present
    assert "layernorm_fwd_kernel" in r.triton_kernels_found
    assert "gelu_act_kernel" in r.triton_kernels_found
    assert not r.danger_findings


def test_oracle_triton_passes(tmp_path):
    r = static_check(TASK_DIR / "oracle" / "new_model_triton.py")
    assert r.passed
    assert r.imports_ok


def test_syntax_error_caught(tmp_path):
    p = _write(tmp_path, "def model_forward(:\n  return 1\n")
    r = static_check(p)
    assert not r.syntax_ok
    assert "line" in r.syntax_error
    assert not r.passed


def test_missing_model_forward(tmp_path):
    p = _write(tmp_path, "import torch\n\ndef other(x):\n    return x\n")
    r = static_check(p)
    assert not r.model_forward_present
    assert not r.passed


def test_disallowed_import_os(tmp_path):
    src = """import os
import torch

def model_forward(x):
    return x
"""
    p = _write(tmp_path, src)
    r = static_check(p)
    assert not r.imports_ok
    assert "os" in r.bad_imports
    assert not r.passed


def test_disallowed_import_subprocess(tmp_path):
    src = """import subprocess
import torch

def model_forward(x):
    return x
"""
    p = _write(tmp_path, src)
    r = static_check(p)
    assert not r.imports_ok
    assert "subprocess" in r.bad_imports


def test_eval_call_flagged(tmp_path):
    src = """import torch

def model_forward(x):
    z = eval("1+1")
    return x * z
"""
    p = _write(tmp_path, src)
    r = static_check(p)
    assert any("eval" in f for f in r.danger_findings)
    assert not r.passed


def test_os_system_call_flagged(tmp_path):
    # Even though `import os` would also be flagged, this tests the call scanner
    src = """import torch
import os

def model_forward(x):
    os.system("rm -rf /")
    return x
"""
    p = _write(tmp_path, src)
    r = static_check(p)
    assert any("os.system" in f for f in r.danger_findings)


def test_allowed_imports_pass(tmp_path):
    src = """import torch
import triton
import triton.language as tl
import numpy as np
import math
from typing import Any
from dataclasses import dataclass
from __future__ import annotations

def model_forward(x):
    return x
"""
    # `from __future__` must be at top, but `dataclasses` can be later. Reorder for valid syntax.
    src = """from __future__ import annotations
import torch
import triton
import triton.language as tl
import numpy as np
import math
from typing import Any
from dataclasses import dataclass

def model_forward(x):
    return x
"""
    p = _write(tmp_path, src)
    r = static_check(p)
    assert r.imports_ok, r.bad_imports


def test_to_dict_has_passed_key(tmp_path):
    r = static_check(TASK_DIR / "old_model_triton.py")
    d = r.to_dict()
    assert d["passed"] is True
    assert "syntax_ok" in d
    assert "danger_findings" in d
