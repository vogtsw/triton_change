"""Fail if any git-tracked file contains a DeepSeek/OpenAI-style API key."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# sk- + 32+ hex chars (DeepSeek / OpenAI key shape)
_KEY_RE = re.compile(r"sk-[a-f0-9]{20,}", re.IGNORECASE)

# Paths that may mention key format but must not contain real keys
_ALLOW_EMPTY_PLACEHOLDER = {".env.example"}


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in out.stdout.splitlines() if line.strip()]


def test_no_api_keys_in_git_tracked_files():
    violations: list[str] = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in _ALLOW_EMPTY_PLACEHOLDER:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if _KEY_RE.search(line):
                violations.append(f"{rel}:{i}")
    assert not violations, "API key(s) found in tracked files:\n  " + "\n  ".join(violations)
