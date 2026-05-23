"""Scan git-tracked files for API keys. Exit 1 if any found.

Usage (before push):
    python scripts/check_no_secrets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_no_secrets import _KEY_RE, _tracked_files, _ALLOW_EMPTY_PLACEHOLDER  # noqa: E402


def main() -> int:
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
                violations.append(f"  {rel}:{i}")
    if violations:
        print("[FAIL] API key pattern found in git-tracked files:")
        print("\n".join(violations))
        print("\nRemove keys from these files. Keys belong only in .env (gitignored).")
        return 1
    print("[OK] no API keys in tracked files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
