"""Validate all task directories against schemas.

Usage:
    python scripts/validate_all_tasks.py
    python scripts/validate_all_tasks.py --from 1 --to 100
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "tasks")
    ap.add_argument("--from", dest="from_n", type=int, default=1)
    ap.add_argument("--to", dest="to_n", type=int, default=100)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    failed = 0
    for n in range(args.from_n, args.to_n + 1):
        tid = f"task_{n:06d}"
        td = args.tasks_dir / tid
        if not td.is_dir():
            print(f"  [skip] {tid}")
            continue
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "validate_task.py"), str(td)]
        if args.strict:
            cmd.append("--strict")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed += 1
            print(f"  [FAIL] {tid}")
            if r.stdout:
                print(r.stdout.strip())
        else:
            print(f"  [ok]   {tid}")

    print(f"\n[done] {args.to_n - args.from_n + 1 - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
