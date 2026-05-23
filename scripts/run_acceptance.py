"""Run spec acceptance gates (no RL training, no DeepSeek required).

Checks:
  1. No secrets in tracked files
  2. Task schema validation (sample or full)
  3. Oracle patch judge (100 tasks)
  4. Phase-1 batch cpu-demo
  5. Trajectory size < 1MB (if present)

Usage:
    python scripts/run_acceptance.py
    python scripts/run_acceptance.py --quick
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="Sample tasks 1,50,100 only")
    ap.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "data" / "acceptance_report.json")
    args = ap.parse_args()

    py = sys.executable
    gates: dict[str, dict] = {}

    rc = _run([py, "scripts/check_no_secrets.py"])
    gates["no_secrets"] = {"pass": rc == 0}

    if args.quick:
        for tid in ("task_000001", "task_000050", "task_000100"):
            rc = _run([py, "scripts/validate_task.py", f"tasks/{tid}"])
            if rc != 0:
                gates["validate_tasks"] = {"pass": False}
                break
        else:
            gates["validate_tasks"] = {"pass": True}
        fr, to = 1, 100
    else:
        rc = _run([py, "scripts/validate_all_tasks.py", "--from", "1", "--to", "100"])
        gates["validate_tasks"] = {"pass": rc == 0}
        fr, to = 1, 100

    rc = _run([py, "scripts/eval_patch_oracle.py", "--from", str(fr), "--to", str(to)])
    gates["oracle_judge"] = {"pass": rc == 0}

    rc = _run([py, "scripts/run_phase1_batch.py", "--from", str(fr), "--to", str(to), "--cpu-demo"])
    gates["phase1_batch"] = {"pass": rc == 0}

    traj_dir = REPO_ROOT / "trajectories"
    oversized = []
    if traj_dir.is_dir():
        for p in traj_dir.glob("*.jsonl"):
            if p.stat().st_size > 1_000_000:
                oversized.append(str(p))
    gates["trajectory_size"] = {"pass": not oversized, "oversized": oversized}

    rc_tests = _run([py, "-m", "pytest", "-q"])
    gates["pytest"] = {"pass": rc_tests == 0}

    all_pass = all(g.get("pass") for g in gates.values())
    report = {"pass": all_pass, "gates": gates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[acceptance] {'PASS' if all_pass else 'FAIL'} -> {args.output}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
