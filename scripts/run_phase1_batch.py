"""Batch Phase-1 oracle evaluation (cpu-demo path, no GPU).

Usage:
    python scripts/run_phase1_batch.py
    python scripts/run_phase1_batch.py --from 1 --to 100 --cpu-demo
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from triton_change.evaluator import evaluate_oracle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "tasks")
    ap.add_argument("--from", dest="from_n", type=int, default=1)
    ap.add_argument("--to", dest="to_n", type=int, default=100)
    ap.add_argument("--cpu-demo", action="store_true", default=True)
    ap.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "data" / "phase1_batch.json")
    args = ap.parse_args()

    results = []
    ok = 0
    for n in range(args.from_n, args.to_n + 1):
        tid = f"task_{n:06d}"
        td = args.tasks_dir / tid
        if not (td / "oracle" / "patch_ops.json").exists():
            continue
        ev = evaluate_oracle(td, use_cpu_demo=args.cpu_demo, device="cpu")
        results.append(ev.to_dict())
        if ev.success:
            ok += 1
        mark = "ok" if ev.success else "FAIL"
        print(f"  [{mark}] {tid} reward={ev.reward.total:.2f}")

    summary = {
        "total": len(results),
        "success": ok,
        "rate": ok / len(results) if results else 0.0,
        "cpu_demo": args.cpu_demo,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n[done] {ok}/{len(results)} -> {args.output}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
