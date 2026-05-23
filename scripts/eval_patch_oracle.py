"""Batch oracle patch evaluation (no LLM / no GPU).

Usage:
    python scripts/eval_patch_oracle.py
    python scripts/eval_patch_oracle.py --from 1 --to 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from triton_change.evaluator.patch_judge import judge_patch_for_task


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "tasks")
    ap.add_argument("--from", dest="from_n", type=int, default=1)
    ap.add_argument("--to", dest="to_n", type=int, default=100)
    ap.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "data" / "oracle_eval.json")
    args = ap.parse_args()

    results = []
    ok = 0
    for n in range(args.from_n, args.to_n + 1):
        tid = f"task_{n:06d}"
        td = args.tasks_dir / tid
        if not (td / "oracle" / "patch_ops.json").exists():
            print(f"  [skip] {tid}: missing")
            continue
        ops = json.loads((td / "oracle" / "patch_ops.json").read_text(encoding="utf-8"))["ops"]
        r = judge_patch_for_task(td, ops)
        results.append(r.to_dict())
        if r.correct:
            ok += 1
        mark = "ok" if r.correct else "FAIL"
        print(f"  [{mark}] {tid} ops_match={r.ops_match_oracle} ast={r.ast_match_oracle}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {"total": len(results), "correct": ok, "rate": ok / len(results) if results else 0.0}
    args.output.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\n[done] {ok}/{len(results)} oracle-correct -> {args.output}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
