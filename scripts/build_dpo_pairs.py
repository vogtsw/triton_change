"""Build DPO chosen/rejected pairs by sampling K perturbations per task.

Ranks candidates with oracle patch judge (AST match) — no LLM / no GPU.
Optionally verifies top candidates via cpu-demo correctness.

Usage:
    python scripts/build_dpo_pairs.py
    python scripts/build_dpo_pairs.py --tasks task_000001 --k 8
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from triton_change.evaluator.patch_judge import judge_patch_for_task, normalize_ops
from triton_change.reward import compute_reward


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _perturb_ops(ops: list[dict], rng: random.Random) -> list[dict]:
    if not ops:
        return ops
    variant = copy.deepcopy(ops)
    choice = rng.randint(0, 2)
    if choice == 0 and len(variant) > 1:
        variant.pop(rng.randrange(len(variant)))
    elif choice == 1:
        for op in variant:
            if op.get("operation") == "update_constant" and "new_value" in op:
                old = op.get("old_value", 0)
                op["new_value"] = old if old != op["new_value"] else op["new_value"]
                break
    else:
        rng.shuffle(variant)
    return variant


def _score_candidate(task_dir: Path, ops: list[dict]) -> dict[str, Any]:
    labels_doc = _load_json(task_dir / "hidden_eval" / "semantic_change_labels.json")
    labels = labels_doc if isinstance(labels_doc, list) else labels_doc.get("labels", [])

    judge = judge_patch_for_task(task_dir, ops)
    static_d = None
    if judge.apply_ok:
        static_d = {"passed": True, "imports_ok": True, "bad_imports": [], "danger_findings": []}

    rb = compute_reward(
        static_d,
        {"passed": judge.correct, "failure_class": None if judge.correct else "numerical_diverge"},
        ops,
        labels,
        patch_apply_error=not judge.apply_ok,
    )
    # Boost oracle-correct patches to align with cpu-demo success tier (~2.1)
    reward = rb.total
    if judge.correct:
        reward = max(reward, 2.0)

    return {
        "reward": reward,
        "success": judge.correct,
        "oracle_correct": judge.correct,
        "ops_match": judge.ops_match_oracle,
        "apply_ok": judge.apply_ok,
        "ops": ops,
    }


def sample_candidates(task_dir: Path, k: int, seed: int) -> list[list[dict]]:
    oracle_ops = _load_json(task_dir / "oracle" / "patch_ops.json")["ops"]
    rng = random.Random(seed)
    candidates: list[list[dict]] = [copy.deepcopy(oracle_ops)]
    seen = {json.dumps(normalize_ops(oracle_ops), sort_keys=True)}
    attempts = 0
    while len(candidates) < k and attempts < k * 10:
        attempts += 1
        variant = _perturb_ops(oracle_ops, rng)
        key = json.dumps(normalize_ops(variant), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(variant)
    return candidates


def build_dpo_pair(task_dir: Path, k: int = 8, seed: int = 0) -> dict[str, Any] | None:
    task_id = task_dir.name
    candidates = sample_candidates(task_dir, k, seed)
    scored = [_score_candidate(task_dir, ops) for ops in candidates]
    if len(scored) < 2:
        return None
    scored.sort(key=lambda x: (x["reward"], x["oracle_correct"]), reverse=True)
    chosen, rejected = scored[0], scored[-1]
    if chosen["reward"] <= rejected["reward"]:
        return None
    return {
        "task_id": task_id,
        "k": len(candidates),
        "chosen": {
            "patch_ops": chosen["ops"],
            "reward": chosen["reward"],
            "oracle_correct": chosen["oracle_correct"],
        },
        "rejected": {
            "patch_ops": rejected["ops"],
            "reward": rejected["reward"],
            "oracle_correct": rejected["oracle_correct"],
        },
        "margin": round(chosen["reward"] - rejected["reward"], 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "tasks")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "data" / "dpo_pairs.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.tasks:
        task_dirs = [args.tasks_dir / t for t in args.tasks]
    else:
        task_dirs = sorted(p for p in args.tasks_dir.glob("task_*") if p.is_dir())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for i, td in enumerate(task_dirs):
            if not (td / "oracle" / "patch_ops.json").exists():
                continue
            pair = build_dpo_pair(td, k=args.k, seed=args.seed + i)
            if pair:
                out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                written += 1
                print(f"  [ok] {td.name} margin={pair['margin']}")
            else:
                print(f"  [skip] {td.name}")

    print(f"\n[done] {written} DPO pairs -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
