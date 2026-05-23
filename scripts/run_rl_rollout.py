"""Run RL rollouts and compute GRPO/RLOO advantages (Phase 6 scaffolding).

Usage:
    python scripts/run_rl_rollout.py --policy oracle --cpu-demo
    python scripts/run_rl_rollout.py --policy oracle --advantage grpo
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

from triton_change.agent.policy import OraclePolicy
from triton_change.training.grpo import compute_grpo_advantages
from triton_change.training.rollout import rollout_tasks
from triton_change.training.rloo import compute_rloo_advantages


def _oracle_factory(cpu_demo: bool):
    def factory(task_dir: Path) -> OraclePolicy:
        patch = (
            task_dir / "oracle" / "cpu_demo_patch_ops.json"
            if cpu_demo
            else task_dir / "oracle" / "patch_ops.json"
        )
        return OraclePolicy(task_dir=task_dir, patch_ops_path=patch)
    return factory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks-dir", type=Path, default=REPO_ROOT / "tasks")
    ap.add_argument("--from", dest="from_n", type=int, default=1)
    ap.add_argument("--to", dest="to_n", type=int, default=5)
    ap.add_argument("--policy", choices=["oracle"], default="oracle")
    ap.add_argument("--cpu-demo", action="store_true")
    ap.add_argument("--advantage", choices=["grpo", "rloo", "none"], default="grpo")
    ap.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "trajectories" / "rl_rollout.jsonl")
    ap.add_argument("--no-langgraph", action="store_true")
    args = ap.parse_args()

    task_dirs = []
    for n in range(args.from_n, args.to_n + 1):
        td = args.tasks_dir / f"task_{n:06d}"
        if td.is_dir():
            task_dirs.append(td)

    policy = _oracle_factory(args.cpu_demo)

    trajs = rollout_tasks(
        task_dirs,
        policy,
        output_jsonl=args.output,
        device="cpu",
        use_langgraph=not args.no_langgraph,
    )

    rewards = [t.get("final_reward", 0.0) for t in trajs]
    summary: dict = {"n": len(trajs), "rewards": rewards, "mean_reward": sum(rewards) / len(rewards) if rewards else 0}

    if args.advantage == "grpo" and rewards:
        batch = compute_grpo_advantages(rewards)
        summary["grpo"] = {"baseline": batch.baseline, "advantages": batch.advantages}
    elif args.advantage == "rloo" and rewards:
        batch = compute_rloo_advantages(rewards)
        summary["rloo"] = {"advantages": batch.advantages}

    report = REPO_ROOT / "data" / "rl_rollout_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] trajectories -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
