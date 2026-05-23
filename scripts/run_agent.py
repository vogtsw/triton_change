"""Run the Phase 2 agent on a single task and dump its trajectory.

Usage:
    python scripts/run_agent.py tasks/task_000001 --policy oracle
    python scripts/run_agent.py tasks/task_000001 --policy deepseek --device cuda
    python scripts/run_agent.py tasks/task_000001 --policy oracle --max-steps 4

Policies:
    oracle    Replays oracle/patch_ops.json (sanity baseline).
    deepseek  DeepSeek V3 chat — requires DEEPSEEK_API_KEY in env or .env.
    mock      Hard-coded for testing only (not exposed via CLI).

Outputs:
    tasks/<task>/trajectory.json     pretty single trajectory
    trajectories/<task>__<run>.jsonl appended one-line trajectory
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
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

from triton_change.agent import (
    AgentRunner,
    DeepSeekPolicy,
    OraclePolicy,
    write_trajectory_jsonl,
)
from triton_change.agent.trajectory import write_trajectory_pretty, validate_trajectory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--policy", choices=["oracle", "deepseek"], default="oracle")
    ap.add_argument("--patch-ops", type=Path, default=None,
                    help="Override oracle patch_ops path (oracle policy only).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--max-patch-attempts", type=int, default=5)
    ap.add_argument("--correctness-timeout", type=float, default=60.0)
    ap.add_argument("--out-jsonl", type=Path, default=None,
                    help="Append trajectory to this jsonl file (default trajectories/agent.jsonl).")
    ap.add_argument("--no-validate", action="store_true",
                    help="Skip trajectory schema validation.")
    args = ap.parse_args()

    task_dir = args.task_dir.resolve()
    if not task_dir.is_dir():
        print(f"[err] task dir not found: {task_dir}")
        return 2

    if args.policy == "oracle":
        policy = OraclePolicy(task_dir=task_dir, patch_ops_path=args.patch_ops)
    elif args.policy == "deepseek":
        try:
            policy = DeepSeekPolicy()
        except ValueError as e:
            print(f"[err] DeepSeek client init failed: {e}")
            return 2
    else:
        print(f"[err] unknown policy: {args.policy}")
        return 2

    print(f"[agent] task={task_dir.name} policy={args.policy} device={args.device}")
    print(f"[agent] max_steps={args.max_steps} max_patch_attempts={args.max_patch_attempts}")

    runner = AgentRunner(
        task_dir=task_dir,
        policy=policy,
        max_steps=args.max_steps,
        max_patch_attempts=args.max_patch_attempts,
        device=args.device,
        correctness_timeout=args.correctness_timeout,
    )

    result = runner.run()
    traj = result.to_dict()

    print(f"[agent] success={result.success}  final_reward={result.final_reward:.4f}"
          + (f"  failure_class={result.failure_class}" if result.failure_class else ""))
    print(f"[agent] steps={result.total_steps} total_patch_ops={result.total_patch_ops}")
    if result.llm_call_log:
        n = len(result.llm_call_log)
        toks = sum(c.get("total_tokens", 0) for c in result.llm_call_log)
        secs = sum(c.get("duration_s", 0.0) for c in result.llm_call_log)
        print(f"[agent] llm calls={n}  tokens={toks}  total_time={secs:.2f}s")

    for s in result.steps:
        tool = s.action.get("tool", "?")
        ok = "ok" if not s.failure_class else f"FAIL({s.failure_class})"
        print(f"  step {s.step_idx}: tool={tool:<22s} reward={s.step_reward:+.2f}  {ok}")

    # Write outputs
    pretty_path = task_dir / "trajectory.json"
    write_trajectory_pretty(traj, pretty_path)
    print(f"[agent] wrote {pretty_path}")

    jsonl_path = args.out_jsonl or (REPO_ROOT / "trajectories" / "agent.jsonl")
    write_trajectory_jsonl([traj], jsonl_path)
    print(f"[agent] appended to {jsonl_path}")

    if not args.no_validate:
        try:
            validate_trajectory(traj)
            print("[agent] trajectory schema: ok")
        except Exception as e:
            print(f"[warn] trajectory schema validation failed: {e}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
