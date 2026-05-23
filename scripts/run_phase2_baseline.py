"""Run the agent across all generated tasks; emit a summary table.

Default policy is `oracle` (sanity check that the agent loop wires up correctly).
Use `--policy deepseek` for the real Phase -1 frontier baseline.

Per-task outputs are appended to `trajectories/baseline_<policy>.jsonl`. A
human-readable summary is printed and saved to `trajectories/baseline_<policy>.md`.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", choices=["oracle", "deepseek"], default="oracle")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--correctness-timeout", type=float, default=60.0)
    ap.add_argument("--cpu-demo", action="store_true",
                    help="Use oracle/cpu_demo_patch_ops.json instead of the real oracle "
                         "(useful on Windows where Triton is not available).")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Subset of task ids; default is all tasks under tasks/.")
    args = ap.parse_args()

    tasks_dir = REPO_ROOT / "tasks"
    if args.tasks:
        task_dirs = [tasks_dir / t for t in args.tasks]
    else:
        task_dirs = sorted(p for p in tasks_dir.iterdir() if p.is_dir() and p.name.startswith("task_"))

    out_jsonl = REPO_ROOT / "trajectories" / f"baseline_{args.policy}.jsonl"
    out_md = REPO_ROOT / "trajectories" / f"baseline_{args.policy}.md"

    rows = []
    successes = 0
    total_tokens = 0
    total_calls = 0

    for td in task_dirs:
        if not td.is_dir():
            print(f"[skip] {td.name}: not a dir")
            continue

        if args.policy == "oracle":
            patch_ops_path = (td / "oracle" / "cpu_demo_patch_ops.json"
                              if args.cpu_demo else td / "oracle" / "patch_ops.json")
            policy = OraclePolicy(task_dir=td, patch_ops_path=patch_ops_path)
        else:
            try:
                policy = DeepSeekPolicy()
            except ValueError as e:
                print(f"[err] DeepSeek client init failed: {e}")
                return 2

        print(f"\n=== {td.name} ({args.policy}) ===")
        runner = AgentRunner(
            task_dir=td,
            policy=policy,
            max_steps=args.max_steps,
            device=args.device,
            correctness_timeout=args.correctness_timeout,
        )
        result = runner.run()
        traj = result.to_dict()
        write_trajectory_jsonl([traj], out_jsonl)

        if result.success:
            successes += 1

        n_calls = len(result.llm_call_log)
        n_toks = sum(c.get("total_tokens", 0) for c in result.llm_call_log)
        total_calls += n_calls
        total_tokens += n_toks

        print(f"  success={result.success} reward={result.final_reward:.3f}"
              f" steps={result.total_steps}"
              + (f" llm_calls={n_calls} tokens={n_toks}" if n_calls else ""))
        rows.append({
            "task_id": result.task_id,
            "tier": result.tier,
            "success": result.success,
            "final_reward": result.final_reward,
            "failure_class": result.failure_class,
            "total_steps": result.total_steps,
            "total_patch_ops": result.total_patch_ops,
            "llm_calls": n_calls,
            "tokens": n_toks,
        })

    # Summary
    n = max(1, len(rows))
    success_rate = successes / n
    avg_reward = sum(r["final_reward"] for r in rows) / n
    avg_steps = sum(r["total_steps"] for r in rows) / n

    print("\n" + "=" * 60)
    print(f"BASELINE: policy={args.policy}  tasks={len(rows)}  device={args.device}")
    print(f"  success_rate = {successes}/{len(rows)} ({success_rate*100:.1f}%)")
    print(f"  avg_reward   = {avg_reward:.3f}")
    print(f"  avg_steps    = {avg_steps:.2f}")
    if total_calls:
        print(f"  llm_calls    = {total_calls}")
        print(f"  tokens       = {total_tokens}")

    # Markdown report
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Baseline: {args.policy}\n\n")
        f.write(f"- tasks: {len(rows)}\n")
        f.write(f"- device: {args.device}\n")
        f.write(f"- success_rate: {success_rate*100:.1f}% ({successes}/{len(rows)})\n")
        f.write(f"- avg_reward: {avg_reward:.3f}\n")
        f.write(f"- avg_steps: {avg_steps:.2f}\n")
        if total_calls:
            f.write(f"- llm_calls: {total_calls}, tokens: {total_tokens}\n")
        f.write("\n| task | tier | success | reward | failure | steps | ops | llm calls | tokens |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['task_id']} | {r['tier']} | {r['success']} | "
                f"{r['final_reward']:.3f} | {r['failure_class'] or ''} | "
                f"{r['total_steps']} | {r['total_patch_ops']} | "
                f"{r['llm_calls']} | {r['tokens']} |\n"
            )
    print(f"\n[wrote] {out_md}")
    print(f"[wrote] {out_jsonl}")

    return 0 if successes == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
