"""Export trajectory JSONL to SFT (observation -> action) pairs.

Usage:
    python scripts/export_sft.py trajectories/baseline_oracle.jsonl
    python scripts/export_sft.py trajectories/*.jsonl -o data/sft_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _observation_to_prompt(obs: dict[str, Any]) -> str:
    parts = [
        f"task_id: {obs.get('task_id', '')}",
        f"tier: {obs.get('tier', '')}",
        f"change_types: {json.dumps(obs.get('change_types', []), ensure_ascii=False)}",
    ]
    if obs.get("semantic_labels"):
        parts.append(f"semantic_labels: {json.dumps(obs['semantic_labels'], ensure_ascii=False)}")
    if obs.get("code_summary"):
        parts.append(f"code_summary: {json.dumps(obs['code_summary'], ensure_ascii=False)}")
    if obs.get("last_tool_result"):
        parts.append(f"last_tool_result: {json.dumps(obs['last_tool_result'], ensure_ascii=False)}")
    if obs.get("hint"):
        parts.append(f"hint: {obs['hint']}")
    return "\n".join(parts)


def _action_to_response(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False)


def extract_sft_pairs(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """One trajectory -> many (observation, action) training rows."""
    task_id = trajectory.get("task_id", "")
    tier = trajectory.get("tier", "")
    pairs: list[dict[str, Any]] = []
    for step in trajectory.get("steps", []):
        action = step.get("action") or {}
        tool = action.get("tool")
        if tool not in {"apply_patch_ops", "propose_patch", "inspect_code_region"}:
            continue
        obs = dict(step.get("observation") or {})
        obs.setdefault("task_id", task_id)
        obs.setdefault("tier", tier)
        pairs.append({
            "task_id": task_id,
            "step_idx": step.get("step_idx"),
            "observation": obs,
            "action": action,
            "prompt": _observation_to_prompt(obs),
            "response": _action_to_response(action),
            "step_reward": step.get("step_reward"),
            "trajectory_success": trajectory.get("success"),
            "agent_model": trajectory.get("agent_model"),
        })
    return pairs


def export_sft(
    input_paths: list[Path],
    output_path: Path,
    *,
    success_only: bool = False,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for inp in input_paths:
            for traj in _iter_jsonl(inp):
                if success_only and not traj.get("success"):
                    continue
                for pair in extract_sft_pairs(traj):
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path, help="Trajectory JSONL file(s)")
    ap.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "data" / "sft_pairs.jsonl")
    ap.add_argument("--success-only", action="store_true")
    args = ap.parse_args()

    n = export_sft(args.inputs, args.output, success_only=args.success_only)
    print(f"[done] wrote {n} SFT pairs -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
